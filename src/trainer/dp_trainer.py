import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from opacus import PrivacyEngine
from pathlib import Path

from models.generator import Generator
from models.discriminator import Discriminator
from scheduler.adaptive_scheduler import DStepScheduler, ConstDStepScheduler

def train_cgan_dp(dataset, hyperparams, sigma, device, seed, pbar=None, models_dir="saved_models", model_prefix=""):
    """
    Trains a conditional GAN with Differential Privacy (DP) and an adaptive
    discriminator step scheduler without AMP.
    """
    models_path = Path(models_dir) / f"sigma_{sigma}" / model_prefix
    models_path.mkdir(parents=True, exist_ok=True)

    # get file based on run
    model_path = models_path / f"run_{seed}.pt"
    if model_path.exists():
        return torch.load(model_path, weights_only=False)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    dp_dataloader = DataLoader(
        dataset, 
        batch_size=hyperparams["batch_size"], 
        shuffle=True, 
        pin_memory=(device.type == "cuda"),
        num_workers=4,
        persistent_workers=True
    )

    dp_enabled = (sigma > 0)

    generator = Generator(
        hyperparams["latent_dim"],
        hyperparams["n_classes"],
        hyperparams["img_shape"]
    ).to(device)
    discriminator = Discriminator(
        hyperparams["n_classes"],
        hyperparams["img_shape"]
    ).to(device)

    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=hyperparams["lr"],
        betas=(hyperparams["beta1"], hyperparams["beta2"])
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=hyperparams["lr"],
        betas=(hyperparams["beta1"], hyperparams["beta2"])
    )

    # Use BCEWithLogitsLoss for numerical stability (discriminator outputs raw logits)
    criterion = nn.BCELoss()

    # Use the adaptive scheduler
    if hyperparams.get("ds", False):
        scheduler = DStepScheduler(
            d_steps_rate_init=hyperparams["d_steps_per_g_step"],
            grace=hyperparams["grace"],
            thresh=hyperparams["thresh"],
            beta=hyperparams["ds_beta"],
            max_d_steps=hyperparams["max_d_steps"]
        )
    else:
        scheduler = ConstDStepScheduler(
            d_steps_rate=hyperparams["d_steps_per_g_step"]
        )

    # Enable Differential Privacy only if sigma > 0
    # if dp_enabled:
    privacy_engine = PrivacyEngine()
    discriminator, optimizer_D, dp_dataloader = privacy_engine.make_private(
        module=discriminator,
        optimizer=optimizer_D,
        data_loader=dp_dataloader,
        noise_multiplier=sigma,
        max_grad_norm=1.0,
        poisson_sampling=True
    )

    d_step = 0
    g_step = 0
    ema_beta = hyperparams["ds_beta"]  # Momentum for moving average
    ema_loss_D = None  # Exponential moving average for Discriminator Loss
    ema_loss_G = None  # Exponential moving average for Generator Loss

    while d_step < hyperparams["num_d_steps"]:
        for imgs, labels in dp_dataloader:
            current_batch_size = imgs.size(0)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            real_labels = torch.ones(current_batch_size, 1, device=device)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)

            # --- Discriminator Update ---
            d_steps = scheduler.get_d_steps_rate()
            for _ in range(d_steps):
                optimizer_D.zero_grad()
                
                # Forward pass for real images
                real_validity = discriminator(imgs, labels)
                loss_real = criterion(real_validity, real_labels)
                    
                # Forward pass for fake images
                noise = torch.randn(current_batch_size, hyperparams["latent_dim"], device=device)
                random_labels = torch.randint(0, hyperparams["n_classes"], (current_batch_size,), device=device)
                fake_imgs = generator(noise, random_labels)
                fake_validity = discriminator(fake_imgs.detach(), random_labels)
                loss_fake = criterion(fake_validity, fake_labels)
                    
                loss_D = loss_real + loss_fake
                loss_D.backward()
                optimizer_D.step()

                d_step += 1
                scheduler.d_step()
                if pbar:
                    pbar.update(1)

                ema_loss_D = loss_D.item() if ema_loss_D is None else (
                    ema_beta * ema_loss_D + (1 - ema_beta) * loss_D.item()
                )

            # --- Generator Update ---
            if scheduler.is_g_step_time():
                optimizer_G.zero_grad()
                noise = torch.randn(current_batch_size, hyperparams["latent_dim"], device=device)
                random_labels = torch.randint(0, hyperparams["n_classes"], (current_batch_size,), device=device)

                # Temporarily disable DP hooks during generator update if DP is enabled
                # if dp_enabled:
                discriminator.disable_hooks()

                gen_validity = discriminator(generator(noise, random_labels), random_labels)
                loss_G = criterion(gen_validity, real_labels)
                loss_G.backward()
                optimizer_G.step()

                # if dp_enabled:
                discriminator.enable_hooks()

                g_step += 1
                scheduler.g_step(loss_real.item())
                ema_loss_G = loss_G.item() if ema_loss_G is None else (
                    ema_beta * ema_loss_G + (1 - ema_beta) * loss_G.item()
                )

            if pbar:
                pbar.set_postfix({
                    "loss_D": f"{ema_loss_D:.4f}" if ema_loss_D is not None else "N/A",
                    "loss_G": f"{ema_loss_G:.4f}" if ema_loss_G is not None else "N/A",
                    "d_steps": scheduler.get_d_steps_rate()
                })

            if d_step >= hyperparams["num_d_steps"]:
                break
    if pbar:
        pbar.reset()

    torch.save((generator, discriminator), model_path)

    return generator, discriminator