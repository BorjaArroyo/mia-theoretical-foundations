import gc
import json
from tqdm import tqdm
import torch
from torch.utils.data import Subset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.kde import estimate_log_density
from data.dataset import get_numerator_dataset, get_denominator_dataset
from experiment.config import hyperparams, device
from trainer.dp_trainer import train_cgan_dp
import warnings
warnings.filterwarnings("ignore")

def estimate_epsilon_across_sigmas(sigmas, classes, hyperparams, output_filename):
    samples_per_class = hyperparams["samples_per_class"]
    num_runs = hyperparams["num_runs"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = dset.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    sigma_results = {}
    
    pbar = tqdm(sigmas, desc="Sigma values", unit="sigma", ncols=100)
    for sigma in pbar:
        epsilons_sigma = []
        for target_class in tqdm(classes, desc=f"Target classes (sigma={sigma})", leave=False, ncols=100):
            target_indices = [i for i, (_, label) in enumerate(full_dataset) if label == target_class]
            if len(target_indices) == 0:
                continue
            query_index = target_indices[0]
            x_q, label_q = full_dataset[query_index]
            x_q = x_q.to(device)
                        
            # Agregamos el target_label y las imágenes reales para FID al diccionario de hyperparams
            hyperparams["target_label"] = label_q
            
            numerator_dataset = get_numerator_dataset(target_class, query_index, full_dataset, samples_per_class)
            denominator_dataset = get_denominator_dataset(target_class, full_dataset, samples_per_class)
            
            epsilon_runs = []
            pbar_runs = tqdm(range(num_runs), desc="Runs", leave=False, ncols=100)
            pbar_train = tqdm(total=hyperparams["num_d_steps"], desc=f"Training (sigma={sigma})", ncols=120, leave=False)
            for run in pbar_runs:
                # Entrenamos victim y reference con verificación del FID
                generator_victim, discriminator_victim = train_cgan_dp(numerator_dataset, hyperparams, sigma, device, run, pbar_train, model_prefix="numerator")
                generator_reference, discriminator_ref = train_cgan_dp(denominator_dataset, hyperparams, sigma, device, run, pbar_train, model_prefix="denominator")
                
                log_density_v = estimate_log_density(generator_victim, hyperparams["latent_dim"], classes, x_q, device)
                log_density_r = estimate_log_density(generator_reference, hyperparams["latent_dim"], classes, x_q, device)
                
                epsilon_run = log_density_v - log_density_r
                epsilon_runs.append(epsilon_run)
                del generator_victim, generator_reference, discriminator_victim, discriminator_ref
                torch.cuda.empty_cache()
                gc.collect()
            if epsilon_runs:
                epsilon_target = max(epsilon_runs)
                epsilons_sigma.append(epsilon_target)
        if epsilons_sigma:
            max_epsilon = max(epsilons_sigma)
            sigma_results[sigma] = max_epsilon
        else:
            sigma_results[sigma] = None
        
        pbar.set_postfix({"max_epsilon": sigma_results[sigma] if sigma_results[sigma] is not None else "N/A"})
        with open(output_filename, "w") as fp:
            json.dump(sigma_results, fp, indent=4)
    return sigma_results


if __name__ == "__main__":
    sigmas = [0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0][::-1]
    sigmas = [0.0, 0.5, 1.0, 1.5, 2.0]
    classes = list(range(10))
    classes = [0]
    
    output_filename = "epsilon_results.json"
    results = estimate_epsilon_across_sigmas(sigmas, classes, hyperparams, output_filename)
    
    print("\nFinal Results (max epsilon per sigma):")
    for sigma, eps in results.items():
        if eps is not None:
            print(f"Sigma {sigma}: max epsilon = {eps:.4f}")
        else:
            print(f"Sigma {sigma}: no result")