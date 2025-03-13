import torch
import cupy as cp  # RAPIDS uses CuPy for GPU arrays
import numpy as np
from cuml import PCA
from cuml.neighbors import KernelDensity

def estimate_log_density(generator, latent_dim, class_list, x_q, device, n_samples=10000, n_components=4):
    """
    Estimate the log density of x_q using RAPIDS-accelerated Kernel Density Estimation (KDE) 
    with samples generated from multiple classes.

    Args:
        generator: The GAN generator model.
        latent_dim: Dimensionality of the latent space.
        class_list: List of class labels to generate from.
        x_q: Query image (tensor).
        device: Device to run computations on (CPU/GPU).
        n_samples: Total number of samples to generate.
        n_components: Number of PCA components for dimensionality reduction.

    Returns:
        log_density: Log-likelihood of x_q under the RAPIDS KDE model.
    """
    n_classes = len(class_list)
    samples_per_class = n_samples // n_classes  # Equal samples per class

    all_gen_imgs = []
    
    for class_label in class_list:
        noise = torch.randn(samples_per_class, latent_dim, device=device)
        labels = torch.full((samples_per_class,), class_label, device=device, dtype=torch.long)

        with torch.no_grad():
            gen_imgs = generator(noise, labels)

        gen_imgs = gen_imgs.cpu().numpy().reshape(samples_per_class, -1)  # Flatten
        all_gen_imgs.append(gen_imgs)

    # 🔹 Concatenate all generated samples from different classes
    gen_imgs_all = np.vstack(all_gen_imgs)

    # 🔹 Convert NumPy arrays to CuPy for GPU acceleration
    gen_imgs_gpu = cp.asarray(gen_imgs_all)
    x_q_gpu = cp.asarray(x_q.cpu().numpy().reshape(1, -1))

    # 🔹 GPU-accelerated PCA using cuML
    pca = PCA(n_components=n_components, svd_solver='full')
    gen_imgs_pca = pca.fit_transform(gen_imgs_gpu)
    x_q_pca = pca.transform(x_q_gpu)

    # 🔹 Fit RAPIDS GPU KDE with optimized bandwidth
    kde = KernelDensity(kernel='gaussian', bandwidth=1.)
    kde.fit(gen_imgs_pca)

    # 🔹 Compute Log Density
    log_density = kde.score_samples(x_q_pca)[0].item()
    
    return log_density
