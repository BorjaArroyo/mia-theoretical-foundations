# Configuration settings and hyperparameters for the experiments
import torch

def get_device():
    """Get the device to use for computations"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global device instance
device = get_device()

hyperparams = {
    # === Training Parameters ===
    "batch_size": 50 * 50,
    "latent_dim": 128,
    "n_classes": 10,
    "img_shape": (1, 28, 28),
    "lr": 5e-4,
    "beta1": 0.5,
    "beta2": 0.9,
    "samples_per_class": 50,
    "num_d_steps": 50000,  # Total number of discriminator updates

    # === Adaptive Scheduling ===
    "ds": True,  # Enable adaptive D-step scheduling
    "d_steps_per_g_step": 1,  # Initial D-steps per G-step
    "ds_beta": 0.99,  # EMA smoothing factor for D-step scheduling
    "thresh": 0.65,  # If loss < thresh, increase D-steps
    "grace": 1,  # Grace period before adjusting D-steps
    "max_d_steps": 50,  # Upper limit for D-steps

    # === Additional Parameters ===
    "num_runs": 5,  # Number of runs to estimate epsilon
}

hyperparams["max_physical_batch_size"] = .8 * hyperparams["batch_size"]