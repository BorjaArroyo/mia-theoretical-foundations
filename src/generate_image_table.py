import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.dataset import get_numerator_dataset
from trainer.dp_trainer import train_cgan_dp
from experiment.config import hyperparams, device
from experiment.visualization import create_comparison_figure

def main():
    target_class = 3  
    sigma_values = [0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]  # Define sigma dynamically
    sigma_values = [0.0, 2.0]  # Define sigma dynamically

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = dset.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataset = get_numerator_dataset(target_class, 0, full_dataset, hyperparams["samples_per_class"])

    # Train a generator for each sigma value
    generators_dict = {}
    for sigma in sigma_values:
        print(f"\nTraining CGAN with sigma={sigma} ...")
        gen, _ = train_cgan_dp(dataset, hyperparams, sigma, device)
        generators_dict[sigma] = gen

    output_path = "comparison_figure.png"
    create_comparison_figure(generators_dict, full_dataset, target_class, sigma_values, hyperparams, device, output_path)

    print("Process completed.")

if __name__ == "__main__":
    main()