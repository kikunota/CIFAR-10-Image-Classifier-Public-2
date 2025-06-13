#!/usr/bin/env python3
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def setup_cifar10():
    print("Setting up CIFAR-10 dataset...")
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create data directory
    data_dir = './data/cifar10'
    os.makedirs(data_dir, exist_ok=True)
    
    # Download CIFAR-10
    print("Downloading CIFAR-10...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transforms.ToTensor()
    )
    
    print(f"Dataset downloaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {classes}")
    
    # Create sample visualization
    print("Creating sample visualization...")
    
    # Collect 5 samples from each class
    class_samples = {i: [] for i in range(10)}
    
    for idx, (image, label) in enumerate(train_dataset):
        if len(class_samples[label]) < 5:
            class_samples[label].append((image, label))
        
        # Stop when we have enough samples
        if all(len(samples) >= 5 for samples in class_samples.values()):
            break
    
    # Create the visualization
    fig, axes = plt.subplots(10, 5, figsize=(15, 30))
    fig.suptitle('CIFAR-10 Dataset - 5 Samples per Class', fontsize=16, fontweight='bold')
    
    for class_idx in range(10):
        for sample_idx in range(5):
            if sample_idx < len(class_samples[class_idx]):
                image, label = class_samples[class_idx][sample_idx]
                
                # Convert tensor to numpy and transpose (C, H, W) -> (H, W, C)
                img_np = image.permute(1, 2, 0).numpy()
                
                axes[class_idx, sample_idx].imshow(img_np)
                axes[class_idx, sample_idx].set_title(f'{classes[label]}', fontsize=10)
            
            axes[class_idx, sample_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_samples.png', dpi=300, bbox_inches='tight')
    print("Sample visualization saved to 'cifar10_samples.png'")
    
    # Print dataset statistics
    print("\n=== CIFAR-10 Dataset Statistics ===")
    print(f"Image size: 32x32 pixels")
    print(f"Number of channels: 3 (RGB)")
    print(f"Number of classes: 10")
    print(f"Training samples: 50,000")
    print(f"Test samples: 10,000")
    print(f"Samples per class: 5,000 (training), 1,000 (test)")
    
    print("\n=== Class Names ===")
    for i, class_name in enumerate(classes):
        print(f"{i}: {class_name}")
    
    return train_dataset, test_dataset, classes

if __name__ == "__main__":
    train_dataset, test_dataset, classes = setup_cifar10()
    print("\nCIFAR-10 setup complete!")