#!/usr/bin/env python3
"""
🔥 Train ResNet on CIFAR-10 for Higher Accuracy

Enhanced training script specifically for ResNet models to achieve 75-85% accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from src.data.cifar10_dataset import CIFAR10Dataset
from src.models.cifar10_cnn import create_cifar10_model
from src.utils.utils import AverageMeter, accuracy, save_checkpoint

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, 
                desc=f'Epoch {epoch+1}/{total_epochs} [Train]',
                leave=False)
    
    for i, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{top1.avg:.2f}%'
        })
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device, epoch, total_epochs):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, 
                    desc=f'Epoch {epoch+1}/{total_epochs} [Val]',
                    leave=False)
        
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{top1.avg:.2f}%'
            })
    
    return losses.avg, top1.avg

def main():
    print("🔥 Training ResNet on CIFAR-10 for Higher Accuracy")
    print("=" * 60)
    
    # ResNet-optimized configuration
    config = {
        'model_type': 'resnet',      # Use ResNet architecture
        'epochs': 50,                # More epochs for better convergence
        'batch_size': 32,            # Larger batch size
        'learning_rate': 0.01,       # Higher initial LR for ResNet
        'weight_decay': 5e-4,        # Standard ResNet weight decay
        'momentum': 0.9,             # SGD momentum
        'patience': 15,              # Longer patience for ResNet
        'num_workers': 0,
        'validation_split': 0.1
    }
    
    print(f"🎯 Target Accuracy: 75-85%")
    print(f"📊 Model: CIFAR-10 ResNet")
    print(f"⚡ Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Data loading
    print("📂 Loading CIFAR-10 dataset...")
    cifar10 = CIFAR10Dataset(
        data_dir='./data/cifar10',
        validation_split=config['validation_split'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    train_loader, val_loader, test_loader = cifar10.get_data_loaders()
    class_to_idx = cifar10.get_class_to_idx()
    
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print()
    
    # Model creation
    print(f"🏗️  Creating {config['model_type']} model...")
    model = create_cifar10_model(config['model_type'], num_classes=10)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print()
    
    # Optimized training setup for ResNet
    criterion = nn.CrossEntropyLoss()
    
    # Use SGD with momentum for ResNet (often better than Adam)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Cosine annealing scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    
    # Training tracking
    early_stopping = EarlyStopping(patience=config['patience'])
    best_val_acc = 0.0
    start_time = time.time()
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    print("🎯 Starting ResNet training...")
    print("-" * 60)
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config['epochs']
        )
        
        # Validation
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, config['epochs']
        )
        
        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        learning_rates.append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"Epoch {epoch+1:2d}/{config['epochs']:2d} | "
              f"Train: {train_loss:.4f} ({train_acc:.2f}%) | "
              f"Val: {val_loss:.4f} ({val_acc:.2f}%) | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"  🌟 New best validation accuracy: {val_acc:.2f}%")
        
        # Checkpoint saving
        os.makedirs('checkpoints', exist_ok=True)
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'class_to_idx': class_to_idx,
            'config': config
        }, is_best, 'checkpoints/cifar10_resnet')
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"  🛑 Early stopping after {epoch + 1} epochs")
            break
        
        # Plot progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_training_progress(train_losses, train_accuracies, val_losses, 
                                 val_accuracies, learning_rates, epoch + 1)
    
    # Training completed
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"🎉 ResNet training completed!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Improvement over Simple CNN: +{best_val_acc - 68:.1f}%")
    
    # Final evaluation
    print("\n🧪 Final evaluation on test set...")
    checkpoint = torch.load('checkpoints/cifar10_resnet/model_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, val_loader, criterion, device, 0, 1)
    print(f"  Final test accuracy: {test_acc:.2f}%")
    
    # Save training summary
    summary = {
        'model_type': config['model_type'],
        'total_epochs': epoch + 1,
        'best_val_accuracy': best_val_acc,
        'final_test_accuracy': test_acc,
        'total_training_time_minutes': total_time / 60,
        'total_parameters': total_params,
        'improvement_over_simple_cnn': best_val_acc - 68,
        'config': config
    }
    
    with open('resnet_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    plot_training_progress(train_losses, train_accuracies, val_losses, 
                          val_accuracies, learning_rates, epoch + 1, final=True)
    
    print(f"\n📊 Training summary saved to 'resnet_training_summary.json'")
    print(f"💾 Best model saved to 'checkpoints/cifar10_resnet/model_best.pth'")
    print(f"📈 Training plots saved to 'resnet_training_progress.png'")

def plot_training_progress(train_losses, train_accs, val_losses, val_accs, lrs, epoch, final=False):
    """Plot training progress"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    ax3.plot(epochs, lrs, 'g-', linewidth=2)
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Accuracy improvement
    simple_cnn_acc = 68.0
    improvement = [acc - simple_cnn_acc for acc in val_accs]
    ax4.plot(epochs, improvement, 'purple', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_title('Accuracy Improvement over Simple CNN')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Improvement (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'ResNet Training Progress - Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'resnet_training_progress.png' if final else f'resnet_progress_epoch_{epoch}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Progress plot saved to '{filename}'")

if __name__ == "__main__":
    main()