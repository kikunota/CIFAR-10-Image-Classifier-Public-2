#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.cifar10_dataset import CIFAR10Dataset
from src.models.cifar10_cnn import create_cifar10_model
from src.utils.utils import AverageMeter, accuracy, save_checkpoint

class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience=7, min_delta=0):
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

class TrainingLogger:
    """Logger for training metrics and plotting."""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Log metrics for one epoch."""
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        
    def plot_metrics(self, save_path='training_progress.png'):
        """Create and save training progress plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CIFAR-10 CNN Training Progress', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training & Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training & Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax3.plot(epochs, self.learning_rates, 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Loss difference plot
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
        ax4.plot(epochs, loss_diff, 'purple', linewidth=2)
        ax4.set_title('Train-Validation Loss Difference')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('|Train Loss - Val Loss|')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved to {save_path}")
        
        return fig

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train the model for one epoch."""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # Progress bar
    pbar = tqdm(train_loader, 
                desc=f'Epoch {epoch+1}/{total_epochs} [Train]',
                leave=False)
    
    for i, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Calculate accuracy
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{top1.avg:.2f}%'
        })
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device, epoch, total_epochs):
    """Validate the model."""
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
    print("🚀 Starting CIFAR-10 CNN Training")
    print("=" * 50)
    
    # CPU-optimized settings
    config = {
        'model_type': 'simple',  # Start with simple model for CPU
        'epochs': 10,            # Initial test with 10 epochs
        'batch_size': 16,        # Small batch size for CPU
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'patience': 5,           # Early stopping patience
        'num_workers': 0,        # 0 for CPU to avoid multiprocessing issues
        'validation_split': 0.1
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Set device (CPU for this example)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create data loaders
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
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print()
    
    # Create model
    print(f"🏗️  Creating {config['model_type']} CNN model...")
    model = create_cifar10_model(config['model_type'], num_classes=10)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")
    print()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Initialize tracking
    logger = TrainingLogger()
    early_stopping = EarlyStopping(patience=config['patience'])
    best_val_acc = 0.0
    start_time = time.time()
    
    print("🎯 Starting training...")
    print("-" * 50)
    
    # Training loop
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config['epochs']
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, config['epochs']
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"Epoch {epoch+1:2d}/{config['epochs']:2d} | "
              f"Train: {train_loss:.4f} ({train_acc:.2f}%) | "
              f"Val: {val_loss:.4f} ({val_acc:.2f}%) | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"  ✨ New best validation accuracy: {val_acc:.2f}%")
        
        # Save checkpoint
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
        }, is_best, 'checkpoints/cifar10_simple_cnn')
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"  🛑 Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Create progress plot every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == config['epochs'] - 1:
            logger.plot_metrics(f'training_progress_epoch_{epoch+1}.png')
    
    # Training completed
    total_time = time.time() - start_time
    print("-" * 50)
    print(f"🎉 Training completed!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Final learning rate: {current_lr:.6f}")
    
    # Create final plots
    logger.plot_metrics('final_training_progress.png')
    
    # Test the best model
    print("\n🧪 Testing best model on validation set...")
    
    # Load best model
    checkpoint = torch.load('checkpoints/cifar10_simple_cnn/model_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    test_loss, test_acc = validate(model, val_loader, criterion, device, 0, 1)
    print(f"  Test accuracy: {test_acc:.2f}%")
    
    # Save training summary
    summary = {
        'model_type': config['model_type'],
        'total_epochs': epoch + 1,
        'best_val_accuracy': best_val_acc,
        'final_test_accuracy': test_acc,
        'total_training_time_minutes': total_time / 60,
        'total_parameters': total_params,
        'config': config
    }
    
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Training summary saved to 'training_summary.json'")
    print(f"📊 Final plots saved to 'final_training_progress.png'")
    print(f"💾 Best model saved to 'checkpoints/cifar10_simple_cnn/model_best.pth'")

if __name__ == "__main__":
    main()