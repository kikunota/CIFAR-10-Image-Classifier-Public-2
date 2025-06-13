import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.dataset import get_data_loaders
from src.models.models import create_model
from src.utils.utils import save_checkpoint, load_checkpoint, AverageMeter, accuracy
from src.utils.config import get_config

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
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
            'Acc@1': f'{top1.avg:.3f}'
        })
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
    
    return losses.avg, top1.avg

def main():
    parser = argparse.ArgumentParser(description='Train Image Classifier')
    parser.add_argument('--config', type=str, default='configs/default.json', help='Config file path')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    config = get_config(args.config) if os.path.exists(args.config) else {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    train_loader, val_loader, class_to_idx = get_data_loaders(
        train_dir, val_dir, batch_size=args.batch_size
    )
    
    num_classes = len(class_to_idx)
    model = create_model(args.model, num_classes, pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    training_log = []
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}%')
        
        training_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'class_to_idx': class_to_idx
        }, is_best, f'checkpoints/{args.model}')
        
        with open(f'logs/training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(training_log, f, indent=2)
    
    print(f'Training completed. Best validation accuracy: {best_acc:.3f}%')

if __name__ == '__main__':
    main()