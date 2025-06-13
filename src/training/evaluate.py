import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.dataset import ImageClassificationDataset
from src.data.transforms import ImageTransforms
from src.models.models import create_model
from src.utils.utils import load_checkpoint

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_accuracies(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracies)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_metrics(y_true, y_pred, y_probs, class_names):
    accuracy = np.mean(y_true == y_pred)
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    results = {
        'overall_accuracy': accuracy,
        'classification_report': report,
        'per_class_accuracy': {}
    }
    
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    for i, class_name in enumerate(class_names):
        results['per_class_accuracy'][class_name] = class_accuracies[i]
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Image Classifier')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file path')
    parser.add_argument('--test-dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    checkpoint = load_checkpoint(args.checkpoint)
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    
    num_classes = len(class_to_idx)
    model = create_model(args.model, num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    test_transform = ImageTransforms.get_test_transforms()
    test_dataset = ImageClassificationDataset(args.test_dir, transform=test_transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print('Evaluating model...')
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device, class_names)
    
    results = calculate_metrics(y_true, y_pred, y_probs, class_names)
    
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print("\nPer-class Accuracy:")
    for class_name, acc in results['per_class_accuracy'].items():
        print(f"  {class_name}: {acc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_confusion_matrix(y_true, y_pred, class_names, 
                         os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    plot_class_accuracies(y_true, y_pred, class_names,
                         os.path.join(args.output_dir, 'class_accuracies.png'))
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == '__main__':
    main()