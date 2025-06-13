#!/usr/bin/env python3
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cifar10_cnn import create_cifar10_model, count_parameters

def visualize_model_comparison():
    """Create visualization comparing different CNN architectures."""
    
    models_info = []
    
    # Define models to compare
    model_configs = [
        ('simple', 'Simple CNN\n(3 Conv Blocks)', 'lightblue'),
        ('advanced', 'Advanced CNN\n(ResNet Blocks)', 'lightgreen'),
        ('resnet', 'CIFAR-10 ResNet\n(Lightweight)', 'lightcoral'),
    ]
    
    print("Analyzing CNN architectures for CIFAR-10...")
    
    for model_type, label, color in model_configs:
        model = create_cifar10_model(model_type)
        params = count_parameters(model)
        
        # Calculate theoretical FLOPs (simplified estimation)
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            _ = model(dummy_input)
        
        models_info.append({
            'name': label,
            'type': model_type,
            'parameters': params,
            'color': color,
            'model': model
        })
        
        print(f"{label.replace(chr(10), ' ')}: {params:,} parameters")
    
    # Create comparison visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CIFAR-10 CNN Architectures Comparison', fontsize=16, fontweight='bold')
    
    # 1. Parameter count comparison
    names = [info['name'] for info in models_info]
    params = [info['parameters'] for info in models_info]
    colors = [info['color'] for info in models_info]
    
    bars1 = ax1.bar(names, params, color=colors, alpha=0.7)
    ax1.set_title('Model Complexity (Parameters)')
    ax1.set_ylabel('Number of Parameters')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, param in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.01,
                f'{param:,}', ha='center', va='bottom', fontsize=10)
    
    # 2. Architecture depth comparison
    depths = [3, 8, 9]  # Approximate layer depths
    bars2 = ax2.bar(names, depths, color=colors, alpha=0.7)
    ax2.set_title('Network Depth (Layers)')
    ax2.set_ylabel('Number of Layers')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, depth in zip(bars2, depths):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{depth}', ha='center', va='bottom', fontsize=10)
    
    # 3. Memory footprint (estimated)
    # Rough estimation based on parameters and intermediate activations
    memory_mb = [p * 4 / (1024**2) * 2 for p in params]  # 4 bytes per float, factor for activations
    bars3 = ax3.bar(names, memory_mb, color=colors, alpha=0.7)
    ax3.set_title('Estimated Memory Usage')
    ax3.set_ylabel('Memory (MB)')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, mem in zip(bars3, memory_mb):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory_mb)*0.01,
                f'{mem:.1f}MB', ha='center', va='bottom', fontsize=10)
    
    # 4. Feature extraction capability (qualitative)
    capabilities = [6, 9, 8]  # Subjective scores out of 10
    bars4 = ax4.bar(names, capabilities, color=colors, alpha=0.7)
    ax4.set_title('Feature Extraction Capability')
    ax4.set_ylabel('Capability Score (1-10)')
    ax4.set_ylim(0, 10)
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, cap in zip(bars4, capabilities):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{cap}/10', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('cifar10_models_comparison.png', dpi=300, bbox_inches='tight')
    print("Model comparison saved to 'cifar10_models_comparison.png'")
    
    return models_info

def create_architecture_diagram():
    """Create a visual diagram of the Simple CNN architecture."""
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Define layer information
    layers = [
        {'name': 'Input\n32×32×3', 'pos': (1, 4), 'size': (1.5, 2), 'color': 'lightgray'},
        {'name': 'Conv1\n32×32×32', 'pos': (3, 4), 'size': (1.5, 2), 'color': 'lightblue'},
        {'name': 'Pool1\n16×16×32', 'pos': (5, 4), 'size': (1.5, 2), 'color': 'lightblue'},
        {'name': 'Conv2\n16×16×64', 'pos': (7, 4), 'size': (1.5, 2), 'color': 'lightgreen'},
        {'name': 'Pool2\n8×8×64', 'pos': (9, 4), 'size': (1.5, 2), 'color': 'lightgreen'},
        {'name': 'Conv3\n8×8×128', 'pos': (11, 4), 'size': (1.5, 2), 'color': 'lightcoral'},
        {'name': 'Pool3\n4×4×128', 'pos': (13, 4), 'size': (1.5, 2), 'color': 'lightcoral'},
        {'name': 'Global\nAvg Pool\n1×1×128', 'pos': (15, 4), 'size': (1.5, 2), 'color': 'lightyellow'},
        {'name': 'FC\n10 classes', 'pos': (17, 4), 'size': (1.5, 2), 'color': 'lightpink'},
    ]
    
    # Draw layers
    for layer in layers:
        rect = plt.Rectangle(layer['pos'], layer['size'][0], layer['size'][1], 
                           facecolor=layer['color'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Add text
        ax.text(layer['pos'][0] + layer['size'][0]/2, layer['pos'][1] + layer['size'][1]/2,
               layer['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    for i in range(len(layers) - 1):
        start_x = layers[i]['pos'][0] + layers[i]['size'][0]
        end_x = layers[i+1]['pos'][0]
        y = layers[i]['pos'][1] + layers[i]['size'][1]/2
        
        ax.annotate('', xy=(end_x, y), xytext=(start_x, y), arrowprops=arrow_props)
    
    # Add layer explanations
    explanations = [
        "RGB Image Input",
        "Extract Low-level Features\n(edges, corners)",
        "Reduce Spatial Size",
        "Extract Mid-level Features\n(shapes, textures)", 
        "Reduce Spatial Size",
        "Extract High-level Features\n(objects, patterns)",
        "Reduce Spatial Size",
        "Reduce to Single Vector",
        "Classify into 10 Classes"
    ]
    
    for i, (layer, explanation) in enumerate(zip(layers, explanations)):
        ax.text(layer['pos'][0] + layer['size'][0]/2, layer['pos'][1] - 0.5,
               explanation, ha='center', va='top', fontsize=9, style='italic')
    
    ax.set_xlim(0, 19)
    ax.set_ylim(1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Simple CNN Architecture for CIFAR-10', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('simple_cnn_architecture.png', dpi=300, bbox_inches='tight')
    print("Architecture diagram saved to 'simple_cnn_architecture.png'")

def explain_residual_connections():
    """Create a visual explanation of residual connections."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Standard CNN block
    ax1.set_title('Standard CNN Block', fontsize=14, fontweight='bold')
    
    # Draw standard block
    layers_std = [
        {'name': 'Input\nX', 'pos': (1, 3), 'size': (1.5, 1), 'color': 'lightgray'},
        {'name': 'Conv\nBN\nReLU', 'pos': (1, 2), 'size': (1.5, 0.8), 'color': 'lightblue'},
        {'name': 'Conv\nBN', 'pos': (1, 1), 'size': (1.5, 0.8), 'color': 'lightblue'},
        {'name': 'ReLU', 'pos': (1, 0.2), 'size': (1.5, 0.6), 'color': 'lightgreen'},
        {'name': 'Output', 'pos': (1, -0.5), 'size': (1.5, 0.6), 'color': 'lightgray'},
    ]
    
    for layer in layers_std:
        rect = plt.Rectangle(layer['pos'], layer['size'][0], layer['size'][1], 
                           facecolor=layer['color'], edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(layer['pos'][0] + layer['size'][0]/2, layer['pos'][1] + layer['size'][1]/2,
               layer['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows for standard block
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    for i in range(len(layers_std) - 1):
        start_y = layers_std[i]['pos'][1]
        end_y = layers_std[i+1]['pos'][1] + layers_std[i+1]['size'][1]
        x = layers_std[i]['pos'][0] + layers_std[i]['size'][0]/2
        ax1.annotate('', xy=(x, end_y), xytext=(x, start_y), arrowprops=arrow_props)
    
    ax1.set_xlim(0, 3.5)
    ax1.set_ylim(-1, 4.5)
    ax1.axis('off')
    
    # Residual CNN block
    ax2.set_title('Residual Block (ResNet)', fontsize=14, fontweight='bold')
    
    # Draw residual block
    layers_res = [
        {'name': 'Input\nX', 'pos': (1, 3), 'size': (1.5, 1), 'color': 'lightgray'},
        {'name': 'Conv\nBN\nReLU', 'pos': (1, 2), 'size': (1.5, 0.8), 'color': 'lightblue'},
        {'name': 'Conv\nBN', 'pos': (1, 1), 'size': (1.5, 0.8), 'color': 'lightblue'},
        {'name': 'Add', 'pos': (1, 0.2), 'size': (1.5, 0.6), 'color': 'yellow'},
        {'name': 'ReLU', 'pos': (1, -0.3), 'size': (1.5, 0.4), 'color': 'lightgreen'},
        {'name': 'Output\nF(X) + X', 'pos': (1, -0.8), 'size': (1.5, 0.4), 'color': 'lightgray'},
    ]
    
    for layer in layers_res:
        rect = plt.Rectangle(layer['pos'], layer['size'][0], layer['size'][1], 
                           facecolor=layer['color'], edgecolor='black', linewidth=1)
        ax2.add_patch(rect)
        ax2.text(layer['pos'][0] + layer['size'][0]/2, layer['pos'][1] + layer['size'][1]/2,
               layer['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw main path arrows
    for i in range(len(layers_res) - 1):
        if i == 2:  # Skip the Add layer connection
            continue
        start_y = layers_res[i]['pos'][1]
        end_y = layers_res[i+1]['pos'][1] + layers_res[i+1]['size'][1]
        x = layers_res[i]['pos'][0] + layers_res[i]['size'][0]/2
        ax2.annotate('', xy=(x, end_y), xytext=(x, start_y), arrowprops=arrow_props)
    
    # Draw skip connection
    skip_props = dict(arrowstyle='->', lw=3, color='red')
    ax2.annotate('', xy=(2.8, 0.5), xytext=(2.8, 3.5), 
                arrowprops=dict(arrowstyle='->', lw=3, color='red', 
                               connectionstyle="arc3,rad=0.3"))
    
    # Add skip connection label
    ax2.text(3.2, 2, 'Skip\nConnection\n(Identity)', ha='left', va='center', 
            fontsize=10, fontweight='bold', color='red')
    
    # Connect conv output to add
    ax2.annotate('', xy=(1.75, 0.8), xytext=(1.75, 1), arrowprops=arrow_props)
    ax2.annotate('', xy=(1.75, 0.2), xytext=(1.75, 0.8), arrowprops=arrow_props)
    
    ax2.set_xlim(0, 4)
    ax2.set_ylim(-1.2, 4.5)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('residual_connections_explained.png', dpi=300, bbox_inches='tight')
    print("Residual connections explanation saved to 'residual_connections_explained.png'")

if __name__ == "__main__":
    print("Creating CNN architecture visualizations for CIFAR-10...\n")
    
    # Create model comparison
    models_info = visualize_model_comparison()
    
    print("\nCreating architecture diagrams...")
    create_architecture_diagram()
    
    print("\nCreating residual connections explanation...")
    explain_residual_connections()
    
    print("\n=== Summary ===")
    print("Generated visualizations:")
    print("1. cifar10_models_comparison.png - Model comparison charts")
    print("2. simple_cnn_architecture.png - Simple CNN architecture diagram") 
    print("3. residual_connections_explained.png - Residual connections explanation")
    
    print("\n=== Model Recommendations ===")
    print("• Simple CNN: Good for learning and quick experiments")
    print("• Advanced CNN: Better accuracy with residual connections") 
    print("• CIFAR-10 ResNet: Balanced performance and efficiency")
    print("• For production: Consider Advanced CNN or CIFAR-10 ResNet")