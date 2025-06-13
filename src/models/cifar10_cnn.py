import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimpleCIFAR10CNN(nn.Module):
    """
    Simple CNN architecture for CIFAR-10 classification.
    
    Architecture:
    - 3 Convolutional blocks with increasing channels
    - Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool
    - Global Average Pooling
    - Fully connected layer for classification
    
    Input: 32x32x3 (CIFAR-10 images)
    Output: 10 classes
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(SimpleCIFAR10CNN, self).__init__()
        
        # First convolutional block
        # Purpose: Extract low-level features (edges, corners)
        # Input: 32x32x3 -> Output: 16x16x32
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB input
            out_channels=32,    # 32 feature maps
            kernel_size=3,      # 3x3 filters
            stride=1,           # Move 1 pixel at a time
            padding=1           # Keep spatial dimensions
        )
        self.bn1 = nn.BatchNorm2d(32)    # Normalize activations
        self.pool1 = nn.MaxPool2d(2, 2)  # Downsample by 2x
        
        # Second convolutional block
        # Purpose: Extract mid-level features (shapes, textures)
        # Input: 16x16x32 -> Output: 8x8x64
        self.conv2 = nn.Conv2d(
            in_channels=32,     # From previous layer
            out_channels=64,    # Double the channels
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        # Purpose: Extract high-level features (objects, parts)
        # Input: 8x8x64 -> Output: 4x4x128
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,   # Further increase channels
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling
        # Purpose: Reduce spatial dimensions to 1x1
        # Replaces large fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final classification layer
        # Purpose: Map features to class probabilities
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First block: Extract basic features
        x = self.conv1(x)           # (B, 3, 32, 32) -> (B, 32, 32, 32)
        x = self.bn1(x)             # Normalize
        x = F.relu(x)               # Non-linearity
        x = self.pool1(x)           # (B, 32, 32, 32) -> (B, 32, 16, 16)
        
        # Second block: Extract intermediate features
        x = self.conv2(x)           # (B, 32, 16, 16) -> (B, 64, 16, 16)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)           # (B, 64, 16, 16) -> (B, 64, 8, 8)
        
        # Third block: Extract high-level features
        x = self.conv3(x)           # (B, 64, 8, 8) -> (B, 128, 8, 8)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)           # (B, 128, 8, 8) -> (B, 128, 4, 4)
        
        # Global pooling and classification
        x = self.global_avg_pool(x) # (B, 128, 4, 4) -> (B, 128, 1, 1)
        x = torch.flatten(x, 1)     # (B, 128, 1, 1) -> (B, 128)
        x = self.dropout(x)         # Regularization
        x = self.fc(x)              # (B, 128) -> (B, 10)
        
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with skip connections.
    
    Purpose: Allows gradients to flow directly through skip connections,
    enabling training of deeper networks without vanishing gradients.
    
    F(x) + x where F(x) is the learned residual function.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path: two convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If dimensions change, use 1x1 conv to match
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

class AdvancedCIFAR10CNN(nn.Module):
    """
    Advanced CNN with residual connections for CIFAR-10.
    
    Improvements over simple CNN:
    1. Residual connections to enable deeper networks
    2. More sophisticated feature extraction
    3. Better gradient flow
    4. Higher capacity for complex patterns
    
    Architecture:
    - Initial convolution
    - 3 groups of residual blocks with increasing channels
    - Global average pooling
    - Classification layer
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(AdvancedCIFAR10CNN, self).__init__()
        
        # Initial convolution
        # Purpose: Transform RGB input to feature maps
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual groups with increasing complexity
        # Group 1: 64 channels, maintain spatial size
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        
        # Group 2: 128 channels, downsample by 2
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        
        # Group 3: 256 channels, downsample by 2
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout and classification
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a group of residual blocks."""
        layers = []
        
        # First block may change dimensions
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks maintain dimensions
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor (batch_size, 3, 32, 32)
            
        Returns:
            Output tensor (batch_size, num_classes)
        """
        # Initial feature extraction
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 3, 32, 32) -> (B, 64, 32, 32)
        
        # Residual groups
        x = self.layer1(x)  # (B, 64, 32, 32) -> (B, 64, 32, 32)
        x = self.layer2(x)  # (B, 64, 32, 32) -> (B, 128, 16, 16)
        x = self.layer3(x)  # (B, 128, 16, 16) -> (B, 256, 8, 8)
        
        # Global pooling and classification
        x = self.global_avg_pool(x)  # (B, 256, 8, 8) -> (B, 256, 1, 1)
        x = torch.flatten(x, 1)      # (B, 256, 1, 1) -> (B, 256)
        x = self.dropout(x)
        x = self.fc(x)               # (B, 256) -> (B, 10)
        
        return x

class CIFAR10ResNet(nn.Module):
    """
    ResNet-style architecture specifically designed for CIFAR-10.
    
    Key differences from ImageNet ResNet:
    - Smaller initial kernel (3x3 instead of 7x7)
    - No initial max pooling (maintains 32x32 resolution)
    - Adapted for smaller input size
    
    This is similar to the ResNet architectures used in the original
    CIFAR-10 experiments.
    """
    
    def __init__(self, num_classes=10):
        super(CIFAR10ResNet, self).__init__()
        
        # Initial convolution (no pooling for small images)
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # ResNet-style layers
        self.layer1 = self._make_layer(16, 16, 3, 1)   # 32x32
        self.layer2 = self._make_layer(16, 32, 3, 2)   # 16x16
        self.layer3 = self._make_layer(32, 64, 3, 2)   # 8x8
        
        # Final pooling and classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_cifar10_model(model_type='simple', num_classes=10, **kwargs):
    """
    Factory function to create CIFAR-10 models.
    
    Args:
        model_type: 'simple', 'advanced', 'resnet', or 'pretrained'
        num_classes: Number of output classes
        **kwargs: Additional arguments for model constructors
        
    Returns:
        PyTorch model instance
    """
    if model_type == 'simple':
        return SimpleCIFAR10CNN(num_classes, **kwargs)
    elif model_type == 'advanced':
        return AdvancedCIFAR10CNN(num_classes, **kwargs)
    elif model_type == 'resnet':
        return CIFAR10ResNet(num_classes)
    elif model_type == 'pretrained':
        # Use pretrained ResNet and adapt for CIFAR-10
        model = models.resnet18(pretrained=True)
        # Modify first layer for 32x32 input
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        # Remove max pooling
        model.maxpool = nn.Identity()
        # Modify final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, input_size=(3, 32, 32)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Create dummy input to trace through model
    dummy_input = torch.randn(1, *input_size)
    
    print("\nLayer-wise output shapes:")
    print("-" * 40)
    
    if hasattr(model, 'conv1'):
        x = dummy_input
        print(f"Input: {tuple(x.shape)}")
        
        # Try to trace through common architectures
        if isinstance(model, SimpleCIFAR10CNN):
            x = F.relu(model.bn1(model.conv1(x)))
            x = model.pool1(x)
            print(f"After conv1+pool1: {tuple(x.shape)}")
            
            x = F.relu(model.bn2(model.conv2(x)))
            x = model.pool2(x)
            print(f"After conv2+pool2: {tuple(x.shape)}")
            
            x = F.relu(model.bn3(model.conv3(x)))
            x = model.pool3(x)
            print(f"After conv3+pool3: {tuple(x.shape)}")
            
            x = model.global_avg_pool(x)
            print(f"After global pooling: {tuple(x.shape)}")
            
            x = torch.flatten(x, 1)
            print(f"After flatten: {tuple(x.shape)}")
            
            x = model.fc(x)
            print(f"Final output: {tuple(x.shape)}")

if __name__ == "__main__":
    # Create and compare different models
    print("=== CIFAR-10 CNN Models Comparison ===\n")
    
    models_to_test = [
        ('simple', 'Simple CNN'),
        ('advanced', 'Advanced CNN with ResNet'),
        ('resnet', 'CIFAR-10 ResNet'),
    ]
    
    for model_type, name in models_to_test:
        print(f"{name}:")
        model = create_cifar10_model(model_type)
        model_summary(model)
        print("\n" + "="*50 + "\n")