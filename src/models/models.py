import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, model_name='resnet18', pretrained=True):
        super(ImageClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        elif model_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        elif model_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            self.backbone.classifier = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if self.model_name in ['resnet18', 'resnet50']:
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.model_name == 'efficientnet_b0':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

class CustomCNN(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_model(model_name, num_classes, pretrained=True):
    if model_name == 'custom_cnn':
        return CustomCNN(num_classes)
    else:
        return ImageClassifier(num_classes, model_name, pretrained)