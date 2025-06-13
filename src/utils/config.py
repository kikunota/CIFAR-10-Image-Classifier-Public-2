import json
import os
from typing import Dict, Any

class Config:
    def __init__(self, config_dict: Dict[str, Any] = None):
        if config_dict is None:
            config_dict = {}
        
        self.model = config_dict.get('model', {})
        self.training = config_dict.get('training', {})
        self.data = config_dict.get('data', {})
        self.inference = config_dict.get('inference', {})
        self.logging = config_dict.get('logging', {})
        
        self._set_defaults()
    
    def _set_defaults(self):
        # Model defaults
        self.model.setdefault('name', 'resnet18')
        self.model.setdefault('pretrained', True)
        self.model.setdefault('num_classes', 10)
        
        # Training defaults
        self.training.setdefault('epochs', 100)
        self.training.setdefault('batch_size', 32)
        self.training.setdefault('learning_rate', 0.001)
        self.training.setdefault('weight_decay', 1e-4)
        self.training.setdefault('momentum', 0.9)
        self.training.setdefault('scheduler', 'StepLR')
        self.training.setdefault('scheduler_params', {'step_size': 30, 'gamma': 0.1})
        self.training.setdefault('early_stopping_patience', 10)
        self.training.setdefault('save_best_only', True)
        
        # Data defaults
        self.data.setdefault('image_size', 224)
        self.data.setdefault('num_workers', 4)
        self.data.setdefault('pin_memory', True)
        self.data.setdefault('augmentation', True)
        
        # Inference defaults
        self.inference.setdefault('batch_size', 16)
        self.inference.setdefault('tta', False)  # Test Time Augmentation
        
        # Logging defaults
        self.logging.setdefault('log_interval', 10)
        self.logging.setdefault('save_checkpoint_interval', 10)
        self.logging.setdefault('log_dir', 'logs')
        self.logging.setdefault('checkpoint_dir', 'checkpoints')
    
    def to_dict(self):
        return {
            'model': self.model,
            'training': self.training,
            'data': self.data,
            'inference': self.inference,
            'logging': self.logging
        }
    
    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_file(cls, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def update(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict) and isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)

def get_config(config_path: str = None) -> Config:
    if config_path and os.path.exists(config_path):
        return Config.from_file(config_path)
    else:
        return Config()

def create_default_config(save_path: str = 'configs/default.json'):
    config = Config()
    config.save(save_path)
    return config

# Example configurations for different scenarios
def get_quick_train_config():
    config_dict = {
        'model': {
            'name': 'resnet18',
            'pretrained': True
        },
        'training': {
            'epochs': 10,
            'batch_size': 64,
            'learning_rate': 0.01
        },
        'data': {
            'image_size': 224,
            'num_workers': 2
        }
    }
    return Config(config_dict)

def get_production_config():
    config_dict = {
        'model': {
            'name': 'resnet50',
            'pretrained': True
        },
        'training': {
            'epochs': 200,
            'batch_size': 32,
            'learning_rate': 0.001,
            'scheduler': 'CosineAnnealingLR',
            'scheduler_params': {'T_max': 200}
        },
        'data': {
            'image_size': 224,
            'num_workers': 8,
            'augmentation': True
        }
    }
    return Config(config_dict)