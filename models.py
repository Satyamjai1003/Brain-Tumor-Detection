"""
Model definitions for Brain Tumor Classification.
Uses timm for pretrained models with custom classification heads.
"""
import torch
import torch.nn as nn
import timm

from utils import NUM_CLASSES


def create_model(model_name, num_classes=NUM_CLASSES, pretrained=True, drop_rate=0.3):
    """
    Create a model from timm with a custom classification head.
    
    Args:
        model_name: One of 'efficientnetv2', 'resnet50', 'densenet121'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained ImageNet weights
        drop_rate: Dropout rate before classifier
    
    Returns:
        PyTorch model
    """
    timm_name_map = {
        'efficientnetv2': 'tf_efficientnetv2_b2',
        'resnet50': 'resnet50',
        'densenet121': 'densenet121',
    }
    
    timm_name = timm_name_map.get(model_name, model_name)
    
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    
    return model


def get_model_names():
    """Return list of model names used in the ensemble."""
    return ['efficientnetv2', 'resnet50', 'densenet121']


class EnsembleModel(nn.Module):
    """Ensemble of multiple models using soft voting."""
    
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            out = model(x)
            out = torch.softmax(out, dim=1)
            outputs.append(out)
        # Average probabilities
        avg_output = torch.stack(outputs, dim=0).mean(dim=0)
        return avg_output
