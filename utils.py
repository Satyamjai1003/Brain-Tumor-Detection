"""
Utility functions for Brain Tumor Classification.
"""
import os
import random
import numpy as np
import torch
from collections import Counter


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Class mapping
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


def extract_label_from_filename(filename):
    """Extract the tumor class from a training filename.
    
    Filenames look like: 00000_p (383)_pituitary_tumor.png
    The label is the last part before .png: pituitary_tumor, glioma_tumor, 
    meningioma_tumor, or no_tumor.
    """
    name = os.path.splitext(filename)[0]  # Remove .png
    
    for class_name in CLASS_NAMES:
        if name.endswith(class_name):
            return class_name
    
    raise ValueError(f"Could not extract label from filename: {filename}")


def compute_class_weights(labels, num_classes=NUM_CLASSES):
    """Compute inverse-frequency class weights for imbalanced data."""
    counter = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)
        weights.append(total / (num_classes * count))
    weights = torch.FloatTensor(weights)
    # Normalize so min weight = 1.0
    weights = weights / weights.min()
    return weights


def print_class_distribution(labels):
    """Print the distribution of classes in the dataset."""
    counter = Counter(labels)
    total = len(labels)
    print(f"\n{'Class':<25} {'Count':>6} {'Percentage':>10}")
    print("-" * 45)
    for idx in sorted(counter.keys()):
        name = IDX_TO_CLASS[idx]
        count = counter[idx]
        pct = 100.0 * count / total
        print(f"{name:<25} {count:>6} {pct:>9.1f}%")
    print(f"{'TOTAL':<25} {total:>6} {'100.0%':>10}")
    print()
