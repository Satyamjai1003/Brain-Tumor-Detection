"""
Dataset and data loading for Brain Tumor Classification.
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from utils import extract_label_from_filename, CLASS_TO_IDX, NUM_CLASSES, print_class_distribution


class BrainTumorDataset(Dataset):
    """PyTorch Dataset for brain MRI images with 4-class labels."""

    def __init__(self, image_paths, labels=None, transform=None):
        """
        Args:
            image_paths: List of absolute paths to images.
            labels: List of integer labels (None for test set).
            transform: torchvision transforms to apply.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


def get_train_transforms():
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms():
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transforms():
    """Test-Time Augmentation transforms (horizontal flip)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),  # Always flip
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_train_data(train_dir, val_split=0.15, seed=42):
    """
    Load training data, extract labels from filenames, and split into train/val.
    
    Returns:
        train_paths, val_paths, train_labels, val_labels
    """
    all_files = sorted([f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    all_paths = []
    all_labels = []
    
    for fname in all_files:
        fpath = os.path.join(train_dir, fname)
        label_name = extract_label_from_filename(fname)
        label_idx = CLASS_TO_IDX[label_name]
        all_paths.append(fpath)
        all_labels.append(label_idx)
    
    print(f"Loaded {len(all_paths)} training images")
    print_class_distribution(all_labels)
    
    # Stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, 
        test_size=val_split, 
        stratify=all_labels, 
        random_state=seed
    )
    
    print(f"Train split: {len(train_paths)} images")
    print(f"Val split:   {len(val_paths)} images")
    
    return train_paths, val_paths, train_labels, val_labels


def load_test_data(test_dir):
    """Load test data (no labels)."""
    all_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    test_paths = [os.path.join(test_dir, f) for f in all_files]
    test_filenames = all_files
    
    print(f"Loaded {len(test_paths)} test images")
    return test_paths, test_filenames
