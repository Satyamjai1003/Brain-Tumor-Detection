"""
Prediction script for Brain Tumor Classification.
Loads trained ensemble, applies TTA, and generates submission.csv.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from utils import set_seed, IDX_TO_CLASS, NUM_CLASSES, CLASS_NAMES
from dataset import BrainTumorDataset, get_val_transforms, get_tta_transforms, load_test_data
from models import create_model, get_model_names


# ======================== CONFIG ========================
TEST_DIR = os.path.join('data', 'test', 'test')
CHECKPOINT_DIR = 'checkpoints_model'
BATCH_SIZE = 16
SEED = 42
USE_TTA = False
OUTPUT_FILE = 'submission.csv'
DETAILED_OUTPUT_FILE = 'detailed_predictions.csv'
# ========================================================


def predict_with_model(model, loader, device):
    """Run prediction for a single model, returns probabilities."""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    return np.concatenate(all_probs, axis=0)


def main():
    set_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load test data
    test_paths, test_filenames = load_test_data(TEST_DIR)
    
    # Create test datasets
    test_dataset_normal = BrainTumorDataset(test_paths, labels=None, transform=get_val_transforms())
    test_loader_normal = DataLoader(test_dataset_normal, batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=0, pin_memory=False)
    
    if USE_TTA:
        test_dataset_flip = BrainTumorDataset(test_paths, labels=None, transform=get_tta_transforms())
        test_loader_flip = DataLoader(test_dataset_flip, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=0, pin_memory=False)
    
    # Ensemble prediction
    model_names = get_model_names()
    all_probs = np.zeros((len(test_paths), NUM_CLASSES))
    n_models = 0
    model_preds = {}
    
    for model_name in model_names:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_best.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠ Checkpoint not found: {checkpoint_path}, skipping {model_name}")
            continue
        
        print(f"\nLoading {model_name}...")
        model = create_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print(f"  Best val F1: {checkpoint.get('val_f1', 'N/A')}")
        print(f"  Best epoch:  {checkpoint.get('epoch', 'N/A')}")
        
        # Normal prediction
        probs = predict_with_model(model, test_loader_normal, device)
        
        if USE_TTA:
            # TTA: horizontal flip
            probs_flip = predict_with_model(model, test_loader_flip, device)
            probs = (probs + probs_flip) / 2.0
        
        model_preds[model_name] = np.argmax(probs, axis=1)
        all_probs += probs
        n_models += 1
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    if n_models == 0:
        print("ERROR: No model checkpoints found! Run train.py first.")
        return
    
    # Average across models
    all_probs /= n_models
    print(f"\nEnsembled {n_models} models")
    
    # Get predictions
    predictions = np.argmax(all_probs, axis=1)
    pred_labels = [IDX_TO_CLASS[p] for p in predictions]
    confidence_scores = np.max(all_probs, axis=1)
    
    # Calculate agreement
    agreement_counts = []
    for i in range(len(test_paths)):
        votes = [preds[i] for preds in model_preds.values()]
        matches = sum(1 for vote in votes if vote == predictions[i])
        if matches == n_models:
            agreement_counts.append("Unanimous")
        elif matches > n_models / 2:
            agreement_counts.append("Majority")
        else:
            agreement_counts.append("Split Vote")
            
    # Create detailed submission
    detailed_df = pd.DataFrame({
        'image_id': test_filenames,
        'final_label': pred_labels,
        'confidence_score': np.round(confidence_scores, 4),
        'model_agreement': agreement_counts,
    })
    
    # Add individual class probabilities
    for i, class_name in IDX_TO_CLASS.items():
        detailed_df[f'prob_{class_name}'] = np.round(all_probs[:, i], 4)
        
    # Add individual model predictions
    for model_name, preds in model_preds.items():
        detailed_df[f'{model_name}_pred'] = [IDX_TO_CLASS[p] for p in preds]
        
    detailed_df.to_csv(DETAILED_OUTPUT_FILE, index=False)
    print(f"\n✓ Detailed report saved to: {os.path.abspath(DETAILED_OUTPUT_FILE)}")
    
    # Create final test submission
    submission = pd.DataFrame({
        'image_id': test_filenames,
        'label': pred_labels,
    })
    
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Submission saved to: {os.path.abspath(OUTPUT_FILE)}")
    print(f"  Total predictions: {len(submission)}")
    
    # Validate submission format
    print(f"\n--- Submission Validation ---")
    print(f"Columns: {list(submission.columns)}")
    print(f"Rows: {len(submission)}")
    print(f"Unique labels: {sorted(submission['label'].unique())}")
    print(f"Label distribution:")
    for label in CLASS_NAMES:
        count = (submission['label'] == label).sum()
        pct = 100.0 * count / len(submission)
        print(f"  {label:<25} {count:>4} ({pct:.1f}%)")
    
    # Show first few rows
    print(f"\nFirst 5 rows:")
    print(submission.head().to_string(index=False))
    
    # Check all test files are included
    expected_count = len(test_filenames)
    actual_count = len(submission)
    if actual_count == expected_count:
        print(f"\n✓ All {expected_count} test images included")
    else:
        print(f"\n⚠ Expected {expected_count} images but got {actual_count}")


if __name__ == '__main__':
    main()
