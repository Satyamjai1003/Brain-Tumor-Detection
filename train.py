    """
    Training script for Brain Tumor Classification.
    Trains EfficientNetV2-B2, ResNet50, and DenseNet121 sequentially.
    """
    import os
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.amp import autocast, GradScaler
    from sklearn.metrics import classification_report, f1_score, accuracy_score

    from utils import set_seed, CLASS_NAMES, compute_class_weights, NUM_CLASSES
    from dataset import (
        BrainTumorDataset, get_train_transforms, get_val_transforms, load_train_data
    )
    from models import create_model, get_model_names


    # ======================== CONFIG ========================
    TRAIN_DIR = os.path.join('data', 'train', 'train')
    CHECKPOINT_DIR = 'checkpoints'
    BATCH_SIZE = 16
    NUM_EPOCHS = 35
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 8
    SEED = 42
    GRAD_ACCUM_STEPS = 2  # Effective batch size = 32
    # ========================================================


    def train_one_model(model_name, train_paths, val_paths, train_labels, val_labels,
                        class_weights, device):
        """Train a single model and save the best checkpoint."""
        
        print(f"\n{'='*60}")
        print(f"  Training: {model_name}")
        print(f"{'='*60}")
        
        # Create datasets
        train_dataset = BrainTumorDataset(train_paths, train_labels, get_train_transforms())
        val_dataset = BrainTumorDataset(val_paths, val_labels, get_val_transforms())
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2, pin_memory=True)
        
        # Create model
        model = create_model(model_name, num_classes=NUM_CLASSES, pretrained=True)
        model = model.to(device)
        
        # Loss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Scheduler: cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Mixed precision
        scaler = GradScaler()
        
        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_best.pth')
        
        for epoch in range(NUM_EPOCHS):
            start = time.time()
            
            # --- Train ---
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            optimizer.zero_grad()
            
            for step, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS
                
                scaler.scale(loss).backward()
                
                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                running_loss += loss.item() * GRAD_ACCUM_STEPS
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
            
            # Handle remaining gradients
            if (step + 1) % GRAD_ACCUM_STEPS != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            scheduler.step()
            
            train_loss = running_loss / len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds, average='weighted')
            
            # --- Validate ---
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels_all = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with autocast(device_type='cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels_all.extend(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_labels_all, val_preds)
            val_f1 = f1_score(val_labels_all, val_preds, average='weighted')
            
            elapsed = time.time() - start
            lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1:>2}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                f"LR: {lr:.2e} | {elapsed:.1f}s")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_name': model_name,
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'epoch': epoch + 1,
                }, checkpoint_path)
                print(f"  ✓ New best model saved (F1={val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Load best and evaluate
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        final_preds = []
        final_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                final_preds.extend(preds)
                final_labels.extend(labels.cpu().numpy())
        
        print(f"\n--- {model_name} Best Model Report ---")
        print(f"Best Epoch: {checkpoint['epoch']}")
        print(f"Val Accuracy: {accuracy_score(final_labels, final_preds):.4f}")
        print(f"Val F1 (weighted): {f1_score(final_labels, final_preds, average='weighted'):.4f}")
        print(classification_report(final_labels, final_preds, target_names=CLASS_NAMES))
        
        return best_val_f1


    def main():
        set_seed(SEED)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load data
        train_paths, val_paths, train_labels, val_labels = load_train_data(
            TRAIN_DIR, val_split=0.15, seed=SEED
        )
        
        # Compute class weights
        class_weights = compute_class_weights(train_labels)
        print(f"Class weights: {class_weights.tolist()}")
        
        # Train each model
        model_names = get_model_names()
        results = {}
        
        for name in model_names:
            f1 = train_one_model(name, train_paths, val_paths, train_labels, val_labels,
                                class_weights, device)
            results[name] = f1
            # Free GPU memory
            torch.cuda.empty_cache()
        
        # Summary
        print(f"\n{'='*60}")
        print("  Training Complete — Summary")
        print(f"{'='*60}")
        for name, f1 in results.items():
            print(f"  {name:<20} Best Val F1: {f1:.4f}")
        print(f"\nCheckpoints saved in: {os.path.abspath(CHECKPOINT_DIR)}")
        print("Run `python predict.py` to generate submission.csv")


    if __name__ == '__main__':
        main()
