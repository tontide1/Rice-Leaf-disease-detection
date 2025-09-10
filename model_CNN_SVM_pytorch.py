import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import datetime
import json
import pandas as pd
import pickle
import seaborn as sns
from tqdm import tqdm
import copy
import time
from torch.cuda.amp import autocast, GradScaler
import psutil
import gc

# Configuration class for better parameter management
class Config:
    # Basic settings
    SEED = 42
    IMG_SIZE = 224
    BATCH_SIZE = 64  # Increased from 32
    EPOCHS_CNN = 30
    
    # Paths
    DATA_DIR = Path("/home/tontide1/coding/deep_learning/Rice-Leaf-disease-detection/data/processed")
    MODELS_DIR = Path("models")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = Path(f"results/hybrid_cnn_svm_pytorch_optimized_{timestamp}")
    LOGS_DIR = Path(f"logs/hybrid_cnn_svm_pytorch_optimized_{timestamp}")
    
    # Training parameters
    LEARNING_RATE = 2e-3  # Slightly increased
    WEIGHT_DECAY = 2e-5  # Added weight decay
    FEATURE_DIM = 512
    LABEL_SMOOTHING = 0.1
    
    # Optimization
    MIXED_PRECISION = True
    NUM_WORKERS = 4  # Reduced from 6 to system-recommended value
    GRAD_ACCUMULATION_STEPS = 2  # Added for larger effective batch size
    
    # Early stopping
    PATIENCE = 10
    LR_PATIENCE = 5
    
    # SVM parameters
    SVM_CV_FOLDS = 3
    
    def __init__(self):
        # Create directories
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Create config
cfg = Config()

# Set reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(cfg.SEED)

# Device setup with memory management
def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Print GPU info
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Reserve GPU memory
        torch.cuda.empty_cache()
        
        # Enable mixed precision if available - FIX FOR DEPRECATION WARNING
        if cfg.MIXED_PRECISION:
            try:
                # New recommended way
                scaler = torch.amp.GradScaler('cuda')
                print("Using torch.amp.GradScaler('cuda')")
            except Exception as e:
                # Fallback for older PyTorch versions
                scaler = torch.cuda.amp.GradScaler()
                print("Fallback to torch.cuda.amp.GradScaler()")
        else:
            scaler = None
        
        # Set optimal CUDA settings
        torch.backends.cudnn.benchmark = True  # Can improve speed for fixed input sizes
    else:
        device = torch.device('cpu')
        print("Using CPU")
        scaler = None
    
    return device, scaler

device, scaler = setup_device()

# Optimized data transforms with improved augmentation
def get_optimized_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.85, 1.0)),  # Improved scale
        transforms.ColorJitter(brightness=[0.8, 1.2], contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Added perspective
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Simple but effective validation transforms
    val_transforms = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.CenterCrop(cfg.IMG_SIZE),  # Added center crop
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

# Optimized data loaders with better memory management
def create_optimized_data_loaders():
    train_transforms, val_transforms = get_optimized_transforms()
    
    # Memory-efficient dataset loading
    train_dataset = datasets.ImageFolder(
        root=cfg.DATA_DIR / 'train',
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        root=cfg.DATA_DIR / 'val',
        transform=val_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        root=cfg.DATA_DIR / 'test',
        transform=val_transforms
    )
    
    # Create data loaders with optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True  # Avoid small last batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Save class mapping
    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    
    with open(cfg.RESULTS_DIR / 'class_indices.json', 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    print("Class names:", class_names)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_names

# Optimized CNN Feature Extractor
class OptimizedCNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super(OptimizedCNNFeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Block 1 - Using SiLU/Swish activation (more efficient)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),  # Removed bias
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),  # Better than ReLU
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature layer with improved regularization
        self.feature_layer = nn.Sequential(
            nn.Linear(256, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Initialize weights with efficient method
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        features = self.feature_layer(x)
        return features

# Complete CNN Classifier with feature extraction
class OptimizedCNNClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super(OptimizedCNNClassifier, self).__init__()
        
        self.feature_extractor = OptimizedCNNFeatureExtractor(feature_dim)
        
        # Improved classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features

# LR warmup and cosine decay scheduler
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Optimized training function with gradient accumulation
def train_optimized_cnn(model, train_loader, val_loader, num_epochs):
    print("\n===== TRAINING OPTIMIZED CNN FEATURE EXTRACTOR =====")
    
    # Training settings
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    
    # AdamW optimizer (better than Adam)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Calculate total steps for cosine scheduler
    total_steps = len(train_loader) // cfg.GRAD_ACCUMULATION_STEPS * num_epochs
    warmup_steps = total_steps // 10  # 10% warmup
    
    # Learning rate scheduler with warmup and cosine decay
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr=1e-6
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(cfg.LOGS_DIR / "cnn_training"))
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': []
    }
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    # Track training time
    start_time = time.time()
    
    # Initialize step for scheduler
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Reset gradients for first mini-batch
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        
        # Batch accumulation counter
        batch_acc_counter = 0
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            # Move to device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=cfg.MIXED_PRECISION):
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels) / cfg.GRAD_ACCUMULATION_STEPS
            
            # Backward pass with scaling
            if cfg.MIXED_PRECISION:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Statistics - scale back the loss for reporting
            loss_item = loss.item() * cfg.GRAD_ACCUMULATION_STEPS
            _, preds = torch.max(outputs, 1)
            running_loss += loss_item * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Update progress bar
            current_acc = running_corrects.double() / total_samples
            progress_bar.set_postfix({
                'Loss': f'{loss_item:.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Gradient accumulation step counter
            batch_acc_counter += 1
            
            # Update weights after accumulating gradients
            if batch_acc_counter == cfg.GRAD_ACCUMULATION_STEPS or batch_idx == len(train_loader) - 1:
                if cfg.MIXED_PRECISION:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Step the scheduler
                scheduler.step()
                global_step += 1
                
                # Reset gradients and counter
                optimizer.zero_grad()
                batch_acc_counter = 0
        
        # Calculate epoch statistics
        train_loss = running_loss / total_samples
        train_acc = running_corrects.double() / total_samples
        
        # Validation phase
        val_loss, val_acc = validate_model(model, val_loader, criterion)
        
        # Learning rate for current epoch
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        history['lr'].append(current_lr)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Print epoch results
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f'Learning Rate: {current_lr:.2e}')
        
        # Memory management
        torch.cuda.empty_cache()
        
        # Track system resource usage
        print(f"RAM Usage: {psutil.virtual_memory().percent}%")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, cfg.MODELS_DIR / f'optimized_cnn_feature_extractor_{cfg.timestamp}.pth')
            
            print(f'New best validation accuracy: {best_acc:.4f}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= cfg.PATIENCE:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Training time
    time_elapsed = time.time() - start_time
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # Load best model
    model.load_state_dict(best_model_wts)
    writer.close()
    
    print(f'\nCNN training completed. Best Val Acc: {best_acc:.4f}')
    return history, best_acc

# Validation function
def validate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Validation'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(enabled=cfg.MIXED_PRECISION):
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    val_loss = running_loss / total_samples
    val_acc = running_corrects.double() / total_samples
    
    return val_loss, val_acc

# Feature extraction with batching
def extract_features_optimized(model, data_loader, set_name):
    print(f"\n===== EXTRACTING FEATURES: {set_name.upper()} SET =====")
    
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f'Extracting {set_name} features'):
            inputs = inputs.to(device, non_blocking=True)
            
            # Extract features with mixed precision
            with autocast(enabled=cfg.MIXED_PRECISION):
                _, features = model(inputs)
            
            # Move to CPU and convert to numpy
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
            # Free memory
            del inputs, features
            
    # Combine all features
    all_features = np.vstack(features_list)
    all_labels = np.hstack(labels_list)
    
    # Free memory
    del features_list, labels_list
    gc.collect()
    
    print(f"{set_name.capitalize()} features shape: {all_features.shape}")
    print(f"{set_name.capitalize()} labels shape: {all_labels.shape}")
    
    return all_features, all_labels

# Optimized SVM training with parallel processing
def train_optimized_svm(train_features, train_labels, val_features, val_labels, class_names):
    print("\n===== TRAINING OPTIMIZED SVM CLASSIFIER =====")
    
    # Standardize features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    
    # Optimized GridSearch parameters
    print("Performing GridSearch for SVM hyperparameters...")
    
    # Focus on most promising parameters based on domain knowledge
    param_grid = {
        'C': [1, 10, 100],  # Focused range
        'kernel': ['rbf'],  # Best kernel for image features
        'gamma': ['scale', 'auto', 0.01, 0.1]  # Reduced options
    }
    
    svm = SVC(random_state=cfg.SEED, probability=True, cache_size=1000)  # Increased cache
    
    # Use more workers and verbosity
    grid_search = GridSearchCV(
        svm, param_grid, cv=cfg.SVM_CV_FOLDS, 
        scoring='accuracy', n_jobs=-1, verbose=2
    )
    
    # Time SVM training
    start_time = time.time()
    grid_search.fit(train_features_scaled, train_labels)
    svm_time = time.time() - start_time
    
    # Best SVM model
    best_svm = grid_search.best_estimator_
    
    print(f"SVM training completed in {svm_time:.2f} seconds")
    print(f"Best SVM parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    val_accuracy = best_svm.score(val_features_scaled, val_labels)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Predictions for detailed analysis
    val_pred = best_svm.predict(val_features_scaled)
    
    # Classification report
    print("\nValidation Classification Report:")
    report = classification_report(val_labels, val_pred, target_names=class_names)
    print(report)
    
    # Save SVM model and scaler
    with open(cfg.MODELS_DIR / f'optimized_svm_classifier_{cfg.timestamp}.pkl', 'wb') as f:
        pickle.dump(best_svm, f)
    
    with open(cfg.MODELS_DIR / f'optimized_feature_scaler_{cfg.timestamp}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"SVM model and scaler saved!")
    
    return best_svm, scaler, grid_search.best_params_

# Evaluation with detailed metrics
def evaluate_hybrid_model_optimized(cnn_model, svm_model, scaler, data_loader, class_names, set_name):
    print(f"\n===== EVALUATING OPTIMIZED HYBRID MODEL: {set_name.upper()} SET =====")
    
    # Extract features
    features, true_labels = extract_features_optimized(cnn_model, data_loader, set_name)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # SVM predictions with timing
    start_time = time.time()
    svm_predictions = svm_model.predict(features_scaled)
    svm_probabilities = svm_model.predict_proba(features_scaled)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, svm_predictions)
    
    print(f"{set_name.capitalize()} Accuracy: {accuracy:.4f}")
    print(f"Inference time: {inference_time:.4f}s for {len(true_labels)} samples")
    print(f"Average inference time: {inference_time*1000/len(true_labels):.2f}ms per sample")
    
    # Detailed classification report
    report = classification_report(true_labels, svm_predictions, target_names=class_names)
    print(f"\nClassification Report ({set_name.capitalize()} Set):")
    print(report)
    
    # Save results with more details
    results_df = pd.DataFrame({
        'y_true': true_labels,
        'y_pred': svm_predictions,
        'confidence': np.max(svm_probabilities, axis=1)
    })
    
    # Add class names
    results_df['true_class'] = results_df['y_true'].apply(lambda x: class_names[x])
    results_df['pred_class'] = results_df['y_pred'].apply(lambda x: class_names[x])
    
    # Add correct/incorrect flag
    results_df['correct'] = results_df['y_true'] == results_df['y_pred']
    
    # Save to CSV
    results_df.to_csv(cfg.RESULTS_DIR / f'hybrid_predictions_{set_name}.csv', index=False)
    
    # Save classification report
    with open(cfg.RESULTS_DIR / f'hybrid_classification_report_{set_name}.txt', 'w') as f:
        f.write(f"Optimized Hybrid CNN+SVM PyTorch Model\n")
        f.write(f"Set: {set_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Inference time: {inference_time:.4f}s for {len(true_labels)} samples\n")
        f.write(f"Average inference time: {inference_time*1000/len(true_labels):.2f}ms per sample\n\n")
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix_optimized(true_labels, svm_predictions, class_names, f"hybrid_{set_name}")
    
    return accuracy, inference_time

# Improved confusion matrix visualization
def plot_confusion_matrix_optimized(y_true, y_pred, class_names, set_name):
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f'Confusion Matrix (Counts) - {set_name.replace("_", " ").title()}')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Plot normalized percentages
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title(f'Confusion Matrix (Normalized) - {set_name.replace("_", " ").title()}')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(cfg.RESULTS_DIR / f'confusion_matrix_{set_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Enhanced training history visualization
def plot_training_history_optimized(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create a more detailed visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_title('CNN Training Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], label='Validation', linewidth=2)
    axes[0, 1].set_title('CNN Training Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(epochs, history['lr'], linewidth=2, color='green')
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training-validation gap plot (to visualize overfitting)
    train_val_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(epochs, train_val_gap, linewidth=2, color='red')
    axes[1, 1].set_title('Train-Validation Accuracy Gap', fontsize=14)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Gap (Train-Val)', fontsize=12)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(cfg.RESULTS_DIR / 'optimized_cnn_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("===== OPTIMIZED HYBRID CNN+SVM PYTORCH MODEL =====")
    start_time = time.time()
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_optimized_data_loaders()
    num_classes = len(class_names)
    
    print(f"\nNumber of classes: {num_classes}")
    
    try:
        # Step 1: Create and train CNN feature extractor
        cnn_model = OptimizedCNNClassifier(num_classes, feature_dim=cfg.FEATURE_DIM).to(device)
        
        # Print model parameters
        total_params = sum(p.numel() for p in cnn_model.parameters())
        trainable_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
        print(f"\nOptimized CNN Model - Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Train CNN
        cnn_history, best_cnn_acc = train_optimized_cnn(
            cnn_model, train_loader, val_loader, cfg.EPOCHS_CNN
        )
        
        # Plot training history
        plot_training_history_optimized(cnn_history)
        
        # Step 2: Extract features
        print("\n" + "="*60)
        print("EXTRACTING FEATURES FOR SVM")
        print("="*60)
        
        train_features, train_labels = extract_features_optimized(cnn_model, train_loader, "train")
        val_features, val_labels = extract_features_optimized(cnn_model, val_loader, "validation")
        test_features, test_labels = extract_features_optimized(cnn_model, test_loader, "test")
        
        # Step 3: Train SVM classifier
        print("\n" + "="*60)
        print("TRAINING OPTIMIZED SVM CLASSIFIER")
        print("="*60)
        
        svm_model, scaler, best_svm_params = train_optimized_svm(
            train_features, train_labels, val_features, val_labels, class_names
        )
        
        # Step 4: Evaluate hybrid model
        print("\n" + "="*60)
        print("EVALUATING OPTIMIZED HYBRID CNN+SVM MODEL")
        print("="*60)
        
        train_acc, train_time = evaluate_hybrid_model_optimized(
            cnn_model, svm_model, scaler, train_loader, class_names, "train"
        )
        
        val_acc, val_time = evaluate_hybrid_model_optimized(
            cnn_model, svm_model, scaler, val_loader, class_names, "validation"
        )
        
        test_acc, test_time = evaluate_hybrid_model_optimized(
            cnn_model, svm_model, scaler, test_loader, class_names, "test"
        )
        
        # Total execution time
        total_time = time.time() - start_time
        
        # Summary
        print(f"\n{'='*60}")
        print("TÓM TẮT KẾT QUẢ OPTIMIZED HYBRID CNN+SVM")
        print(f"{'='*60}")
        print(f"CNN Best Validation Accuracy: {best_cnn_acc:.4f}")
        print(f"Hybrid Train Accuracy: {train_acc:.4f}")
        print(f"Hybrid Validation Accuracy: {val_acc:.4f}")
        print(f"Hybrid Test Accuracy: {test_acc:.4f}")
        print(f"Test Inference Time: {test_time:.2f}s ({test_time*1000/len(test_labels):.2f}ms/sample)")
        print(f"Total Execution Time: {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"Results saved in: {cfg.RESULTS_DIR}")
        
        # Save detailed summary
        summary = {
            'timestamp': cfg.timestamp,
            'model': 'Optimized Hybrid CNN+SVM PyTorch',
            'device': str(device),
            'parameters': {
                'batch_size': cfg.BATCH_SIZE,
                'learning_rate': cfg.LEARNING_RATE,
                'weight_decay': cfg.WEIGHT_DECAY,
                'feature_dim': cfg.FEATURE_DIM,
                'grad_accumulation_steps': cfg.GRAD_ACCUMULATION_STEPS,
                'cnn_epochs': cfg.EPOCHS_CNN,
                'mixed_precision': cfg.MIXED_PRECISION
            },
            'cnn_model': {
                'total_parameters': int(total_params),
                'trainable_parameters': int(trainable_params),
                'best_val_accuracy': float(best_cnn_acc)
            },
            'svm_model': {
                'best_parameters': best_svm_params
            },
            'performance': {
                'train_accuracy': float(train_acc),
                'val_accuracy': float(val_acc),
                'test_accuracy': float(test_acc),
                'test_inference_time': float(test_time),
                'test_inference_time_per_sample': float(test_time*1000/len(test_labels)),
                'total_execution_time': float(total_time)
            },
            'classes': class_names
        }
        
        with open(cfg.RESULTS_DIR / 'optimized_hybrid_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nAll results saved successfully!")
        
    except Exception as e:
        print(f"Error in execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()