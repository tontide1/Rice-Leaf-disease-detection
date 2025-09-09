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
import datetime
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from PIL import Image
import copy
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Enable mixed precision if available
    try:
        scaler = torch.amp.GradScaler('cuda')
    except:
        # Fallback for older PyTorch versions
        scaler = torch.cuda.amp.GradScaler()
    print("Mixed precision enabled")
else:
    scaler = None

# Đường dẫn
DATA_DIR = Path("/home/tontide1/coding/deep_learning/Rice-Leaf-disease-detection/data/processed")
MODELS_DIR = Path("models")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"results/resnet_pytorch_{timestamp}")
LOGS_DIR = Path(f"logs/resnet_pytorch_{timestamp}")

# Tạo thư mục
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Các tham số
IMG_SIZE = 224  # ResNet standard input size
BATCH_SIZE = 32
EPOCHS_TRANSFER = 15  # Transfer learning epochs
EPOCHS_FINETUNE = 25  # Fine-tuning epochs
LEARNING_RATE_TRANSFER = 0.001
LEARNING_RATE_FINETUNE = 0.00005

def get_data_transforms():
    """Tạo data transforms cho training và validation"""
    # Training transforms với augmentation mạnh
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(25),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Validation/Test transforms - chỉ resize và normalize
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def create_data_loaders():
    """Tạo data loaders cho training, validation và test"""
    train_transforms, val_transforms = get_data_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=DATA_DIR / 'train',
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        root=DATA_DIR / 'val',
        transform=val_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        root=DATA_DIR / 'test',
        transform=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Lưu class mapping
    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    
    with open(RESULTS_DIR / 'class_indices.json', 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    print("Class names:", class_names)
    print("Class to index:", class_to_idx)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_names

class ResNetModel(nn.Module):
    """Custom ResNet50 model với transfer learning"""
    
    def __init__(self, num_classes, pretrained=True, feature_extract=True):
        super(ResNetModel, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone cho transfer learning
        if feature_extract:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier
        num_features = self.backbone.fc.in_features
        
        # Custom classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize new layers
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights của custom layers"""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_layers(self, num_layers=25):
        """Unfreeze các layers cuối cho fine-tuning"""
        # Get all layers
        all_layers = list(self.backbone.children())[:-1]  # Exclude FC layer
        total_layers = len(all_layers)
        
        # Freeze first layers, unfreeze last num_layers
        freeze_until = max(0, total_layers - num_layers)
        
        for i, layer in enumerate(all_layers):
            for param in layer.parameters():
                param.requires_grad = i >= freeze_until
        
        # FC layer luôn trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfroze last {num_layers} layers")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return trainable_params

def create_model(num_classes, feature_extract=True):
    """Tạo ResNet50 model"""
    model = ResNetModel(num_classes, pretrained=True, feature_extract=feature_extract)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, epoch, writer=None, use_amp=False):
    """Train một epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        # Update progress bar
        current_acc = running_corrects.double() / total_samples
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.4f}'
        })
        
        # Log to tensorboard
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
            writer.add_scalar('Train/Batch_Accuracy', current_acc, global_step)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, val_loader, criterion):
    """Validate một epoch"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, stage="transfer"):
    """Training loop cho một giai đoạn"""
    print(f"\n===== GIAI ĐOẠN: {stage.upper()} =====")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Loss function với label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5, min_lr=1e-7
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(LOGS_DIR / stage))
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': []
    }
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    early_stopping_patience = 10
    
    # Initialize current_lr
    current_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, writer, 
            use_amp=torch.cuda.is_available()
        )
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        
        # Update learning rate
        old_lr = current_lr
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Manual verbose logging for LR changes
        if current_lr != old_lr:
            print(f'ReduceLROnPlateau reducing learning rate to {current_lr:.2e}')
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f'Learning Rate: {current_lr:.2e}')
        
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
            }, MODELS_DIR / f'resnet50_pytorch_best_{stage}_{timestamp}.pth')
            
            print(f'New best validation accuracy: {best_acc:.4f}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    model.load_state_dict(best_model_wts)
    writer.close()
    
    print(f'\n{stage.capitalize()} completed. Best Val Acc: {best_acc:.4f}')
    
    return history, best_acc

def evaluate_model(model, data_loader, class_names, set_name):
    """Đánh giá mô hình trên một dataset"""
    print(f"\n===== ĐÁNH GIÁ TRÊN TẬP {set_name.upper()} =====")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f'Evaluating {set_name}'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save predictions
    results_df = pd.DataFrame({
        'y_true': all_labels,
        'y_pred': all_preds,
        'confidence': [max(prob) for prob in all_probs]
    })
    results_df.to_csv(RESULTS_DIR / f'predictions_{set_name}.csv', index=False)
    
    # Confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names, set_name)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(f"\nClassification Report ({set_name.capitalize()} Set):")
    print(report)
    
    with open(RESULTS_DIR / f'classification_report_{set_name}.txt', 'w') as f:
        f.write(report)
    
    # Save metrics
    with open(RESULTS_DIR / f'results_{set_name}.txt', 'w') as f:
        f.write(f"ResNet50 PyTorch Transfer Learning + Fine-tuning\n")
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Results for: {set_name} set\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
    
    return accuracy

def plot_confusion_matrix(y_true, y_pred, class_names, set_name):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {set_name.capitalize()} Set\nResNet50 PyTorch')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'confusion_matrix_{set_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_history(history1, history2=None):
    """Vẽ training history"""
    plt.figure(figsize=(20, 5))
    
    # Combine histories nếu có fine-tuning
    if history2 is not None:
        combined_train_loss = history1['train_loss'] + history2['train_loss']
        combined_val_loss = history1['val_loss'] + history2['val_loss']
        combined_train_acc = history1['train_acc'] + history2['train_acc']
        combined_val_acc = history1['val_acc'] + history2['val_acc']
        combined_lr = history1['lr'] + history2['lr']
        
        transfer_epochs = len(history1['train_loss'])
        epochs = range(1, len(combined_train_loss) + 1)
    else:
        combined_train_loss = history1['train_loss']
        combined_val_loss = history1['val_loss']
        combined_train_acc = history1['train_acc']
        combined_val_acc = history1['val_acc']
        combined_lr = history1['lr']
        transfer_epochs = 0
        epochs = range(1, len(combined_train_loss) + 1)
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, combined_train_loss, label='Train', linewidth=2)
    plt.plot(epochs, combined_val_loss, label='Validation', linewidth=2)
    if transfer_epochs > 0:
        plt.axvline(x=transfer_epochs, color='red', linestyle='--', alpha=0.7,
                   label='Fine-tuning starts')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, combined_train_acc, label='Train', linewidth=2)
    plt.plot(epochs, combined_val_acc, label='Validation', linewidth=2)
    if transfer_epochs > 0:
        plt.axvline(x=transfer_epochs, color='red', linestyle='--', alpha=0.7,
                   label='Fine-tuning starts')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, combined_lr, linewidth=2)
    if transfer_epochs > 0:
        plt.axvline(x=transfer_epochs, color='red', linestyle='--', alpha=0.7,
                   label='Fine-tuning starts')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'resnet50_pytorch_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("===== RESNET50 PYTORCH TRANSFER LEARNING + FINE-TUNING =====")
    print("Giai đoạn 1: Transfer Learning (freeze ResNet50)")
    print("Giai đoạn 2: Fine-tuning (unfreeze last layers)")
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders()
    num_classes = len(class_names)
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Create model
    model = create_model(num_classes, feature_extract=True)
    
    # Giai đoạn 1: Transfer Learning
    history1, best_acc1 = train_model(
        model, train_loader, val_loader,
        num_epochs=EPOCHS_TRANSFER,
        learning_rate=LEARNING_RATE_TRANSFER,
        stage="transfer"
    )
    
    # Giai đoạn 2: Fine-tuning
    print("\n" + "="*60)
    print("Bắt đầu Fine-tuning...")
    
    # Unfreeze layers cho fine-tuning
    trainable_params_ft = model.unfreeze_layers(num_layers=25)
    
    history2, best_acc2 = train_model(
        model, train_loader, val_loader,
        num_epochs=EPOCHS_FINETUNE,
        learning_rate=LEARNING_RATE_FINETUNE,
        stage="finetune"
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'history1': history1,
        'history2': history2,
        'best_acc_transfer': best_acc1,
        'best_acc_finetune': best_acc2
    }, MODELS_DIR / f'resnet50_pytorch_final_{timestamp}.pth')
    
    # Plot training history
    plot_training_history(history1, history2)
    
    # Evaluation trên tất cả các tập
    print("\n" + "="*60)
    print("ĐÁNH GIÁ CUỐI CÙNG")
    print("="*60)
    
    train_acc = evaluate_model(model, train_loader, class_names, "train")
    val_acc = evaluate_model(model, val_loader, class_names, "validation")
    test_acc = evaluate_model(model, test_loader, class_names, "test")
    
    # Tóm tắt kết quả
    print(f"\n{'='*60}")
    print("TÓM TẮT KẾT QUẢ RESNET50 PYTORCH")
    print(f"{'='*60}")
    print(f"Transfer Learning Best Val Acc: {best_acc1:.4f}")
    print(f"Fine-tuning Best Val Acc: {best_acc2:.4f}")
    print(f"Final Train Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Run timestamp: {timestamp}")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Logs saved in: {LOGS_DIR}")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'model': 'ResNet50 PyTorch Transfer Learning + Fine-tuning',
        'device': str(device),
        'epochs_transfer': EPOCHS_TRANSFER,
        'epochs_finetune': EPOCHS_FINETUNE,
        'best_acc_transfer': float(best_acc1),
        'best_acc_finetune': float(best_acc2),
        'final_train_accuracy': float(train_acc),
        'final_val_accuracy': float(val_acc),
        'final_test_accuracy': float(test_acc),
        'classes': class_names
    }
    
    with open(RESULTS_DIR / 'run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
