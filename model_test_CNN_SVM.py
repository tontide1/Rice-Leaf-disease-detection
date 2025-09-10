import os
import numpy as np
import pandas as pd
import time
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.cuda.amp import autocast
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class Config:
    # Model settings
    FEATURE_SCALER_PATH = "models/optimized_feature_scaler_20250908_220820_FINAL.pkl"
    SVM_MODEL_PATH = "models/optimized_svm_classifier_20250908_220820_FINAL.pkl"
    CNN_MODEL_PATH = "models/optimized_cnn_feature_extractor_20250908_220820.pth"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 4  # Số lớp cố định: 3 lớp bệnh + 1 lớp khỏe mạnh
    FEATURE_DIM = 512
    MIXED_PRECISION = True
    
    # Đường dẫn cơ sở đến bộ dữ liệu
    BASE_DATA_PATH = "/home/tontide1/.cache/kagglehub/datasets/tcdatt/data-test-deeplearning/versions/1/test_data_deeplearning"
    
    # Ánh xạ tên lớp để xử lý sự không nhất quán
    CLASS_MAPPING = {
        # Tên lớp tiêu chuẩn -> Tên hiển thị
        "bacterial_leaf_blight": "Bacterial Leaf Blight",
        "blast": "Blast",
        "brown_spot": "Brown Spot",
        "normal": "Healthy",
        "healthy": "Healthy",
        "Brown_Spot": "Brown Spot",
        "Healthy": "Healthy",
        "Leaf_Blast": "Blast",
        "Leaf_Blight": "Bacterial Leaf Blight"
    }
    
    # Datasets to test
    EXTERNAL_DATASETS = [
        {
            "name": "AUG Dataset",
            "path": f"{BASE_DATA_PATH}/AUG", 
            "reference": "AUG",
            "classes": ["bacterial_leaf_blight", "blast", "brown_spot", "normal"]
        },
        {
            "name": "DoMinhHuy Dataset",
            "path": f"{BASE_DATA_PATH}/DoMinhHuy", 
            "reference": "DoMinhHuy",
            "classes": ["bacterial_leaf_blight", "blast", "brown_spot", "normal"]
        },
        {
            "name": "Kaggle Dataset",
            "path": f"{BASE_DATA_PATH}/kaggle", 
            "reference": "Kaggle",
            "classes": ["blast", "brown_spot", "normal"]
        },
        {
            "name": "Melody Dataset",
            "path": f"{BASE_DATA_PATH}/meledy", 
            "reference": "Melody",
            "classes": ["bacterial_leaf_blight", "blast", "brown_spot"]
        },
        {
            "name": "Realistic Dataset",
            "path": f"{BASE_DATA_PATH}/realistic", 
            "reference": "Realistic",
            "classes": ["Brown_Spot", "Healthy", "Leaf_Blast", "Leaf_Blight"]
        },
        {
            "name": "Roboflow Blast Dataset",
            "path": f"{BASE_DATA_PATH}/roboflow/Blast", 
            "reference": "Roboflow",
            "classes": ["blast"],
            "is_flat_structure": True
        }
    ]
    
    # Output settings
    RESULTS_DIR = Path("results/external_evaluation_CNN_SVM")
    
    def __init__(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

cfg = Config()

# Định nghĩa kiến trúc mô hình CNN từ file model_hybrid_custom_pytorch_optimized.py
class OptimizedCNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super(OptimizedCNNFeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Block 1 - Using SiLU/Swish activation (more efficient)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
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
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        features = self.feature_layer(x)
        return features

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

# Custom Dataset class for external datasets
class ExternalDataset(Dataset):
    def __init__(self, data_dir, transform=None, expected_classes=None, is_flat_structure=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_flat_structure = is_flat_structure
        
        if is_flat_structure:
            # Xử lý cấu trúc phẳng - tất cả ảnh trong thư mục gốc
            self.classes = expected_classes if expected_classes else ["default_class"]
            self.class_to_idx = {self.classes[0]: 0}
            
            # Tìm tất cả ảnh trong thư mục gốc và các thư mục con
            self.image_paths = []
            self.labels = []
            
            # Tìm trong thư mục gốc
            img_paths = list(self.data_dir.glob('*.[jp][pn]g'))
            for img_path in img_paths:
                self.image_paths.append(img_path)
                self.labels.append(0)  # Lớp đầu tiên
            
            # Nếu không tìm thấy ảnh trong thư mục gốc, tìm trong thư mục con
            if len(self.image_paths) == 0:
                for subdir in self.data_dir.iterdir():
                    if subdir.is_dir():
                        for img_path in subdir.glob('*.[jp][pn]g'):
                            self.image_paths.append(img_path)
                            self.labels.append(0)  # Lớp đầu tiên
            
            self.original_class_names = self.classes
        else:
            # Cấu trúc thông thường - các thư mục con là các lớp
            available_classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
            
            # Sử dụng danh sách lớp được cung cấp nếu có
            self.classes = expected_classes if expected_classes else available_classes
            
            # Kiểm tra các lớp đã được tìm thấy
            for cls in self.classes:
                if cls not in available_classes:
                    print(f"Warning: Expected class '{cls}' not found in {data_dir}")
            
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes) if cls_name in available_classes}
            
            # Tìm tất cả ảnh và nhãn
            self.image_paths = []
            self.labels = []
            
            for cls_name, idx in self.class_to_idx.items():
                cls_dir = self.data_dir / cls_name
                if not cls_dir.exists():
                    continue
                    
                for img_path in cls_dir.glob('*.[jp][pn]g'):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)
            
            self.original_class_names = list(self.class_to_idx.keys())
        
        # Ánh xạ tên lớp gốc sang tên hiển thị
        self.display_class_names = [cfg.CLASS_MAPPING.get(cls, cls) for cls in self.classes]
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in dataset {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    """Get transforms for evaluation"""
    # Sử dụng transforms tương tự như trong val_transforms của mô hình gốc
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.CenterCrop(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_models():
    """Load CNN Feature Extractor, SVM Classifier, and Feature Scaler"""
    print("Loading models:")
    
    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load CNN Feature Extractor
        print(f"  Loading CNN Feature Extractor: {cfg.CNN_MODEL_PATH}")
        cnn_model = OptimizedCNNClassifier(cfg.NUM_CLASSES, feature_dim=cfg.FEATURE_DIM)
        
        # Xử lý cho PyTorch 2.6+ - tải với weights_only=False
        try:
            # Phương pháp 1: Sử dụng add_safe_globals
            import numpy as np
            from torch.serialization import add_safe_globals
            add_safe_globals([np.core.multiarray.scalar])
            checkpoint = torch.load(cfg.CNN_MODEL_PATH, map_location=device)
        except:
            # Phương pháp 2: weights_only=False (chỉ nên dùng nếu tin tưởng nguồn checkpoint)
            print("  Falling back to loading with weights_only=False")
            checkpoint = torch.load(cfg.CNN_MODEL_PATH, map_location=device, weights_only=False)
        
        # In thông tin checkpoint để debug
        if isinstance(checkpoint, dict):
            print(f"  Checkpoint keys: {list(checkpoint.keys())}")
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            cnn_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            cnn_model.load_state_dict(checkpoint['state_dict'])
        else:
            cnn_model.load_state_dict(checkpoint)
        
        cnn_model = cnn_model.to(device)
        cnn_model.eval()
        print("  CNN Feature Extractor loaded successfully")
        
        # Load Feature Scaler
        print(f"  Loading Feature Scaler: {cfg.FEATURE_SCALER_PATH}")
        with open(cfg.FEATURE_SCALER_PATH, 'rb') as f:
            feature_scaler = pickle.load(f)
            
        # Kiểm tra thông tin scaler
        if hasattr(feature_scaler, 'n_features_in_'):
            print(f"  Feature Scaler expects {feature_scaler.n_features_in_} features")
            if feature_scaler.n_features_in_ != cfg.FEATURE_DIM:
                print(f"  Warning: Feature dimension mismatch! CNN outputs {cfg.FEATURE_DIM}, "
                      f"but scaler expects {feature_scaler.n_features_in_}")
        
        print("  Feature Scaler loaded successfully")
        
        # Load SVM Classifier
        print(f"  Loading SVM Classifier: {cfg.SVM_MODEL_PATH}")
        with open(cfg.SVM_MODEL_PATH, 'rb') as f:
            svm_model = pickle.load(f)
        
        # Kiểm tra thông tin SVM
        if hasattr(svm_model, 'n_features_in_'):
            print(f"  SVM expects {svm_model.n_features_in_} features")
        if hasattr(svm_model, 'classes_'):
            print(f"  SVM trained with {len(svm_model.classes_)} classes: {svm_model.classes_}")
            
        print("  SVM Classifier loaded successfully")
        
        return cnn_model, feature_scaler, svm_model, device
        
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, device

def extract_features_with_cnn(model, data_loader, device):
    """Extract features using the CNN model"""
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f"Extracting features"):
            inputs = inputs.to(device, non_blocking=True)
            
            # Extract features with mixed precision
            with autocast(enabled=cfg.MIXED_PRECISION):
                _, features = model(inputs)
            
            # Move to CPU and convert to numpy
            features_np = features.cpu().numpy()
            
            # Kiểm tra NaN và thay thế bằng 0
            if np.isnan(features_np).any():
                nan_count = np.isnan(features_np).sum()
                print(f"  Warning: {nan_count} NaN values detected in batch. Replacing with 0.")
                features_np = np.nan_to_num(features_np, nan=0.0)
                
            features_list.append(features_np)
            labels_list.append(labels.numpy())
            
            # Free memory
            del inputs, features
            
    # Combine all features
    all_features = np.vstack(features_list)
    all_labels = np.hstack(labels_list)
    
    # Kiểm tra lại một lần nữa
    nan_count = np.isnan(all_features).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values still detected in combined features. Replacing with 0.")
        all_features = np.nan_to_num(all_features, nan=0.0)
    
    return all_features, all_labels

def evaluate_dataset(cnn_model, feature_scaler, svm_model, data_loader, device, dataset_name):
    """Evaluate hybrid model on a dataset"""
    print(f"Evaluating on {dataset_name}...")
    
    # Kiểm tra nếu không có dữ liệu
    if len(data_loader.dataset) == 0:
        print(f"Warning: No samples in {dataset_name}, skipping evaluation")
        return {
            "dataset_name": dataset_name,
            "accuracy": 0.0,
            "fps": 0.0,
            "report": {},
            "confusion_matrix": np.array([[]]),
            "classes": [],
            "original_classes": [],
            "num_classes": 0,
            "total_images": 0,
            "total_time": 0.0
        }
    
    # Đo thời gian tổng
    start_time = time.time()
    
    # Extract features using CNN
    features, labels = extract_features_with_cnn(cnn_model, data_loader, device)
    
    # Scale features và xử lý NaN nếu có
    try:
        features_scaled = feature_scaler.transform(features)
        
        # Kiểm tra NaN sau khi scale
        nan_count = np.isnan(features_scaled).sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values detected after scaling. Replacing with 0.")
            features_scaled = np.nan_to_num(features_scaled, nan=0.0)
            
        # Kiểm tra giá trị vô cùng (inf)
        inf_count = np.isinf(features_scaled).sum()
        if inf_count > 0:
            print(f"Warning: {inf_count} infinite values detected. Replacing with large values.")
            features_scaled = np.nan_to_num(features_scaled, posinf=1e10, neginf=-1e10)
    except Exception as e:
        print(f"Error during feature scaling: {e}")
        print("Using unscaled features and replacing NaN with 0")
        features_scaled = np.nan_to_num(features, nan=0.0)
    
    # Predict with SVM
    svm_predictions = svm_model.predict(features_scaled)
    
    # Calculate inference time
    total_time = time.time() - start_time
    
    # Calculate FPS
    total_images = len(data_loader.dataset)
    fps = total_images / total_time if total_time > 0 else 0
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, svm_predictions) * 100
    
    # Xác định các nhãn duy nhất
    unique_labels = sorted(np.unique(labels))
    
    # Lấy tên lớp hiển thị cho các nhãn duy nhất
    all_display_classes = data_loader.dataset.display_class_names
    display_classes = [all_display_classes[i] for i in unique_labels]
    original_classes = [data_loader.dataset.original_class_names[i] for i in unique_labels]
    
    # Generate classification report
    report = classification_report(
        labels, 
        svm_predictions,
        labels=unique_labels,
        target_names=display_classes,
        output_dict=True,
        zero_division=0
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(labels, svm_predictions)
    
    # Create results dict
    results = {
        "dataset_name": dataset_name,
        "accuracy": accuracy,
        "fps": fps,
        "report": report,
        "confusion_matrix": cm,
        "classes": display_classes,
        "original_classes": original_classes,
        "num_classes": len(display_classes),
        "total_images": total_images,
        "total_time": total_time
    }
    
    print(f"{dataset_name} - Accuracy: {accuracy:.2f}% - FPS: {fps:.2f}")
    return results

def create_comparison_table(all_results):
    """Create a comparison table without Total no of images and with correct No of diseases"""
    data = []
    
    for i, result in enumerate(all_results):
        dataset = cfg.EXTERNAL_DATASETS[i]
        
        # Đếm số lớp bệnh từ thông tin ban đầu của dataset
        expected_classes = dataset["classes"]
        disease_classes = [cls for cls in expected_classes if cls not in ["normal", "healthy", "Healthy"]]
        
        data.append({
            "Article reference": dataset["reference"],
            "Dataset reference": dataset["name"],
            "No of diseases": len(disease_classes),
            "Names of diseases": ", ".join([cfg.CLASS_MAPPING.get(cls, cls) for cls in expected_classes]),
            "Accuracy(%)": f"{result['accuracy']:.2f}",
            "FPS": f"{result['fps']:.2f}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(cfg.RESULTS_DIR / "comparison_results_CNN_SVM.csv", index=False)
    
    # Create HTML version for better visualization
    html = df.to_html(index=False)
    with open(cfg.RESULTS_DIR / "comparison_results_CNN_SVM.html", 'w') as f:
        f.write(html)
    
    return df

def main():
    # Load all models
    cnn_model, feature_scaler, svm_model, device = load_models()
    
    if cnn_model is None or feature_scaler is None or svm_model is None:
        print("Failed to load models. Exiting.")
        return
    
    # Get transforms for evaluation
    transform = get_transforms()
    
    all_results = []
    
    # Evaluate each dataset
    for dataset_info in cfg.EXTERNAL_DATASETS:
        # Create dataset
        dataset = ExternalDataset(
            dataset_info["path"], 
            transform=transform,
            expected_classes=dataset_info.get("classes"),
            is_flat_structure=dataset_info.get("is_flat_structure", False)
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Evaluate model on dataset
        results = evaluate_dataset(
            cnn_model, 
            feature_scaler, 
            svm_model, 
            dataloader, 
            device, 
            dataset_info["name"]
        )
        
        all_results.append(results)
        
        # Save detailed report
        with open(cfg.RESULTS_DIR / f"report_CNN_SVM_{dataset_info['name'].replace(' ', '_')}.txt", 'w') as f:
            f.write(f"Dataset: {dataset_info['name']}\n")
            f.write(f"Reference: {dataset_info['reference']}\n")
            f.write(f"Total images: {results['total_images']}\n")
            f.write(f"Original classes: {', '.join(results['original_classes'])}\n")
            f.write(f"Standardized classes: {', '.join(results['classes'])}\n")
            f.write(f"Number of classes: {results['num_classes']}\n")
            f.write(f"Accuracy: {results['accuracy']:.2f}%\n")
            f.write(f"FPS: {results['fps']:.2f}\n")
            f.write(f"Total inference time: {results['total_time']:.2f} seconds\n\n")
            f.write("Classification Report:\n")
            for cls in results['classes']:
                if cls in results['report']:
                    f.write(f"{cls}:\n")
                    f.write(f"  Precision: {results['report'][cls]['precision']:.4f}\n")
                    f.write(f"  Recall: {results['report'][cls]['recall']:.4f}\n")
                    f.write(f"  F1-score: {results['report'][cls]['f1-score']:.4f}\n")
                    f.write(f"  Support: {results['report'][cls]['support']}\n\n")
    
    # Create comparison table
    comparison_df = create_comparison_table(all_results)
    print("\nComparison Table:")
    print(comparison_df)
    
    print(f"\nAll evaluation results saved to {cfg.RESULTS_DIR}")

if __name__ == "__main__":
    main()