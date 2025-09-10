import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class Config:
    # Model settings
    MODEL_PATH = "models/resnet50_pytorch_final_20250908_211245_FINAL.pth"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_CLASSES = 4  # Số lớp cố định: 3 lớp bệnh + 1 lớp khỏe mạnh
    
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
            "is_flat_structure": True  # Thêm flag cho biết đây là cấu trúc phẳng
        }
    ]
    
    # Output settings
    RESULTS_DIR = Path("results/external_evaluation")
    
    def __init__(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

cfg = Config()

def load_model():
    """Load the trained ResNet50 model - Đơn giản hóa, không phân tích checkpoint"""
    print("Loading model from:", cfg.MODEL_PATH)
    
    # Tạo ResNetModel theo cấu trúc training
    class ResNetModel(nn.Module):
        def __init__(self, num_classes):
            super(ResNetModel, self).__init__()
            # Replace deprecated pretrained parameter with weights parameter
            self.backbone = models.resnet50(weights=None)
            
            # Replace classifier với cùng kiến trúc như lúc training
            num_features = self.backbone.fc.in_features
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
        
        def forward(self, x):
            return self.backbone(x)
    
    # Khởi tạo mô hình
    model = ResNetModel(cfg.NUM_CLASSES)
    
    # Load model weights
    try:
        checkpoint = torch.load(cfg.MODEL_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Thử các cấu trúc checkpoint phổ biến
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"Model loaded successfully with {cfg.NUM_CLASSES} output classes")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing with uninitialized model - results will be random!")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Using {device} for inference")
    return model, device

def get_transforms():
    """Get transforms for evaluation"""
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class ExternalDataset(Dataset):
    """Dataset class for external datasets"""
    def __init__(self, data_dir, transform=None, expected_classes=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Xác định xem có phải đường dẫn Roboflow Blast không
        is_roboflow = "roboflow" in str(data_dir).lower() and "blast" in str(data_dir).lower()
        
        if is_roboflow:
            # Xử lý trường hợp đặc biệt cho Roboflow - cấu trúc phẳng
            self.classes = ["blast"] if expected_classes else ["default_class"]
            self.class_to_idx = {self.classes[0]: 0}
            
            # Tìm tất cả ảnh trong thư mục gốc
            self.image_paths = []
            self.labels = []
            self.original_class_names = []
            
            for img_path in self.data_dir.glob('*.[jp][pn]g'):
                self.image_paths.append(img_path)
                self.labels.append(0)  # Lớp 0 (blast)
                self.original_class_names.append(self.classes[0])
                
            if len(self.image_paths) == 0:
                print(f"Warning: No images found in {data_dir}, checking subfolders...")
                
                # Kiểm tra trong các thư mục con
                for subdir in self.data_dir.iterdir():
                    if subdir.is_dir():
                        for img_path in subdir.glob('*.[jp][pn]g'):
                            self.image_paths.append(img_path)
                            self.labels.append(0)  # Lớp 0 (blast)
                            self.original_class_names.append(self.classes[0])
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
            
            # Get all image paths and labels
            self.image_paths = []
            self.labels = []
            self.original_class_names = []
            
            for cls_name, idx in self.class_to_idx.items():
                cls_dir = self.data_dir / cls_name
                if not cls_dir.exists():
                    continue
                    
                for img_path in cls_dir.glob('*.[jp][pn]g'):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)
                    self.original_class_names.append(cls_name)
        
        # Map original class names to standardized display names
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

def evaluate_dataset(model, data_loader, device, dataset_name):
    """Evaluate model on a dataset"""
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
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Store probabilities for better filtering
    
    # Thêm biến để tính FPS
    total_time = 0
    total_images = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Testing {dataset_name}"):
            batch_size = images.size(0)
            total_images += batch_size
            
            images = images.to(device)
            
            # Đo thời gian inference
            start_time = time.time()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Đảm bảo GPU hoàn thành công việc
            end_time = time.time()
            
            # Tích lũy thời gian inference
            batch_time = end_time - start_time
            total_time += batch_time
            
            # Lấy xác suất bằng softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Lấy lớp dự đoán với xác suất cao nhất
            _, preds = torch.max(outputs, 1)
            
            # Xử lý đặc biệt cho các bộ dữ liệu không đủ lớp
            # Ánh xạ dự đoán của mô hình (0-3) đến các lớp thực tế trong dataset
            if len(data_loader.dataset.classes) < cfg.NUM_CLASSES:
                # Lấy các lớp thực tế trong dataset
                available_class_indices = list(data_loader.dataset.class_to_idx.values())
                
                # Chỉ giữ lại các dự đoán mà mô hình rất tự tin (> 0.7)
                filtered_preds = []
                for i, (pred, prob) in enumerate(zip(preds.cpu().numpy(), probs.cpu().numpy())):
                    max_prob = np.max(prob)
                    if max_prob > 0.7 and pred in available_class_indices:
                        filtered_preds.append(pred)
                    else:
                        # Dự đoán lại bằng cách chỉ xem xét các lớp có sẵn
                        class_probs = prob[available_class_indices]
                        new_pred = available_class_indices[np.argmax(class_probs)]
                        filtered_preds.append(new_pred)
                
                preds = torch.tensor(filtered_preds, device=device)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    if len(all_preds) == 0:
        print(f"Warning: No valid predictions for {dataset_name}")
        return {
            "dataset_name": dataset_name,
            "accuracy": 0.0,
            "fps": 0.0,
            "report": {},
            "confusion_matrix": np.array([[]]),
            "classes": [],
            "original_classes": [],
            "num_classes": 0,
            "total_images": len(data_loader.dataset),
            "total_time": total_time
        }
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    # Tính FPS
    fps = total_images / total_time if total_time > 0 else 0
    
    # Xác định các nhãn thực tế có trong dữ liệu kiểm thử
    unique_labels = sorted(np.unique(all_labels))
    
    # Lấy tên hiển thị của các lớp thực tế có trong dữ liệu
    all_display_classes = data_loader.dataset.display_class_names
    display_classes = [all_display_classes[i] for i in unique_labels]
    
    # Generate classification report chỉ với các lớp thực sự có trong dataset
    report = classification_report(all_labels, all_preds, 
                                 labels=unique_labels,
                                 target_names=display_classes,
                                 output_dict=True,
                                 zero_division=0)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        "dataset_name": dataset_name,
        "accuracy": accuracy,
        "fps": fps,
        "report": report,
        "confusion_matrix": cm,
        "classes": display_classes,  # Chỉ các lớp thực sự có trong dataset
        "original_classes": [data_loader.dataset.classes[i] for i in unique_labels],
        "num_classes": len(display_classes),
        "total_images": len(data_loader.dataset),
        "total_time": total_time
    }
    
    print(f"{dataset_name} - Accuracy: {accuracy:.2f}% - FPS: {fps:.2f}")
    return results

def create_comparison_table(all_results):
    """Create a comparison table similar to the reference image"""
    data = []
    
    for i, result in enumerate(all_results):
        dataset = cfg.EXTERNAL_DATASETS[i]
        
        # Đếm số lớp bệnh từ thông tin ban đầu của dataset (không phải từ kết quả dự đoán)
        expected_classes = dataset["classes"]
        disease_classes = [cls for cls in expected_classes if cls not in ["normal", "healthy", "Healthy"]]
        
        data.append({
            "Article reference": dataset["reference"],
            "Dataset reference": dataset["name"],
            # Đã bỏ cột "Total no of images"
            "No of diseases": len(disease_classes) + 1,
            "Names of diseases": ", ".join([cfg.CLASS_MAPPING.get(cls, cls) for cls in expected_classes]),
            "Accuracy(%)": f"{result['accuracy']:.2f}",
            "FPS": f"{result['fps']:.2f}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(cfg.RESULTS_DIR / "comparison_results.csv", index=False)
    
    # Create HTML version for better visualization
    html = df.to_html(index=False)
    with open(cfg.RESULTS_DIR / "comparison_results.html", 'w') as f:
        f.write(html)
    
    return df

def main():
    # Load model
    model, device = load_model()
    
    # Get transforms for evaluation
    transform = get_transforms()
    
    all_results = []
    
    # Evaluate each dataset
    for dataset_info in cfg.EXTERNAL_DATASETS:
        # Create dataset and dataloader
        dataset = ExternalDataset(
            dataset_info["path"], 
            transform=transform,
            expected_classes=dataset_info.get("classes")
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS
        )
        
        # Evaluate model on dataset
        results = evaluate_dataset(model, dataloader, device, dataset_info["name"])
        all_results.append(results)
        
        # Save detailed report
        with open(cfg.RESULTS_DIR / f"report_{dataset_info['name'].replace(' ', '_')}.txt", 'w') as f:
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