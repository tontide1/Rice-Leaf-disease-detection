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
    NUM_CLASSES = 4  # Số lớp cố định: 3 lớp bệnh + 1 lớp khỏe mạnh
    
    # Đường dẫn đến ảnh cần test
    TEST_IMAGE_PATH = "data/test_data_deeplearning/DoMinhHuy/blast/leaf_blast5.jpg"  # Thay đổi đường dẫn này thành ảnh bạn muốn test
    
    
    
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
    
    # Tên các lớp theo thứ tự
    CLASS_NAMES = ["bacterial_leaf_blight", "blast", "brown_spot", "normal"]
    DISPLAY_CLASS_NAMES = ["Bacterial Leaf Blight", "Blast", "Brown Spot", "Healthy"]
    
    # Output settings
    RESULTS_DIR = Path("results/single_image_test")
    
    def __init__(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

cfg = Config()

def load_model():
    """Load the trained ResNet50 model"""
    print("Đang tải mô hình từ:", 'optimized_svm_classifier_20250908_220820_FINAL.pkl')
    
    # Tạo ResNetModel theo cấu trúc training
    class ResNetModel(nn.Module):
        def __init__(self, num_classes):
            super(ResNetModel, self).__init__()
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
            
        print(f"Mô hình đã được tải thành công với {cfg.NUM_CLASSES} lớp đầu ra")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        print("Tiếp tục với mô hình chưa khởi tạo - kết quả sẽ ngẫu nhiên!")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Sử dụng {device} cho inference")
    return model, device

def get_transforms():
    """Get transforms for evaluation"""
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        transform = get_transforms()
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor, image
    except Exception as e:
        print(f"Lỗi khi tải ảnh {image_path}: {e}")
        return None, None

def predict_single_image(model, image_tensor, device):
    """Dự đoán cho một ảnh duy nhất"""
    model.eval()
    
    with torch.no_grad():
        # Chuyển ảnh lên device
        image_tensor = image_tensor.to(device)
        
        # Đo thời gian inference
        start_time = time.time()
        outputs = model(image_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        # Lấy xác suất bằng softmax
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Lấy lớp dự đoán với xác suất cao nhất
        confidence, predicted_class = torch.max(probs, 1)
        
        # Chuyển về CPU để xử lý
        confidence = confidence.cpu().numpy()[0]
        predicted_class = predicted_class.cpu().numpy()[0]
        probabilities = probs.cpu().numpy()[0]
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "inference_time": inference_time,
            "class_name": cfg.CLASS_NAMES[predicted_class],
            "display_name": cfg.DISPLAY_CLASS_NAMES[predicted_class]
        }

def print_results(prediction_result, image_path):
    """In kết quả dự đoán cho một ảnh"""
    print("\n" + "="*60)
    print("KẾT QUẢ DỰ ĐOÁN CHO MỘT ẢNH")
    print("="*60)
    
    print(f"Đường dẫn ảnh: {image_path}")
    print(f"Kết quả dự đoán: {prediction_result['display_name']}")
    print(f"Độ tin cậy: {prediction_result['confidence']:.4f} ({prediction_result['confidence']*100:.2f}%)")
    print(f"Thời gian inference: {prediction_result['inference_time']:.4f} giây")
    
    print("\n" + "-"*40)
    print("XÁC SUẤT CHO TẤT CẢ CÁC LỚP")
    print("-"*40)
    
    for i, (class_name, display_name, prob) in enumerate(zip(cfg.CLASS_NAMES, cfg.DISPLAY_CLASS_NAMES, prediction_result['probabilities'])):
        print(f"{display_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    print("\n" + "-"*40)
    print("ĐÁNH GIÁ KẾT QUẢ")
    print("-"*40)
    
    confidence = prediction_result['confidence']
    if confidence > 0.8:
        print("✅ Dự đoán có độ tin cậy CAO")
    elif confidence > 0.6:
        print("⚠️  Dự đoán có độ tin cậy TRUNG BÌNH")
    else:
        print("❌ Dự đoán có độ tin cậy THẤP")

def save_results(prediction_result, image_path):
    """Lưu kết quả dự đoán vào file"""
    # Tạo tên file từ đường dẫn ảnh
    image_name = Path(image_path).stem
    
    # Lưu báo cáo chi tiết
    with open(cfg.RESULTS_DIR / f"prediction_{image_name}.txt", 'w', encoding='utf-8') as f:
        f.write(f"Đường dẫn ảnh: {image_path}\n")
        f.write(f"Kết quả dự đoán: {prediction_result['display_name']}\n")
        f.write(f"Lớp dự đoán: {prediction_result['class_name']}\n")
        f.write(f"Độ tin cậy: {prediction_result['confidence']:.4f} ({prediction_result['confidence']*100:.2f}%)\n")
        f.write(f"Thời gian inference: {prediction_result['inference_time']:.4f} giây\n\n")
        f.write("Xác suất cho tất cả các lớp:\n")
        for i, (class_name, display_name, prob) in enumerate(zip(cfg.CLASS_NAMES, cfg.DISPLAY_CLASS_NAMES, prediction_result['probabilities'])):
            f.write(f"{display_name}: {prob:.4f} ({prob*100:.2f}%)\n")
    
    # Lưu kết quả dưới dạng JSON
    import json
    result_dict = {
        "image_path": image_path,
        "predicted_class": prediction_result['class_name'],
        "predicted_display_name": prediction_result['display_name'],
        "confidence": float(prediction_result['confidence']),
        "inference_time": float(prediction_result['inference_time']),
        "probabilities": {
            cfg.DISPLAY_CLASS_NAMES[i]: float(prob) 
            for i, prob in enumerate(prediction_result['probabilities'])
        }
    }
    
    with open(cfg.RESULTS_DIR / f"prediction_{image_name}.json", 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nKết quả đã được lưu vào {cfg.RESULTS_DIR}")

def main():
    """Hàm chính để dự đoán một ảnh"""
    print("Bắt đầu dự đoán cho một ảnh duy nhất...")
    
    # Kiểm tra đường dẫn ảnh
    if not os.path.exists(cfg.TEST_IMAGE_PATH):
        print(f"❌ Không tìm thấy ảnh tại đường dẫn: {cfg.TEST_IMAGE_PATH}")
        print("Vui lòng chỉnh sửa TEST_IMAGE_PATH trong Config để trỏ đến ảnh bạn muốn test.")
        return
    
    # Load model
    model, device = load_model()
    
    # Load và preprocess ảnh
    print(f"Đang tải ảnh: {cfg.TEST_IMAGE_PATH}")
    image_tensor, original_image = load_and_preprocess_image(cfg.TEST_IMAGE_PATH)
    
    if image_tensor is None:
        print("❌ Không thể tải ảnh. Vui lòng kiểm tra đường dẫn và định dạng ảnh.")
        return
    
    # Dự đoán
    print("Đang thực hiện dự đoán...")
    prediction_result = predict_single_image(model, image_tensor, device)
    
    # In kết quả
    print_results(prediction_result, cfg.TEST_IMAGE_PATH)
    
    # Lưu kết quả
    save_results(prediction_result, cfg.TEST_IMAGE_PATH)

if __name__ == "__main__":
    main()
