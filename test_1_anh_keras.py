import os
import numpy as np
import time
from PIL import Image
from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

class Config:
    # Model settings
    MODEL_PATH = "models/augmented_training_final_20250908_114654_FINAL.keras"
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
    RESULTS_DIR = Path("results/single_image_test_keras")
    
    def __init__(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

cfg = Config()

def load_model():
    """Load the trained Keras model"""
    print("Đang tải mô hình Keras từ:", cfg.MODEL_PATH)
    
    try:
        # Load model từ file .keras
        model = keras.models.load_model(cfg.MODEL_PATH)
        print(f"Mô hình đã được tải thành công với {cfg.NUM_CLASSES} lớp đầu ra")
        
        # Kiểm tra input shape
        input_shape = model.input_shape
        print(f"Input shape: {input_shape}")
        
        # Kiểm tra output shape
        output_shape = model.output_shape
        print(f"Output shape: {output_shape}")
        
        # Kiểm tra cấu trúc model (chỉ một phần để không quá dài)
        print("Cấu trúc mô hình (một phần):")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
        
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        print("Vui lòng kiểm tra đường dẫn và định dạng file model.")
        return None

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for Keras model - sử dụng cùng preprocessing như training"""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        print(f"Ảnh gốc size: {img.size}")
        
        # Resize image
        img = img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
        print(f"Ảnh sau resize: {img.size}")
        
        # Convert to numpy array
        img_array = np.array(img)
        print(f"Shape ảnh array: {img_array.shape}")
        print(f"Min/Max values: {img_array.min()}/{img_array.max()}")
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Shape sau batch dimension: {img_array.shape}")
        
        # Sử dụng cùng preprocessing như training: rescale=1./255
        img_array = img_array.astype('float32') / 255.0
        print(f"Min/Max values sau normalize: {img_array.min():.4f}/{img_array.max():.4f}")
        
        return img_array, img
        
    except Exception as e:
        print(f"Lỗi khi tải ảnh {image_path}: {e}")
        return None, None

def predict_single_image(model, image_array):
    """Dự đoán cho một ảnh duy nhất với Keras model"""
    try:
        print(f"Input shape cho prediction: {image_array.shape}")
        
        # Đo thời gian inference
        start_time = time.time()
        
        # Thực hiện dự đoán
        predictions = model.predict(image_array, verbose=0)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Raw predictions: {predictions}")
        
        # Lấy xác suất (predictions đã là softmax output)
        probabilities = predictions[0]
        print(f"Probabilities: {probabilities}")
        
        # Lấy lớp dự đoán với xác suất cao nhất
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        print(f"Predicted class index: {predicted_class}")
        print(f"Confidence: {confidence}")
        
        return {
            "predicted_class": int(predicted_class),
            "confidence": float(confidence),
            "probabilities": probabilities.tolist(),
            "inference_time": inference_time,
            "class_name": cfg.CLASS_NAMES[predicted_class],
            "display_name": cfg.DISPLAY_CLASS_NAMES[predicted_class]
        }
        
    except Exception as e:
        print(f"Lỗi khi thực hiện dự đoán: {e}")
        return None

def print_results(prediction_result, image_path):
    """In kết quả dự đoán cho một ảnh"""
    print("\n" + "="*60)
    print("KẾT QUẢ DỰ ĐOÁN CHO MỘT ẢNH (KERAS MODEL)")
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
    with open(cfg.RESULTS_DIR / f"prediction_keras_{image_name}.txt", 'w', encoding='utf-8') as f:
        f.write(f"Model: Keras (.keras)\n")
        f.write(f"Đường dẫn ảnh: {image_path}\n")
        f.write(f"Kết quả dự đoán: {prediction_result['display_name']}\n")
        f.write(f"Lớp dự đoán: {prediction_result['class_name']}\n")
        f.write(f"Độ tin cậy: {prediction_result['confidence']:.4f} ({prediction_result['confidence']*100:.2f}%)\n")
        f.write(f"Thời gian inference: {prediction_result['inference_time']:.4f} giây\n\n")
        f.write("Xác suất cho tất cả các lớp:\n")
        for i, (class_name, display_name, prob) in enumerate(zip(cfg.CLASS_NAMES, cfg.DISPLAY_CLASS_NAMES, prediction_result['probabilities'])):
            f.write(f"{display_name}: {prob:.4f} ({prob*100:.2f}%)\n")
    
    # Lưu kết quả dưới dạng JSON
    result_dict = {
        "model_type": "Keras",
        "model_path": cfg.MODEL_PATH,
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
    
    with open(cfg.RESULTS_DIR / f"prediction_keras_{image_name}.json", 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nKết quả đã được lưu vào {cfg.RESULTS_DIR}")

def main():
    """Hàm chính để dự đoán một ảnh với Keras model"""
    print("Bắt đầu dự đoán cho một ảnh duy nhất với Keras model...")
    
    # Kiểm tra đường dẫn ảnh
    if not os.path.exists(cfg.TEST_IMAGE_PATH):
        print(f"❌ Không tìm thấy ảnh tại đường dẫn: {cfg.TEST_IMAGE_PATH}")
        print("Vui lòng chỉnh sửa TEST_IMAGE_PATH trong Config để trỏ đến ảnh bạn muốn test.")
        return
    
    # Kiểm tra đường dẫn model
    if not os.path.exists(cfg.MODEL_PATH):
        print(f"❌ Không tìm thấy model tại đường dẫn: {cfg.MODEL_PATH}")
        print("Vui lòng chỉnh sửa MODEL_PATH trong Config để trỏ đến model bạn muốn sử dụng.")
        return
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Load và preprocess ảnh
    print(f"Đang tải ảnh: {cfg.TEST_IMAGE_PATH}")
    image_array, original_image = load_and_preprocess_image(cfg.TEST_IMAGE_PATH)
    
    if image_array is None:
        print("❌ Không thể tải ảnh. Vui lòng kiểm tra đường dẫn và định dạng ảnh.")
        return
    
    # Dự đoán
    print("Đang thực hiện dự đoán...")
    prediction_result = predict_single_image(model, image_array)
    
    if prediction_result is None:
        print("❌ Không thể thực hiện dự đoán.")
        return
    
    # In kết quả
    print_results(prediction_result, cfg.TEST_IMAGE_PATH)
    
    # Lưu kết quả
    save_results(prediction_result, cfg.TEST_IMAGE_PATH)

if __name__ == "__main__":
    main()
