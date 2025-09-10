import os
import numpy as np
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class Config:
    # Model settings
    MODEL_PATH = "models/augmented_training_final_20250908_114654_FINAL.keras"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
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
            "is_flat_structure": True
        }
    ]
    
    # Output settings
    RESULTS_DIR = Path("results/external_evaluation_CNN")
    
    def __init__(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

cfg = Config()

def load_model_cnn():
    """Load the trained CNN model"""
    print("Loading model from:", cfg.MODEL_PATH)
    
    # Tránh lỗi Custom objects với TensorFlow Addons nếu có
    try:
        import tensorflow_addons as tfa
        # Tải model với custom objects cho F1Score
        model = load_model(cfg.MODEL_PATH, custom_objects={
            'F1Score': tfa.metrics.F1Score
        })
    except ImportError:
        # Nếu không có TensorFlow Addons, thử tải model thông thường
        try:
            model = load_model(cfg.MODEL_PATH)
        except Exception as e:
            print(f"Error loading model with standard method: {e}")
            print("Trying to load with compile=False...")
            model = load_model(cfg.MODEL_PATH, compile=False)
            
            # Biên dịch lại model với các metrics cơ bản
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    
    print(f"Model loaded successfully with {cfg.NUM_CLASSES} output classes")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    return model

def create_data_generator(data_dir, class_names=None, is_flat_structure=False):
    """Create data generator for a dataset"""
    datagen = ImageDataGenerator(rescale=1./255)
    
    if is_flat_structure:
        # Xử lý cấu trúc phẳng cho Roboflow
        # Tạo thư mục tạm để đặt ảnh vào cấu trúc thư mục cần thiết
        import tempfile
        import shutil
        from glob import glob
        
        temp_dir = tempfile.mkdtemp()
        blast_dir = Path(temp_dir) / "blast"
        blast_dir.mkdir(exist_ok=True)
        
        # Tìm tất cả các ảnh trong thư mục gốc và các thư mục con
        data_path = Path(data_dir)
        image_paths = list(data_path.glob('*.[jp][pn]g'))
        
        # Nếu không tìm thấy ảnh trong thư mục gốc, tìm trong thư mục con
        if len(image_paths) == 0:
            for subdir in data_path.iterdir():
                if subdir.is_dir():
                    image_paths.extend(list(subdir.glob('*.[jp][pn]g')))
        
        # Copy các ảnh vào thư mục blast
        for img_path in image_paths:
            shutil.copy(img_path, blast_dir)
        
        print(f"Copied {len(image_paths)} images to temporary directory structure")
        
        # Tạo generator với thư mục tạm
        try:
            generator = datagen.flow_from_directory(
                temp_dir,
                target_size=cfg.IMG_SIZE,
                batch_size=cfg.BATCH_SIZE,
                class_mode='categorical',
                shuffle=False
            )
            
            # Lưu thông tin để xóa thư mục tạm sau này
            generator.temp_dir = temp_dir
            generator.original_class_names = ["blast"]
            generator.display_class_names = ["Blast"]
            generator.original_classes = ["blast"]
            generator.is_flat_structure = True
            
            return generator
        
        except Exception as e:
            # Xóa thư mục tạm nếu có lỗi
            shutil.rmtree(temp_dir)
            print(f"Error creating generator for flat structure: {e}")
            return None
    else:
        # Xử lý cấu trúc thông thường
        try:
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=cfg.IMG_SIZE,
                batch_size=cfg.BATCH_SIZE,
                class_mode='categorical',
                shuffle=False
            )
            
            # Lấy các tên lớp từ generator
            original_classes = list(generator.class_indices.keys())
            
            # Map tên lớp gốc sang tên hiển thị
            display_class_names = [cfg.CLASS_MAPPING.get(cls, cls) for cls in original_classes]
            
            # Thêm thông tin vào generator để sử dụng sau
            generator.original_class_names = original_classes
            generator.display_class_names = display_class_names
            generator.original_classes = original_classes
            generator.is_flat_structure = False
            
            return generator
            
        except Exception as e:
            print(f"Error creating generator: {e}")
            return None

def evaluate_dataset(model, data_generator, dataset_name):
    """Evaluate model on a dataset"""
    print(f"Evaluating on {dataset_name}...")
    
    # Kiểm tra nếu không có dữ liệu
    if data_generator is None or data_generator.samples == 0:
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
    
    # Chuẩn bị biến để thu thập kết quả
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Tính toán số batches
    steps = len(data_generator)
    
    # Đo thời gian
    total_time = 0
    total_images = 0
    
    # Dự đoán trên từng batch
    for i in tqdm(range(steps), desc=f"Testing {dataset_name}"):
        # Lấy batch
        batch_x, batch_y = data_generator[i]
        batch_size = batch_x.shape[0]
        total_images += batch_size
        
        # Đo thời gian inference
        start_time = time.time()
        batch_preds = model.predict(batch_x, verbose=0)
        end_time = time.time()
        
        # Tích lũy thời gian
        batch_time = end_time - start_time
        total_time += batch_time
        
        # Thu thập kết quả
        all_probs.extend(batch_preds)
        
        # Chuyển đổi one-hot vectors thành class indices
        batch_y_indices = np.argmax(batch_y, axis=1)
        batch_pred_indices = np.argmax(batch_preds, axis=1)
        
        all_labels.extend(batch_y_indices)
        all_preds.extend(batch_pred_indices)
    
    # Tính FPS
    fps = total_images / total_time if total_time > 0 else 0
    
    # Tính độ chính xác
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    # Lấy các nhãn duy nhất
    unique_labels = sorted(np.unique(all_labels))
    
    # Lấy tên lớp hiển thị
    class_names = data_generator.display_class_names
    display_classes = [class_names[i] for i in unique_labels]
    
    # Tạo classification report
    report = classification_report(
        all_labels, 
        all_preds,
        labels=unique_labels,
        target_names=display_classes,
        output_dict=True,
        zero_division=0
    )
    
    # Tính confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Xóa thư mục tạm nếu đã tạo
    if hasattr(data_generator, 'temp_dir') and data_generator.is_flat_structure:
        import shutil
        try:
            shutil.rmtree(data_generator.temp_dir)
            print(f"Removed temporary directory: {data_generator.temp_dir}")
        except Exception as e:
            print(f"Error removing temporary directory: {e}")
    
    # Tạo kết quả
    results = {
        "dataset_name": dataset_name,
        "accuracy": accuracy,
        "fps": fps,
        "report": report,
        "confusion_matrix": cm,
        "classes": display_classes,
        "original_classes": [data_generator.original_classes[i] for i in unique_labels],
        "num_classes": len(display_classes),
        "total_images": data_generator.samples,
        "total_time": total_time
    }
    
    print(f"{dataset_name} - Accuracy: {accuracy:.2f}% - FPS: {fps:.2f}")
    return results

def create_comparison_table(all_results):
    """Create a comparison table without Total no of images and with correct No of diseases"""
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
            "No of diseases": len(disease_classes),
            "Names of diseases": ", ".join([cfg.CLASS_MAPPING.get(cls, cls) for cls in expected_classes]),
            "Accuracy(%)": f"{result['accuracy']:.2f}",
            "FPS": f"{result['fps']:.2f}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(cfg.RESULTS_DIR / "comparison_results_CNN.csv", index=False)
    
    # Create HTML version for better visualization
    html = df.to_html(index=False)
    with open(cfg.RESULTS_DIR / "comparison_results_CNN.html", 'w') as f:
        f.write(html)
    
    return df

def main():
    # Tải model
    model = load_model_cnn()
    
    all_results = []
    
    # Đánh giá từng dataset
    for dataset_info in cfg.EXTERNAL_DATASETS:
        # Tạo data generator cho dataset
        data_generator = create_data_generator(
            dataset_info["path"],
            class_names=dataset_info.get("classes"),
            is_flat_structure=dataset_info.get("is_flat_structure", False)
        )
        
        # Đánh giá model trên dataset
        results = evaluate_dataset(model, data_generator, dataset_info["name"])
        all_results.append(results)
        
        # Lưu báo cáo chi tiết
        with open(cfg.RESULTS_DIR / f"report_CNN_{dataset_info['name'].replace(' ', '_')}.txt", 'w') as f:
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
    
    # Tạo bảng so sánh
    comparison_df = create_comparison_table(all_results)
    print("\nComparison Table:")
    print(comparison_df)
    
    print(f"\nAll evaluation results saved to {cfg.RESULTS_DIR}")

if __name__ == "__main__":
    main()