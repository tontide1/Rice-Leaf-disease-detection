import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from tqdm import tqdm

# Đường dẫn tới dữ liệu đã phân chia
DATA_DIR = Path("data/splits")
PROCESSED_DIR = Path("data/processed")
AUGMENTED_DIR = Path("data/augmented")
CLASSES = ["Brown_Spot", "Leaf_Blast", "Leaf_Blight", "Healthy"]
SPLITS = ["train", "val", "test"]

# Kích thước chuẩn hóa ảnh - dựa trên phân tích kích thước trung bình (241x241)
TARGET_SIZE = (224, 224)  # Kích thước phổ biến cho nhiều mô hình CNN

def create_directories():
    """Tạo thư mục cho dữ liệu đã tiền xử lý và tăng cường"""
    for dir_path in [PROCESSED_DIR, AUGMENTED_DIR]:
        for split in SPLITS:
            for cls in CLASSES:
                path = dir_path / split / cls
                path.mkdir(parents=True, exist_ok=True)
                print(f"Đã tạo thư mục: {path}")

def preprocess_images():
    """Tiền xử lý ảnh: thay đổi kích thước, chuẩn hóa, tăng cường độ tương phản"""
    total_processed = 0
    
    for split in SPLITS:
        for cls in CLASSES:
            src_dir = DATA_DIR / split / cls
            dst_dir = PROCESSED_DIR / split / cls
            
            if not src_dir.exists():
                print(f"Thư mục không tồn tại: {src_dir}")
                continue
                
            files = list(src_dir.glob("*"))
            print(f"Đang xử lý {len(files)} ảnh từ {split}/{cls}...")
            
            for img_path in tqdm(files, desc=f"{split}/{cls}"):
                try:
                    # Đọc ảnh
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"Không thể đọc ảnh: {img_path}")
                        continue
                    
                    # Thay đổi kích thước với phương pháp nội suy tốt hơn cho ảnh nhỏ
                    # Sử dụng INTER_CUBIC cho ảnh nhỏ hơn TARGET_SIZE, INTER_AREA cho ảnh lớn hơn
                    h, w = img.shape[:2]
                    if h < TARGET_SIZE[0] or w < TARGET_SIZE[1]:
                        # Nội suy cubic tốt hơn cho phóng to
                        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
                    else:
                        # Nội suy area tốt hơn cho thu nhỏ
                        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                    
                    # Chuẩn hóa độ sáng và tương phản
                    # Chuyển sang không gian màu LAB để xử lý kênh sáng
                    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    
                    # Ghép lại các kênh
                    lab = cv2.merge((l, a, b))
                    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    
                    # Giảm nhiễu (tùy chọn)
                    img_enhanced = cv2.fastNlMeansDenoisingColored(img_enhanced, None, 10, 10, 7, 21)
                    
                    # Tăng độ sắc nét (tùy chọn)
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    img_enhanced = cv2.filter2D(img_enhanced, -1, kernel)
                    
                    # Lưu ảnh đã xử lý
                    output_path = dst_dir / img_path.name
                    cv2.imwrite(str(output_path), img_enhanced)
                    total_processed += 1
                    
                except Exception as e:
                    print(f"Lỗi khi xử lý {img_path}: {e}")
    
    print(f"Đã xử lý tổng cộng {total_processed} ảnh")

def visualize_preprocessing():
    """Hiển thị một số ảnh trước và sau khi xử lý"""
    plt.figure(figsize=(16, 12))
    
    for i, cls in enumerate(CLASSES):
        # Lấy một ảnh mẫu
        orig_files = list((DATA_DIR / "train" / cls).glob("*"))
        if not orig_files:
            continue
            
        sample_file = orig_files[0].name
        orig_path = DATA_DIR / "train" / cls / sample_file
        proc_path = PROCESSED_DIR / "train" / cls / sample_file
        
        if orig_path.exists() and proc_path.exists():
            # Đọc ảnh gốc và ảnh đã xử lý
            img_orig = cv2.imread(str(orig_path))
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            
            img_proc = cv2.imread(str(proc_path))
            img_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
            
            # Hiển thị ảnh
            plt.subplot(len(CLASSES), 4, i*4 + 1)
            plt.imshow(img_orig)
            plt.title(f"{cls} - Gốc")
            plt.axis('off')
            
            plt.subplot(len(CLASSES), 4, i*4 + 2)
            plt.imshow(img_proc)
            plt.title(f"{cls} - Đã xử lý")
            plt.axis('off')
            
            # Hiển thị histogram
            plt.subplot(len(CLASSES), 4, i*4 + 3)
            for channel, color in enumerate(['r', 'g', 'b']):
                hist = cv2.calcHist([img_orig], [channel], None, [256], [0, 256])
                plt.plot(hist, color=color)
            plt.title("Histogram - Gốc")
            plt.xlim([0, 256])
            
            plt.subplot(len(CLASSES), 4, i*4 + 4)
            for channel, color in enumerate(['r', 'g', 'b']):
                hist = cv2.calcHist([img_proc], [channel], None, [256], [0, 256])
                plt.plot(hist, color=color)
            plt.title("Histogram - Đã xử lý")
            plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png')
    plt.close()


def main():
    print("===== TIỀN XỬ LÝ VÀ TĂNG CƯỜNG DỮ LIỆU NÂNG CAO =====")
    create_directories()
    preprocess_images()
    visualize_preprocessing()
    print("\nĐã hoàn thành tiền xử lý và tăng cường dữ liệu!")

if __name__ == "__main__":
    main()
