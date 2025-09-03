import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Đường dẫn
DATA_DIR = Path("data/augmented")  # Sử dụng dữ liệu đã tăng cường
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
CLASSES = ["Brown_Spot", "Leaf_Blast", "Leaf_Blight", "Healthy"]

# Các tham số
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_test_data():
    """Tải dữ liệu test"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        DATA_DIR / 'test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

def load_all_models():
    """Tải tất cả các mô hình đã huấn luyện"""
    models = {}
    model_files = list(MODELS_DIR.glob('*_best.h5'))
    
    for model_file in model_files:
        model_name = model_file.stem.split('_')[0]
        print(f"Đang tải mô hình: {model_name}")
        try:
            model = load_model(model_file)
            models[model_name] = model
        except Exception as e:
            print(f"Không thể tải mô hình {model_name}: {e}")
    
    return models

def evaluate_model_metrics(model, model_name, test_generator):
    """Đánh giá các metrics của mô hình"""
    # Đo thời gian dự đoán
    start_time = time.time()
    predictions = model.predict(test_generator)
    end_time = time.time()
    
    # Tính toán FPS
    inference_time = end_time - start_time
    num_images = test_generator.samples
    fps = num_images / inference_time
    
    # Tính toán kích thước mô hình
    model_size_mb = os.path.getsize(MODELS_DIR / f'{model_name}_best.h5') / (1024 * 1024)
    
    # Tính toán các metrics
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Tạo dictionary chứa các metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'FPS': fps,
        'Model Size (MB)': model_size_mb,
        'Inference Time (s)': inference_time
    }
    
    return metrics, y_true, y_pred

def plot_comparison_chart(metrics_df):
    """Vẽ biểu đồ so sánh các mô hình"""
    # Vẽ biểu đồ accuracy, precision, recall, f1-score
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Chuyển đổi DataFrame để dễ vẽ
    plot_df = pd.melt(
        metrics_df, 
        id_vars=['Model'], 
        value_vars=metrics_to_plot,
        var_name='Metric', 
        value_name='Value'
    )
    
    sns.barplot(x='Metric', y='Value', hue='Model', data=plot_df)
    plt.title('So sánh các metrics giữa các mô hình')
    plt.ylim(0, 1)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'model_comparison_metrics.png')
    plt.close()
    
    # Vẽ biểu đồ FPS và kích thước mô hình
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Vẽ FPS trên trục y1
    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('FPS', color=color)
    ax1.bar(metrics_df['Model'], metrics_df['FPS'], color=color, alpha=0.7, label='FPS')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Tạo trục y2 cho kích thước mô hình
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Model Size (MB)', color=color)
    ax2.plot(metrics_df['Model'], metrics_df['Model Size (MB)'], 'o-', color=color, linewidth=2, label='Model Size')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Thêm tiêu đề và legend
    fig.tight_layout()
    plt.title('So sánh FPS và kích thước mô hình')
    
    # Thêm legend cho cả hai trục
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.savefig(RESULTS_DIR / 'model_comparison_performance.png')
    plt.close()

def plot_confusion_matrices(all_y_true, all_y_pred, model_names):
    """Vẽ ma trận nhầm lẫn cho tất cả các mô hình"""
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for i, model_name in enumerate(model_names):
        cm = confusion_matrix(all_y_true[i], all_y_pred[i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[i])
        axes[i].set_title(f'{model_name.capitalize()}')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'all_models_confusion_matrices.png')
    plt.close()

def main():
    print("===== ĐÁNH GIÁ VÀ SO SÁNH CÁC MÔ HÌNH =====")
    
    # Tải dữ liệu test
    test_generator = load_test_data()
    print(f"Số lượng ảnh test: {test_generator.samples}")
    
    # Tải tất cả các mô hình
    models = load_all_models()
    if not models:
        print("Không tìm thấy mô hình nào để đánh giá!")
        return
    
    # Đánh giá từng mô hình
    all_metrics = []
    all_y_true = []
    all_y_pred = []
    model_names = []
    
    for model_name, model in models.items():
        print(f"\nĐang đánh giá mô hình: {model_name}")
        metrics, y_true, y_pred = evaluate_model_metrics(model, model_name, test_generator)
        all_metrics.append(metrics)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        model_names.append(model_name)
        
        # In metrics
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        print(f"FPS: {metrics['FPS']:.2f}")
        print(f"Model Size: {metrics['Model Size (MB)']:.2f} MB")
    
    # Tạo DataFrame từ các metrics
    metrics_df = pd.DataFrame(all_metrics)
    
    # Lưu metrics vào file CSV
    metrics_df.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    print(f"\nĐã lưu metrics so sánh vào: {RESULTS_DIR / 'model_comparison.csv'}")
    
    # Vẽ biểu đồ so sánh
    plot_comparison_chart(metrics_df)
    
    # Vẽ ma trận nhầm lẫn cho tất cả các mô hình
    plot_confusion_matrices(all_y_true, all_y_pred, model_names)
    
    print("\nĐã hoàn thành đánh giá và so sánh các mô hình!")
    print(f"Kết quả được lưu trong thư mục: {RESULTS_DIR}")

if __name__ == "__main__":
    main()


