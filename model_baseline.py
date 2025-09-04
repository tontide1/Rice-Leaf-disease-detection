import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import datetime
import time

# Hàm tạo timestamp cho tên file và thư mục
def get_timestamp():
    """Tạo timestamp dạng YYYYMMDD_HHMMSS để sử dụng trong tên file và thư mục"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Timestamp cho phiên huấn luyện hiện tại
TIMESTAMP = get_timestamp()

# Đường dẫn
DATA_DIR = Path("data/augmented")  # Sử dụng dữ liệu đã tăng cường
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
CLASSES = ["Brown_Spot", "Leaf_Blast", "Leaf_Blight", "Healthy"]

# Tạo thư mục lưu mô hình và kết quả với timestamp
MODEL_RUN_DIR = MODELS_DIR / f"run_{TIMESTAMP}"
RESULT_RUN_DIR = RESULTS_DIR / f"run_{TIMESTAMP}"

# Tạo thư mục lưu mô hình và kết quả
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_RUN_DIR.mkdir(exist_ok=True)
RESULT_RUN_DIR.mkdir(exist_ok=True)

# Các tham số
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = len(CLASSES)

def create_data_generators():
    """Tạo data generators cho việc huấn luyện"""
    # Tăng cường dữ liệu cho tập train
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Chỉ rescale cho tập validation và test
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load dữ liệu
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR / 'train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        DATA_DIR / 'val',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        DATA_DIR / 'test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def create_baseline_model():
    """Tạo mô hình CNN cơ bản"""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Biên dịch mô hình
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, val_generator):
    """Huấn luyện mô hình"""
    # Tạo tên file với timestamp
    best_model_path = MODEL_RUN_DIR / f'baseline_best_{TIMESTAMP}.h5'
    final_model_path = MODEL_RUN_DIR / f'baseline_final_{TIMESTAMP}.h5'
    
    # Thiết lập callbacks
    checkpoint = ModelCheckpoint(
        best_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Tạo TensorBoard callback với timestamp
    log_dir = Path("logs") / "fit" / f"run_{TIMESTAMP}"
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
    )
    
    print(f"Logs sẽ được lưu tại: {log_dir}")
    print(f"Model tốt nhất sẽ được lưu tại: {best_model_path}")
    print(f"Model cuối cùng sẽ được lưu tại: {final_model_path}")
    
    # Huấn luyện mô hình
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    )
    
    # Lưu mô hình cuối cùng
    model.save(final_model_path)
    
    return history

def evaluate_model(model, test_generator):
    """Đánh giá mô hình trên tập test"""
    # Đánh giá mô hình
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Dự đoán trên tập test
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Tạo tên file kết quả với timestamp
    results_file = RESULT_RUN_DIR / f'baseline_results_{TIMESTAMP}.txt'
    
    # Lưu kết quả dự đoán
    with open(results_file, 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    
    print(f"Kết quả đánh giá đã được lưu tại: {results_file}")
    
    return y_true, y_pred

def plot_training_history(history):
    """Vẽ biểu đồ lịch sử huấn luyện"""
    plt.figure(figsize=(12, 5))
    
    # Vẽ biểu đồ accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Vẽ biểu đồ loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Tạo tên file biểu đồ với timestamp
    history_plot_path = RESULT_RUN_DIR / f'baseline_training_history_{TIMESTAMP}.png'
    
    plt.tight_layout()
    plt.savefig(history_plot_path)
    plt.close()
    
    print(f"Biểu đồ lịch sử huấn luyện đã được lưu tại: {history_plot_path}")

def plot_confusion_matrix(y_true, y_pred):
    """Vẽ ma trận nhầm lẫn"""
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    # Tính ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    
    cm_path = RESULT_RUN_DIR / f'baseline_confusion_matrix_{TIMESTAMP}.png'
    report_path = RESULT_RUN_DIR / f'baseline_classification_report_{TIMESTAMP}.txt'
    
    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print("Classification Report:")
    print(report)
    
    # Lưu báo cáo phân loại
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Ma trận nhầm lẫn đã được lưu tại: {cm_path}")
    print(f"Báo cáo phân loại đã được lưu tại: {report_path}")

def main():
    print("===== HUẤN LUYỆN MÔ HÌNH CNN CƠ BẢN =====")
    print(f"Timestamp hiện tại: {TIMESTAMP}")
    print(f"Thư mục lưu mô hình: {MODEL_RUN_DIR}")
    print(f"Thư mục lưu kết quả: {RESULT_RUN_DIR}")
    
    # Tạo data generators
    train_generator, val_generator, test_generator = create_data_generators()
    print(f"Số lượng ảnh train: {train_generator.samples}")
    print(f"Số lượng ảnh validation: {val_generator.samples}")
    print(f"Số lượng ảnh test: {test_generator.samples}")
    
    # Tạo mô hình
    model = create_baseline_model()
    model.summary()
    
    # Huấn luyện mô hình
    history = train_model(model, train_generator, val_generator)
    
    # Đánh giá mô hình
    y_true, y_pred = evaluate_model(model, test_generator)
    
    # Vẽ biểu đồ
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred)
    
    print("\nĐã hoàn thành huấn luyện và đánh giá mô hình CNN cơ bản!")
    print(f"Kết quả được lưu trong thư mục: {RESULT_RUN_DIR}")
    print(f"Mô hình được lưu trong thư mục: {MODEL_RUN_DIR}")

if __name__ == "__main__":
    main()
