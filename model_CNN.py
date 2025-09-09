import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import datetime
import json
import pandas as pd

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed precision (nếu GPU hỗ trợ)
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")
except:
    print("Mixed precision not available")

# Đường dẫn với timestamp
DATA_DIR = Path("/kaggle/input/deep-learning-data-set/processed")
MODELS_DIR = Path("models")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"results/run_{timestamp}")
LOGS_DIR = Path(f"logs/run_{timestamp}")

# Tạo thư mục
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Các tham số
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 35

def create_data_generators():
    """Tạo data generators: processed cho training (với augmentation), val/test"""
    # Training với augmentation, nguồn từ dữ liệu processed
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation và test chỉ rescale từ dữ liệu processed
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load dữ liệu training từ processed
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR / 'train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )
    
    # Load dữ liệu validation từ processed
    val_generator = val_datagen.flow_from_directory(
        DATA_DIR / 'val',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=SEED
    )
    
    # Load dữ liệu test từ processed
    test_generator = test_datagen.flow_from_directory(
        DATA_DIR / 'test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=SEED
    )
    
    # Lưu mapping nhãn
    with open(RESULTS_DIR / 'class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f, indent=2)
    
    print("Class indices:", train_generator.class_indices)
    
    return train_generator, val_generator, test_generator

def create_optimized_model(num_classes):
    """Tạo mô hình CNN tối ưu (giảm complexity)"""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3 (cuối cùng)
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # Global Average Pooling
        GlobalAveragePooling2D(),
        
        # Giảm Dense layers (nhẹ hơn, ít tham số hơn)
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', dtype='float32')  # float32 cho output
    ])
    
    return model

def train_model(model, train_generator, val_generator):
    """Huấn luyện mô hình với callbacks tối ưu"""
    # Label smoothing
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    
    # Compile với F1 score
    try:
        import tensorflow_addons as tfa
        metrics = ['accuracy', tfa.metrics.F1Score(num_classes=len(train_generator.class_indices), average='macro')]
    except ImportError:
        print("tensorflow_addons not available, using accuracy only")
        metrics = ['accuracy']
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=loss,
        metrics=metrics
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        str(MODELS_DIR / f'augmented_training_best_{timestamp}.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,  # từ 15 -> 10
        restore_best_weights=True,
        verbose=1
    )
    
    # Chỉ dùng ReduceLROnPlateau (bỏ LearningRateScheduler)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=1
    )
    
    # TensorBoard với histogram_freq=0 để tiết kiệm RAM
    tensorboard_callback = TensorBoard(
        log_dir=str(LOGS_DIR),
        histogram_freq=0,
        write_graph=True,
        write_images=False
    )
    
    # Huấn luyện (bỏ steps_per_epoch và validation_steps)
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    )
    
    # Lưu mô hình cuối cùng
    model.save(str(MODELS_DIR / f'augmented_training_final_{timestamp}.keras'))
    
    return history

def evaluate_and_report(model, generator, set_name, target_names):
    """Đánh giá mô hình trên một tập dữ liệu cụ thể và tạo báo cáo."""
    print(f"\n===== ĐÁNH GIÁ TRÊN TẬP {set_name.upper()} =====")
    
    # Đánh giá mô hình
    metrics = model.evaluate(generator, verbose=0)
    # Tạo một dict để dễ truy cập
    metrics_dict = {name: val for name, val in zip(model.metrics_names, metrics)}
    
    # Tìm key cho accuracy một cách linh hoạt
    accuracy_key = next((key for key in metrics_dict if 'accuracy' in key), None)

    print(f"Loss: {metrics_dict['loss']:.4f}")
    if accuracy_key:
        print(f"Accuracy: {metrics_dict[accuracy_key]:.4f}")
    if 'f1_score' in metrics_dict:
        print(f"F1 Score (Macro): {metrics_dict['f1_score']:.4f}")

    # Dự đoán
    predictions = model.predict(generator, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = generator.classes
    
    # Lưu kết quả dự đoán
    results_df = pd.DataFrame({
        "file": generator.filenames,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": np.max(predictions, axis=1)
    })
    results_df.to_csv(RESULTS_DIR / f'predictions_{set_name}.csv', index=False)
    
    # Lưu metrics
    with open(RESULTS_DIR / f'results_{set_name}.txt', 'w') as f:
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Results for: {set_name} set\n")
        for name, val in metrics_dict.items():
            f.write(f"{name.capitalize()}: {val:.4f}\n")
        f.write(f"Total parameters: {model.count_params():,}\n")

    # Vẽ ma trận nhầm lẫn và báo cáo phân loại
    plot_confusion_matrix(y_true, y_pred, target_names, set_name)

def plot_training_history(history):
    """Vẽ biểu đồ lịch sử huấn luyện"""
    plt.figure(figsize=(15, 5))
    
    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Learning rate
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'augmented_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, target_names, set_name):
    """Vẽ ma trận nhầm lẫn với target_names từ generator"""
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    # Tính ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    
    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {set_name.capitalize()} Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'confusion_matrix_{set_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Báo cáo phân loại
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(f"\nClassification Report ({set_name.capitalize()} Set):")
    print(report)
    
    # Lưu báo cáo
    with open(RESULTS_DIR / f'classification_report_{set_name}.txt', 'w') as f:
        f.write(report)

def main():
    print("===== HUẤN LUYỆN MÔ HÌNH CNN VỚI DỮ LIỆU PROCESSED =====")
    print("Training: Processed data (with on-the-fly augmentation)")
    print("Validation/Test: Processed data")
    
    # Tạo data generators
    train_generator, val_generator, test_generator = create_data_generators()
    print(f"Số lượng ảnh train (processed): {train_generator.samples}")
    print(f"Số lượng ảnh validation (processed): {val_generator.samples}")
    print(f"Số lượng ảnh test (processed): {test_generator.samples}")

    # Tạo generator riêng cho việc đánh giá tập train (không augmentation, không shuffle)
    eval_datagen = ImageDataGenerator(rescale=1./255)
    train_eval_generator = eval_datagen.flow_from_directory(
        DATA_DIR / 'train', # Dùng cùng nguồn với train_generator
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False, # Quan trọng: không xáo trộn
        seed=SEED
    )
    
    # Tạo target names từ generator
    idx2class = {v: k for k, v in train_generator.class_indices.items()}
    target_names = [idx2class[i] for i in range(len(idx2class))]
    print(f"Target names: {target_names}")
    
    # Tạo mô hình
    model = create_optimized_model(len(target_names))
    print(f"Total parameters: {model.count_params():,}")
    model.summary()
    
    # Huấn luyện mô hình
    history = train_model(model, train_generator, val_generator)
    
    # Vẽ biểu đồ lịch sử huấn luyện
    plot_training_history(history)

    # Đánh giá và trực quan hóa trên cả 3 tập
    evaluate_and_report(model, train_eval_generator, "train", target_names)
    evaluate_and_report(model, val_generator, "validation", target_names)
    evaluate_and_report(model, test_generator, "test", target_names)
    
    print("\nĐã hoàn thành huấn luyện và đánh giá mô hình CNN!")
    print("Sử dụng dữ liệu processed cho tất cả các tập (có tăng cường on-the-fly cho training).")
    print(f"Run timestamp: {timestamp}")
    print(f"Kết quả được lưu trong thư mục: {RESULTS_DIR}")
    print(f"Logs được lưu trong thư mục: {LOGS_DIR}")
    print(f"Models được lưu trong thư mục: {MODELS_DIR}")

if __name__ == "__main__":
    main()
