import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import datetime

# Đường dẫn
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs")
CLASSES = ["Brown_Spot", "Leaf_Blast", "Leaf_Blight", "Healthy"]

# Tạo thư mục lưu mô hình và kết quả
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Các tham số
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = len(CLASSES)

def create_data_generators():
    """Tạo data generators cho việc huấn luyện (dùng preprocess của ResNet)."""
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

def create_resnet_model(trainable_backbone: bool = False):
    """Tạo mô hình ResNet50 với head phân loại tuỳ chỉnh.

    trainable_backbone: nếu True sẽ fine-tune backbone, mặc định freeze.
    """
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = trainable_backbone

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def ensure_directory(path: Path):
    if path.exists() and path.is_file():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)

def train_model(model, train_generator, val_generator):
    """Huấn luyện mô hình ResNet."""
    checkpoint = ModelCheckpoint(
        MODELS_DIR / 'resnet_processed_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    log_dir = LOGS_DIR / 'fit' / 'resnet_processed' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    ensure_directory(LOGS_DIR)
    ensure_directory(LOGS_DIR / 'fit')
    ensure_directory(log_dir.parent)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    )

    model.save(MODELS_DIR / 'resnet_processed_final.h5')
    return history

def evaluate_model(model, test_generator):
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    with open(RESULTS_DIR / 'resnet_processed_results.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")

    return y_true, y_pred

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy (ResNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss (ResNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'resnet_processed_training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix (ResNet, Processed Data)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'resnet_processed_confusion_matrix.png')
    plt.close()

    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print("Classification Report:")
    print(report)
    with open(RESULTS_DIR / 'resnet_processed_classification_report.txt', 'w') as f:
        f.write(report)

def main():
    print("===== HUẤN LUYỆN MÔ HÌNH RESNET VỚI DỮ LIỆU ĐÃ CHUẨN HÓA =====")

    train_generator, val_generator, test_generator = create_data_generators()
    print(f"Số lượng ảnh train: {train_generator.samples}")
    print(f"Số lượng ảnh validation: {val_generator.samples}")
    print(f"Số lượng ảnh test: {test_generator.samples}")

    model = create_resnet_model(trainable_backbone=False)
    model.summary()

    history = train_model(model, train_generator, val_generator)

    y_true, y_pred = evaluate_model(model, test_generator)

    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred)

    print("\nĐã hoàn thành huấn luyện và đánh giá mô hình ResNet với dữ liệu đã chuẩn hóa!")
    print(f"Kết quả được lưu trong thư mục: {RESULTS_DIR}")

if __name__ == "__main__":
    main()


