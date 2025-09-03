import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import datetime

# Đường dẫn
DATA_DIR = Path("data/augmented")  # Sử dụng dữ liệu đã tăng cường
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
CLASSES = ["Brown_Spot", "Leaf_Blast", "Leaf_Blight", "Healthy"]

# Tạo thư mục lưu mô hình và kết quả
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

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

def create_mobilenet_model():
    """Tạo mô hình sử dụng MobileNetV2"""
    # Load mô hình pretrained
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Đóng băng các lớp của mô hình cơ sở
    for layer in base_model.layers:
        layer.trainable = False
    
    # Thêm các lớp mới
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Tạo mô hình mới
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Biên dịch mô hình
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, 'mobilenet'

def create_resnet_model():
    """Tạo mô hình sử dụng ResNet50V2"""
    # Load mô hình pretrained
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Đóng băng các lớp của mô hình cơ sở
    for layer in base_model.layers:
        layer.trainable = False
    
    # Thêm các lớp mới
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Tạo mô hình mới
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Biên dịch mô hình
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, 'resnet'

def create_efficientnet_model():
    """Tạo mô hình sử dụng EfficientNetB0"""
    # Load mô hình pretrained
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Đóng băng các lớp của mô hình cơ sở
    for layer in base_model.layers:
        layer.trainable = False
    
    # Thêm các lớp mới
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Tạo mô hình mới
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Biên dịch mô hình
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, 'efficientnet'

def fine_tune_model(model, model_name, train_generator, val_generator):
    """Fine-tune mô hình"""
    # Huấn luyện các lớp mới trước
    print(f"Huấn luyện các lớp mới của {model_name}...")
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=5,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE
    )
    
    # Mở khóa một số lớp cuối của mô hình cơ sở để fine-tune
    if model_name == 'mobilenet':
        for layer in model.layers[:100]:
            layer.trainable = False
        for layer in model.layers[100:]:
            layer.trainable = True
    elif model_name == 'resnet':
        for layer in model.layers[:150]:
            layer.trainable = False
        for layer in model.layers[150:]:
            layer.trainable = True
    elif model_name == 'efficientnet':
        for layer in model.layers[:200]:
            layer.trainable = False
        for layer in model.layers[200:]:
            layer.trainable = True
    
    # Biên dịch lại mô hình với learning rate thấp hơn
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, model_name, train_generator, val_generator):
    """Huấn luyện mô hình"""
    # Thiết lập callbacks
    checkpoint = ModelCheckpoint(
        MODELS_DIR / f'{model_name}_best.h5',
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
        min_lr=0.000001,
        verbose=1
    )
    
    # Tạo TensorBoard callback
    log_dir = f"logs/fit/{model_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
    )
    
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
    model.save(MODELS_DIR / f'{model_name}_final.h5')
    
    return history

def evaluate_model(model, model_name, test_generator):
    """Đánh giá mô hình trên tập test"""
    # Đánh giá mô hình
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Dự đoán trên tập test
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Lưu kết quả dự đoán
    with open(RESULTS_DIR / f'{model_name}_results.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    
    return y_true, y_pred

def plot_training_history(history, model_name):
    """Vẽ biểu đồ lịch sử huấn luyện"""
    plt.figure(figsize=(12, 5))
    
    # Vẽ biểu đồ accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name.capitalize()} - Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Vẽ biểu đồ loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name.capitalize()} - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{model_name}_training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Vẽ ma trận nhầm lẫn"""
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    # Tính ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    
    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'{model_name.capitalize()} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{model_name}_confusion_matrix.png')
    plt.close()
    
    # Tạo báo cáo phân loại
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print(f"Classification Report for {model_name}:")
    print(report)
    
    # Lưu báo cáo phân loại
    with open(RESULTS_DIR / f'{model_name}_classification_report.txt', 'w') as f:
        f.write(report)

def main():
    print("===== HUẤN LUYỆN MÔ HÌNH TRANSFER LEARNING =====")
    
    # Tạo data generators
    train_generator, val_generator, test_generator = create_data_generators()
    print(f"Số lượng ảnh train: {train_generator.samples}")
    print(f"Số lượng ảnh validation: {val_generator.samples}")
    print(f"Số lượng ảnh test: {test_generator.samples}")
    
    # Danh sách các mô hình cần huấn luyện
    models_to_train = [
        create_mobilenet_model,
        create_resnet_model,
        create_efficientnet_model
    ]
    
    # Huấn luyện từng mô hình
    for create_model_func in models_to_train:
        # Tạo mô hình
        model, model_name = create_model_func()
        print(f"\n===== Huấn luyện mô hình {model_name.upper()} =====")
        
        # Fine-tune mô hình
        model = fine_tune_model(model, model_name, train_generator, val_generator)
        
        # Huấn luyện mô hình
        history = train_model(model, model_name, train_generator, val_generator)
        
        # Đánh giá mô hình
        y_true, y_pred = evaluate_model(model, model_name, test_generator)
        
        # Vẽ biểu đồ
        plot_training_history(history, model_name)
        plot_confusion_matrix(y_true, y_pred, model_name)
        
        print(f"\nĐã hoàn thành huấn luyện và đánh giá mô hình {model_name.upper()}!")
    
    print("\nĐã hoàn thành huấn luyện tất cả các mô hình transfer learning!")
    print(f"Kết quả được lưu trong thư mục: {RESULTS_DIR}")

if __name__ == "__main__":
    main()


