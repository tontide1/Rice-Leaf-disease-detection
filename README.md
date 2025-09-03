# Dự Án Phát Hiện Bệnh Trên Lá Lúa Sử Dụng CNN

## Giới thiệu
Dự án này sử dụng các mô hình học sâu (Deep Learning) để phát hiện và phân loại các bệnh trên lá lúa từ hình ảnh. Hệ thống có thể phân loại lá lúa thành 4 loại: Đốm nâu (Brown Spot), Đạo ôn (Leaf Blast), Bạc lá (Leaf Blight) và Lá khỏe mạnh (Healthy).

## Cấu trúc dự án
```
Rice-Leaf-disease-detection/
├── data/
│   ├── Rice_Leaf_Disease_Images/ # Dữ liệu gốc
│   ├── splits/                   # Dữ liệu đã phân chia
│   ├── processed/                # Dữ liệu đã tiền xử lý
│   └── augmented/                # Dữ liệu đã tăng cường
├── models/                       # Lưu trữ các mô hình đã huấn luyện
├── results/                      # Kết quả đánh giá và hình ảnh
├── split_dataset.py              # Script phân chia dữ liệu
├── explore_data.py               # Script khám phá và trực quan hóa dữ liệu
├── preprocess_data.py            # Script tiền xử lý và tăng cường dữ liệu
├── model_baseline.py             # Mô hình CNN cơ bản
├── model_transfer.py             # Mô hình sử dụng transfer learning
├── model_custom.py               # Mô hình kiến trúc tùy chỉnh
├── evaluate_models.py            # Script đánh giá và so sánh các mô hình
└── README.md                     # Tài liệu dự án
```

## Các bước thực hiện
1. **Phân chia dữ liệu**: Chia dữ liệu thành tập huấn luyện (70%), xác thực (15%) và kiểm tra (15%)
2. **Khám phá dữ liệu**: Phân tích và trực quan hóa dữ liệu
3. **Tiền xử lý và tăng cường dữ liệu**: Chuẩn hóa kích thước, tăng cường độ tương phản, tăng cường dữ liệu
4. **Xây dựng mô hình**: Từ mô hình CNN cơ bản đến các mô hình pretrained và kiến trúc tùy chỉnh
5. **Đánh giá mô hình**: So sánh hiệu suất các mô hình dựa trên nhiều tiêu chí


## Các lớp bệnh
1. **Brown Spot (Đốm nâu)**: Gây ra bởi nấm Cochliobolus miyabeanus
2. **Leaf Blast (Đạo ôn)**: Gây ra bởi nấm Magnaporthe oryzae
3. **Leaf Blight (Bạc lá)**: Gây ra bởi vi khuẩn Xanthomonas oryzae
4. **Healthy (Khỏe mạnh)**: Lá lúa không bị bệnh

## Hướng dẫn sử dụng
1. **Phân chia dữ liệu**:
   ```
   python split_dataset.py
   ```

2. **Khám phá dữ liệu**:
   ```
   python explore_data.py
   ```

3. **Tiền xử lý và tăng cường dữ liệu**:
   ```
   python preprocess_data.py
   ```

4. **Huấn luyện mô hình cơ bản**:
   ```
   python model_baseline.py
   ```

5. **Huấn luyện mô hình transfer learning**:
   ```
   python model_transfer.py
   ```

6. **Đánh giá và so sánh các mô hình**:
   ```
   python evaluate_models.py
   ```
