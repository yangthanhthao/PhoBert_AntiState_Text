import os
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
data = pd.read_csv('dataset2.csv')

# Khởi tạo tokenizer và model PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
phobert_model = AutoModel.from_pretrained("vinai/phobert-large")

# Hàm mã hóa văn bản bằng PhoBERT
def encode_text(texts):
    # Mã hóa văn bản
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=256)
    
    # Trích xuất đặc trưng từ PhoBERT
    with torch.no_grad():
        model_output = phobert_model(**encoded_input)
    
    # Sử dụng vector CLS làm đặc trưng cho văn bản
    text_features = model_output.last_hidden_state[:, 0, :].cpu().numpy()
    text_features = tf.convert_to_tensor(text_features, dtype=tf.float32)
    
    # Mô hình xử lý đặc trưng sau PhoBERT
    x = Sequential()
    x.add(Dense(64, activation='relu', input_shape=(text_features.shape[1],)))
    x.add(Dropout(0.4))
    x.add(Dense(32, activation='relu'))
    x.add(Dropout(0.4))
    
    # Biến đổi đặc trưng
    text_features_transformed = x(text_features, training=False).numpy()
    return text_features_transformed

# Gọi hàm encode_text để trích xuất đặc trưng từ văn bản
text_features = encode_text(data['title'].values.tolist())

# Chuẩn bị dữ liệu
labels = data['isAntiState'].values
X_train, X_test, y_train, y_test = train_test_split(text_features, labels, test_size=0.2, random_state=42)

# Khởi tạo mô hình
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(text_features.shape[1],)))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

# Dự đoán trên tập kiểm tra
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# Tính toán độ đo cuối cùng
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# In ra kết quả
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
