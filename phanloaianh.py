import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Chia tập train thành train và validation
X_val, y_val = X_train[50000:60000, :], y_train[50000:60000]
X_train, y_train = X_train[:50000, :], y_train[:50000]

print("Kích thước tập train:", X_train.shape)
print("Kích thước tập validation:", X_val.shape)
print("Kích thước tập test:", X_test.shape)

# 2. Reshape và chuẩn hóa dữ liệu
# Thêm chiều kênh (1 cho grayscale) và chuẩn hóa về [0, 1]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

print("Kích thước sau reshape (train):", X_train.shape)

# 3. One-hot encoding cho nhãn
Y_train = to_categorical(y_train, 10)
Y_val = to_categorical(y_val, 10)
Y_test = to_categorical(y_test, 10)

print('Dữ liệu y ban đầu:', y_train[0])
print('Dữ liệu y sau one-hot encoding:', Y_train[0])

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),  # Thêm padding='same'
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


# 5. Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Thiết lập callback để lưu mô hình tốt nhất và dừng sớm nếu không cải thiện
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# 7. Huấn luyện mô hình
H = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=32,
    epochs=10,
    verbose=1,
    callbacks=[checkpoint, early_stopping]
)

# 8. Lưu lịch sử huấn luyện vào file CSV
history_df = pd.DataFrame({
    'epoch': np.arange(1, len(H.history['loss']) + 1),
    'train_loss': H.history['loss'],
    'val_loss': H.history['val_loss'],
    'train_accuracy': H.history['accuracy'],
    'val_accuracy': H.history['val_accuracy']
})
history_df.to_csv('training_history.csv', index=False)
print("Đã lưu lịch sử huấn luyện vào 'training_history.csv'")

# 9. Vẽ đồ thị loss và accuracy
plt.figure(figsize=(10, 5))

# Đồ thị Loss
plt.subplot(1, 2, 1)
plt.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss')
plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Đồ thị Accuracy
plt.subplot(1, 2, 2)
plt.plot(history_df['epoch'], history_df['train_accuracy'], label='Training Accuracy')
plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_plots.png')  # Lưu đồ thị vào file
plt.show()
print("Đã lưu đồ thị vào 'training_plots.png'")

# 10. Đánh giá mô hình trên tập test
score = model.evaluate(X_test, Y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

# 11. Lưu mô hình đã huấn luyện
model.save('final_model.keras')
print("Đã lưu mô hình vào 'final_model.keras'")

# 12. Phần dự đoán ảnh (được comment lại theo yêu cầu)

plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
y_predict = model.predict(X_test[1].reshape(1, 28, 28, 1))
print('Giá trị dự đoán:', np.argmax(y_predict))
