import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # Import hàm load_model
from tensorflow.keras.datasets import mnist

# 1. Load dữ liệu MNIST (chỉ cần tập test để dự đoán)
(_, _), (X_test, y_test) = mnist.load_data()

# 2. Reshape và chuẩn hóa dữ liệu (chỉ cho tập test)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# 3. Load mô hình đã huấn luyện từ file (ĐOẠN CODE LẤY MODEL)
model = load_model('final_model.keras')
print("Đã load mô hình từ 'final_model.keras'")

# 4. Dự đoán ảnh
plt.imshow(X_test[8].reshape(28, 28), cmap='gray')
print(X_test[8])
y_predict = model.predict(X_test[8].reshape(1, 28, 28, 1))
print('Giá trị dự đoán:', np.argmax(y_predict))
plt.show()