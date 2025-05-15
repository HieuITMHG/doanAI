import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# 1. Load dữ liệu MNIST (load cả tập huấn luyện và test)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Reshape và chuẩn hóa dữ liệu (cho tập test)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
y_test = to_categorical(y_test, 10)  # One-hot encoding cho nhãn

print("Kích thước tập test sau reshape:", X_test.shape)

# 3. Load mô hình đã huấn luyện từ file
model = load_model('final_model.keras')
print("Đã load mô hình từ 'final_model.keras'")

# 4. Đánh giá mô hình trên tập test
test_score = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", test_score[0])
print("Test Accuracy:", test_score[1])

# 5. Dự đoán trên một ảnh ngẫu nhiên từ tập test
# Chọn một index ngẫu nhiên trong phạm vi số lượng mẫu của tập test
num_samples = X_test.shape[0]  # Số lượng mẫu trong tập test (10,000)
index = np.random.randint(0, num_samples)  # Chọn ngẫu nhiên một chỉ số từ 0 đến 9999

# Hiển thị ảnh và dự đoán
plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
print("Dữ liệu ảnh (một phần):", X_test[index].flatten()[:10])  # In một phần dữ liệu ảnh để kiểm tra
y_predict = model.predict(X_test[index].reshape(1, 28, 28, 1))
print('Chỉ số được chọn ngẫu nhiên:', index)
print('Giá trị dự đoán:', np.argmax(y_predict))
print('Giá trị thực tế:', np.argmax(y_test[index]))
plt.show()