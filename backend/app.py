from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import io
import os

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# --- Cấu hình ---
# Tên file model của bạn
MODEL_PATH = '../final_model.keras' # Hoặc '../final_model.keras'
# Kích thước ảnh đầu vào mà model của bạn mong đợi (ví dụ: MNIST là 28x28)
IMG_WIDTH = 28
IMG_HEIGHT = 28

# --- Load Model ---
# Load model một lần khi ứng dụng khởi động để tránh load lại mỗi request
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Đặt model thành None nếu load thất bại

# --- Hàm tiền xử lý ảnh ---
def preprocess_image(image_bytes):
    """
    Tiền xử lý ảnh: đọc bytes, chuyển grayscale, resize, chuẩn hóa, reshape.
    """
    try:
        # Mở ảnh từ bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Chuyển sang ảnh xám (nếu model của bạn mong đợi ảnh xám)
        # MNIST là ảnh xám, nên chúng ta chuyển sang 'L' (Luminance)
        if image.mode != 'L':
            image = image.convert('L')

        # Resize ảnh về kích thước mà model mong đợi
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))

        # Chuyển ảnh sang mảng numpy
        img_array = np.array(image)

        # Chuẩn hóa dữ liệu (ví dụ: chia cho 255 nếu model được train với giá trị pixel từ 0-1)
        # MNIST thường được chuẩn hóa về 0-1
        img_array = img_array / 255.0

        # Reshape mảng để phù hợp với input layer của model
        # Model Keras thường mong đợi shape (batch_size, height, width, channels)
        # Với ảnh xám 28x28, batch size 1, shape sẽ là (1, 28, 28, 1)
        img_array = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)

        return img_array

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# --- Routes (Các đường dẫn URL) ---

# Route cho trang chủ - hiển thị file index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route để nhận ảnh và dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Kiểm tra xem request có chứa file ảnh không
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']

    # Nếu người dùng không chọn file, trình duyệt gửi file trống
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Kiểm tra định dạng file (tùy chọn)
    # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    # if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
    #     return jsonify({'error': 'Invalid file type'}), 400

    if file:
        try:
            # Đọc nội dung file ảnh dưới dạng bytes
            img_bytes = file.read()

            # Tiền xử lý ảnh
            processed_image = preprocess_image(img_bytes)

            if processed_image is None:
                 return jsonify({'error': 'Failed to preprocess image'}), 500

            # Thực hiện dự đoán
            predictions = model.predict(processed_image)

            # Lấy kết quả dự đoán (chỉ số của lớp có xác suất cao nhất)
            predicted_class = np.argmax(predictions, axis=1)[0]

            # Trả về kết quả dưới dạng JSON
            return jsonify({'prediction': int(predicted_class)}) # Chuyển numpy int sang int Python

        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

    # Trường hợp không xử lý được file
    return jsonify({'error': 'An unknown error occurred'}), 500

# --- Chạy ứng dụng ---
if __name__ == '__main__':
    # Để chạy ở chế độ debug (tự động reload khi thay đổi code)
    # app.run(debug=True)
    # Để chạy production, bỏ debug=True
    app.run(debug=True) # Chạy trên http://127.0.0.1:5000/