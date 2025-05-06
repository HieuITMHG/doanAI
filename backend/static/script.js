document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const predictionResult = document.getElementById('predictionResult');

    // Lắng nghe sự kiện khi người dùng chọn file
    imageUpload.addEventListener('change', async (event) => {
        const file = event.target.files[0]; // Lấy file đầu tiên được chọn

        if (file) {
            // Hiển thị ảnh xem trước
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block'; // Hiển thị ảnh preview
            }
            reader.readAsDataURL(file); // Đọc file dưới dạng Data URL để hiển thị

            // Reset kết quả cũ
            predictionResult.textContent = 'Đang dự đoán...';
            predictionResult.style.color = 'black'; // Đặt màu mặc định

            // Gửi ảnh đến backend để dự đoán
            const formData = new FormData();
            formData.append('image', file); // 'image' phải khớp với tên mà backend mong đợi (request.files['image'])

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    // Xử lý lỗi HTTP (ví dụ: 400, 500)
                    const errorData = await response.json(); // Thử đọc body lỗi nếu có
                    throw new Error(`HTTP error! status: ${response.status}, details: ${errorData.error || response.statusText}`);
                }

                const data = await response.json(); // Phân tích phản hồi JSON

                if (data.prediction !== undefined) {
                    predictionResult.textContent = data.prediction; // Hiển thị kết quả dự đoán
                    predictionResult.style.color = 'green'; // Đổi màu cho dễ nhìn
                } else if (data.error) {
                     predictionResult.textContent = `Lỗi: ${data.error}`;
                     predictionResult.style.color = 'red';
                } else {
                    predictionResult.textContent = 'Không nhận được kết quả dự đoán hợp lệ.';
                    predictionResult.style.color = 'orange';
                }


            } catch (error) {
                console.error('Lỗi khi gửi ảnh hoặc nhận kết quả:', error);
                predictionResult.textContent = `Lỗi: Không thể kết nối hoặc xử lý yêu cầu. (${error.message})`;
                predictionResult.style.color = 'red';
            }
        } else {
            // Nếu không có file nào được chọn (ví dụ: người dùng hủy)
            imagePreview.src = '#';
            imagePreview.style.display = 'none';
            predictionResult.textContent = 'Chưa có';
            predictionResult.style.color = 'black';
        }
    });
});