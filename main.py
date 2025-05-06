try:
    from backend.app import app
    print("Successfully imported Flask app from backend/app.py")
except ImportError as e:
    print(f"Error importing Flask app: {e}")
    print("Please ensure you have a file 'backend/app.py' and it defines a Flask app instance named 'app'.")
    print("Also, make sure you are running this file from the project root directory (DOANAI/).")
    exit(1) # Thoát nếu không import được

# Đoạn mã này chỉ chạy khi bạn thực thi file main.py trực tiếp
if __name__ == '__main__':
    print("Starting the web application...")
    # Chạy ứng dụng Flask
    # app.run() sẽ sử dụng cấu hình mặc định hoặc cấu hình đã đặt trong backend/app.py
    # Nếu bạn muốn ghi đè cấu hình (ví dụ: port), bạn có thể truyền tham số vào đây
    # Ví dụ: app.run(debug=True, port=8000)
    app.run(debug=True) # Sử dụng debug=True cho quá trình phát triển

    print("Web application stopped.")