# Sử dụng base image có sẵn PyTorch và CUDA
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết (nếu có)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl aria2 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gdown

# Copy source code
COPY main.py .
COPY model_loader.py .
COPY entrypoint.sh .

# Cấp quyền thực thi cho script khởi động
RUN chmod +x entrypoint.sh

# Khai báo port API
EXPOSE 8000

# Chạy entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
