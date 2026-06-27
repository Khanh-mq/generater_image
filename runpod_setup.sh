#!/bin/bash
set -e

echo "==========================================="
echo "🚀 BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG CHO RUNPOD..."
echo "==========================================="

# 1. Cập nhật và cài đặt phần mềm hỗ trợ tải tốc độ cao (aria2)
echo "📥 [1/3] Đang cài đặt aria2..."
apt update -q && apt install aria2 -y -q

# 2. Cài đặt các thư viện Python cần thiết
echo "🐍 [2/3] Đang cài đặt thư viện Python (PyTorch, Diffusers, FastAPI...)..."
pip install --no-cache-dir -r requirements.txt gdown

# 3. Kích hoạt quyền thực thi và chạy kịch bản chính
echo "🔥 [3/3] Đang cấp quyền và khởi chạy hệ thống..."
chmod +x entrypoint.sh
./entrypoint.sh
