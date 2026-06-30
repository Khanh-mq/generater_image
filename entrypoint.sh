#!/bin/bash
set -e

mkdir -p /app/models

# Load các biến môi trường từ file .env (nếu chạy trực tiếp không qua Docker)
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Thiết lập URL mặc định nếu chưa có
if [ -z "$MODEL_URL" ]; then
    MODEL_URL="https://huggingface.co/Laxhar/noobai-XL-1.1/resolve/main/NoobAI-XL-v1.1.safetensors"
fi
if [ -z "$LORA_URL" ]; then
    LORA_URL="https://huggingface.co/khanhmq/lung_mat_lora/resolve/main/lung_mat-04.safetensors"
fi

MODEL_PATH="/app/models/noobai_xl_v1.1.safetensors"
LORA_PATH="/app/models/lora_lung_mat.safetensors"

# 1. Kéo Base Model nếu chưa có
if [ ! -f "$MODEL_PATH" ]; then
    echo "Base model không tồn tại. Đang tiến hành kéo từ $MODEL_URL..."
    aria2c "$MODEL_URL" --console-log-level=warn -c -s 16 -x 16 -k 10M -d /app/models -o noobai_xl_v1.1.safetensors
    echo "✅ Kéo Base model thành công!"
else
    echo "✅ Base model đã tồn tại, bỏ qua bước tải."
fi

# 2. Kéo LoRA nếu chưa có
if [ ! -f "$LORA_PATH" ]; then
    echo "LoRA không tồn tại. Đang tiến hành kéo từ $LORA_URL..."
    if [[ "$LORA_URL" == *"drive.google.com"* ]]; then
        gdown "$LORA_URL" -O "$LORA_PATH"
    else
        aria2c "$LORA_URL" --console-log-level=warn -c -s 16 -x 16 -k 10M -d /app/models -o lora_lung_mat.safetensors
    fi
    echo "✅ Kéo LoRA thành công!"
else
    echo "✅ LoRA đã tồn tại, bỏ qua bước tải."
fi

# Kiểm tra chốt chặn cuối cùng: Nếu thiếu LoRA thì đập máy nghỉ chơi!
if [ ! -f "$LORA_PATH" ]; then
    echo "❌ LỖI NGHIÊM TRỌNG: Không tìm thấy file LoRA tại $LORA_PATH sau khi tải!"
    echo "❌ Server không thể khởi động vì bạn yêu cầu bắt buộc phải có LoRA."
    exit 1
fi

# 3. Khởi chạy Server
echo "🚀 Đang khởi động FastAPI Server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
