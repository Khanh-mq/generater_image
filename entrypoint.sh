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

# 2. Kéo LoRA nếu chưa có (Yêu cầu phải có biến môi trường LORA_URL)
if [ ! -f "$LORA_PATH" ]; then
    if [ "$LORA_URL" != "https://huggingface.co/duongdan_cua_ban/lora_lung_mat.safetensors" ] && [ -n "$LORA_URL" ]; then
        echo "LoRA không tồn tại. Đang tiến hành kéo từ $LORA_URL..."
        # Kiểm tra nếu là link Google Drive thì dùng gdown, nếu link trực tiếp thì dùng aria2c
        if [[ "$LORA_URL" == *"drive.google.com"* ]]; then
            gdown "$LORA_URL" -O "$LORA_PATH"
        else
            aria2c "$LORA_URL" --console-log-level=warn -c -s 16 -x 16 -k 10M -d /app/models -o lora_lung_mat.safetensors
        fi
        echo "✅ Kéo LoRA thành công!"
    else
        echo "⚠️ CẢNH BÁO: Chưa cấu hình LORA_URL hợp lệ trong docker-compose.yml, hoặc file lora chưa được copy vào thư mục ./models/"
        echo "⚠️ Model sẽ chạy MÀ KHÔNG CÓ LORA nếu bạn không cung cấp."
    fi
else
    echo "✅ LoRA đã tồn tại, bỏ qua bước tải."
fi

# 3. Khởi chạy Server
echo "🚀 Đang khởi động FastAPI Server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
