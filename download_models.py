import os
import subprocess

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subprocess.run([
        "aria2c", url,
        "--console-log-level=warn",
        "-c", "-s", "16", "-x", "16", "-k", "10M",
        "-d", os.path.dirname(output_path),
        "-o", os.path.basename(output_path)
    ])
    print(f"Downloaded {output_path}")

if __name__ == "__main__":
    MODEL_URL = "https://huggingface.co/Laxhar/noobai-XL-1.1/resolve/main/NoobAI-XL-v1.1.safetensors"
    # URL LoRA (bạn cần thay thế bằng URL tải LoRA của bạn hoặc copy thẳng vào máy ảo)
    # LORA_URL = "..." 
    
    download_file(MODEL_URL, "/workspace/models/noobai_xl_v1.1.safetensors")
    
    print("Mô hình đã được tải xong. Hãy đảm bảo bạn đã copy file LoRA của mình vào /workspace/models/lora_lung_mat.safetensors")
