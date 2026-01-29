# ==============================================================================
# SCRIPT: MODEL CONFIGURATION REPAIR
# TÁC VỤ: Tải và bổ sung file cấu hình preprocessor còn thiếu
# ==============================================================================

from transformers import ViTImageProcessor
import os

# --- CẤU HÌNH ĐƯỜNG DẪN (Chính xác theo ổ D của bạn) ---
MODEL_PATH = "D:/AI_Projects/model_birds_cub200_final"

def repair_model_config():
    print(f"--> [INFO] Đang kiểm tra thư mục: {MODEL_PATH}")
    
    # Kiểm tra xem folder có tồn tại không
    if not os.path.exists(MODEL_PATH):
        print(f"❌ [LỖI] Không tìm thấy thư mục model tại: {MODEL_PATH}")
        print("    Gợi ý: Hãy kiểm tra lại xem bạn đã tạo folder này chưa?")
        return

    try:
        print("--> [PROCESS] Đang tải cấu hình chuẩn từ Google Cloud...")
        # Tải cấu hình xử lý ảnh gốc của Google ViT
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        
        print(f"--> [SAVE] Đang ghi file cấu hình vào: {MODEL_PATH}")
        # Lưu file 'preprocessor_config.json' vào ổ cứng
        processor.save_pretrained(MODEL_PATH)
        
        print("\n" + "="*60)
        print("✅ THÀNH CÔNG! MODEL ĐÃ ĐỦ TIÊU CHUẨN.")
        print("="*60)
        print("Thư mục hiện tại đã chứa đủ:")
        print("   1. pytorch_model.bin (File nặng - Weights)")
        print("   2. config.json (Cấu trúc mạng)")
        print("   3. preprocessor_config.json (Vừa tạo - Xử lý ảnh)")
        
    except Exception as e:
        print(f"\n❌ [CRITICAL ERROR]: {e}")

if __name__ == "__main__":
    repair_model_config()