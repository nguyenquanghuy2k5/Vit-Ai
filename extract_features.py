# ==============================================================================
# MODULE: COMPREHENSIVE MODEL EVALUATION PIPELINE
# VERSION: 1.0.2 (DATA COLLATOR FIX)
# DESCRIPTION: Fix lỗi TypeError do tồn tại đối tượng PIL Image trong Batch.
# ==============================================================================

import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
from sklearn.metrics import (
    accuracy_score, 
    top_k_accuracy_score, 
    classification_report, 
    precision_recall_fscore_support,
    average_precision_score
)
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# ==============================================================================
# 1. TASK CONFIGURATION
# ==============================================================================
TASK_MODE = "BIRDS"

if TASK_MODE == "BIRDS":
    print(f"--> [SYSTEM] MODE: BIRD SPECIES RECOGNITION (CUB-200)")
    MODEL_PATH  = "./model_birds_cub200"
    DATASET_DIR = "./dataset_birds"
    
elif TASK_MODE == "DOGS":
    print(f"--> [SYSTEM] MODE: DOG BREEDS RECOGNITION (Stanford Dogs)")
    MODEL_PATH  = "./model_dogs_stanford"
    DATASET_DIR = "./dataset_dogs"

elif TASK_MODE == "FOOD":
    print(f"--> [SYSTEM] MODE: VIETNAMESE FOOD RECOGNITION")
    MODEL_PATH  = "./model_vietnamese_food"
    DATASET_DIR = "./dataset_food"

else:
    print(f"[FATAL ERROR] Invalid TASK_MODE: '{TASK_MODE}'. Exiting.")
    sys.exit(1)

# --- CONFIG FIX FOR WINDOWS ---
BATCH_SIZE   = 32
NUM_WORKERS  = 0  # Giữ nguyên 0 cho Windows
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("="*60)
    print(f"INITIALIZING EVALUATION PROTOCOL | DEVICE: {DEVICE.upper()}")
    print("="*60)

    # --------------------------------------------------------------------------
    # 2. MODEL LOADING
    # --------------------------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"[CRITICAL ERROR] Model artifacts not found at: {MODEL_PATH}")
        return

    try:
        print(f"[1/5] Loading model weights and configuration...")
        model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
        processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return

    # --------------------------------------------------------------------------
    # 3. DATA PIPELINE SETUP
    # --------------------------------------------------------------------------
    print(f"[2/5] Configuring data loader for dataset: {DATASET_DIR}...")
    
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]

    val_transforms = Compose([
        Resize(256),
        CenterCrop(size),
        ToTensor(),
        Normalize(mean=image_mean, std=image_std)
    ])

    # --- FIX QUAN TRỌNG TẠI ĐÂY ---
    def preprocess_batch(examples):
        # 1. Chuyển đổi ảnh sang Tensor và gán vào cột 'pixel_values'
        examples['pixel_values'] = [val_transforms(img.convert("RGB")) for img in examples['image']]
        
        # 2. [FIX] Xóa cột 'image' gốc (dạng PIL) để tránh lỗi DataLoader
        # DataLoader không thể đóng gói (collate) các đối tượng PIL Image, chỉ nhận Tensors
        if 'image' in examples:
            del examples['image']
            
        return examples

    try:
        ds = load_dataset("imagefolder", data_dir=DATASET_DIR, split="test")
        # Áp dụng hàm preprocess đã sửa
        ds.set_transform(preprocess_batch)
        
        test_loader = DataLoader(
            ds, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=True if DEVICE == "cuda" else False
        )
        print(f"      -> Test samples: {len(ds)}")
        print(f"      -> Number of Classes: {len(ds.features['label'].names)}")
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        return

    # --------------------------------------------------------------------------
    # 4. INFERENCE EXECUTION
    # --------------------------------------------------------------------------
    print(f"[3/5] Executing inference loop...")
    
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference Progress"):
            # Lúc này batch chỉ còn chứa 'pixel_values' và 'label' (đều là Tensor)
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["label"]

            outputs = model(pixel_values)
            logits = outputs.logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    class_names = ds.features['label'].names

    # --------------------------------------------------------------------------
    # 5. METRICS CALCULATION
    # --------------------------------------------------------------------------
    print(f"[4/5] Computing statistical metrics...")

    acc_top1 = accuracy_score(y_true, y_pred)
    
    k_val = 5 if len(class_names) >= 5 else len(class_names)
    acc_top5 = top_k_accuracy_score(y_true, y_probs, k=k_val)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    y_true_onehot = np.eye(len(class_names))[y_true]
    try:
        m_ap = average_precision_score(y_true_onehot, y_probs, average="macro")
    except Exception as e:
        print(f"[WARNING] Could not calculate mAP: {e}")
        m_ap = 0.0

    # --------------------------------------------------------------------------
    # 6. PERFORMANCE REPORT
    # --------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"PERFORMANCE REPORT: {TASK_MODE}")
    print("="*60)
    
    print(f"{'METRIC':<30} | {'VALUE':<10}")
    print("-" * 45)
    print(f"{'Accuracy (Top-1)':<30} | {acc_top1 * 100:.2f}%")
    print(f"{f'Top-{k_val} Accuracy':<30} | {acc_top5 * 100:.2f}%")
    print(f"{'Macro Precision':<30} | {precision * 100:.2f}%")
    print(f"{'Macro Recall':<30} | {recall * 100:.2f}%")
    print(f"{'Macro F1-Score':<30} | {f1 * 100:.2f}%")
    print(f"{'Mean Average Precision (mAP)':<30} | {m_ap * 100:.2f}%")
    print("-" * 45)

    # --------------------------------------------------------------------------
    # 7. ERROR ANALYSIS
    # --------------------------------------------------------------------------
    print("\n[5/5] ERROR ANALYSIS: Identifying worst performing classes...")
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    class_metrics = {k: v for k, v in report_dict.items() if isinstance(v, dict)}
    
    sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['f1-score'])
    
    print(f"\n>>> TOP 5 CLASSES REQUIRING DATA AUGMENTATION / CLEANING:")
    print(f"{'Class Name':<40} | {'Prec.':<8} | {'Recall':<8} | {'F1':<8}")
    print("-" * 80)
    for class_name, metrics in sorted_classes[:5]:
        p = metrics['precision'] * 100
        r = metrics['recall'] * 100
        f = metrics['f1-score'] * 100
        print(f"{class_name:<40} | {p:6.2f}%  | {r:6.2f}%  | {f:6.2f}%")

    print("\n" + "="*60)
    print("EVALUATION PROTOCOL COMPLETED.")
    print("="*60)

if __name__ == "__main__":
    main()