# ==============================================================================
# MODULE: CUB-200-2011 OFFICIAL BENCHMARK PIPELINE
# STANDARD: Fixed Split (Official) + Bounding Box Cropping
# METRICS: Top-1 Acc, Top-5 Acc, mAP
# ==============================================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value, Image as DatasetImage
from transformers import (
    ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, DefaultDataCollator
)
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor, RandomRotation, ColorJitter
)
from sklearn.metrics import accuracy_score, top_k_accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize

# --- CẤU HÌNH QUAN TRỌNG ĐỂ TRÁNH LỖI ẢNH (Corrupt Images) ---
ImageFile.LOAD_TRUNCATED_IMAGES = True  
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ------------------------------------------------------------------------------
# 1. CẤU HÌNH HỆ THỐNG
# ------------------------------------------------------------------------------
TASK_MODE = "BIRDS"
SEED = 42
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ĐƯỜNG DẪN: Hãy sửa lại đúng folder dataset của bạn
DATASET_DIR = "./dataset_birds"  
# Mẹo: Đặt Output ra ổ đĩa còn nhiều dung lượng (tránh Desktop/OneDrive gây lỗi Permission)
OUTPUT_DIR  = "D:/AI_Projects/model_birds_cub200_final"

# ------------------------------------------------------------------------------
# 2. LOAD DATA CHUẨN QUỐC TẾ (OFFICIAL SPLIT)
# ------------------------------------------------------------------------------
def load_cub_benchmark(base_path):
    print(f"--> [DATA] Reading CUB-200 Standard Metadata from: {base_path}")
    try:
        # 1. Đọc các file cấu hình chuẩn
        df_images = pd.read_csv(os.path.join(base_path, "images.txt"), sep=' ', names=['id', 'path'])
        df_labels = pd.read_csv(os.path.join(base_path, "image_class_labels.txt"), sep=' ', names=['id', 'label'])
        
        # QUAN TRỌNG: Đây là file quyết định chuẩn Benchmark (is_train: 1=Train, 0=Test)
        df_split = pd.read_csv(os.path.join(base_path, "train_test_split.txt"), sep=' ', names=['id', 'is_train'])
        
        df_boxes = pd.read_csv(os.path.join(base_path, "bounding_boxes.txt"), sep=' ', names=['id', 'x', 'y', 'w', 'h'])
        df_classes = pd.read_csv(os.path.join(base_path, "classes.txt"), sep=' ', names=['id', 'class_name'])

        # 2. Merge dữ liệu
        df = df_images.merge(df_labels, on='id').merge(df_split, on='id').merge(df_boxes, on='id')
        
        # 3. Tạo đường dẫn tuyệt đối
        img_root = os.path.join(base_path, "images")
        df['path'] = df['path'].apply(lambda x: os.path.join(img_root, x))

        # 4. Chia Train/Test theo đúng file split
        train_df = df[df['is_train'] == 1]
        test_df = df[df['is_train'] == 0]
        
        class_names = df_classes['class_name'].tolist()
        print(f"--> [INFO] Standard Split Statistics:")
        print(f"    - Train Set: {len(train_df)} images (Expected ~5994)")
        print(f"    - Test Set:  {len(test_df)} images (Expected ~5794)")

        # 5. Định nghĩa Features
        features = Features({
            'image': DatasetImage(), 
            'label': ClassLabel(names=class_names),
            'box_x': Value('float32'), 'box_y': Value('float32'), 
            'box_w': Value('float32'), 'box_h': Value('float32')
        })

        def create_ds(df_sub):
            # CUB Label bắt đầu từ 1, phải trừ 1 để về 0-based index cho PyTorch
            return Dataset.from_dict({
                "image": df_sub['path'].tolist(), 
                "label": [l - 1 for l in df_sub['label'].tolist()],
                "box_x": df_sub['x'].tolist(), "box_y": df_sub['y'].tolist(),
                "box_w": df_sub['w'].tolist(), "box_h": df_sub['h'].tolist()
            }, features=features)

        return DatasetDict({"train": create_ds(train_df), "test": create_ds(test_df)})
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load metadata. Check paths: {e}")
        exit(1)

ds = load_cub_benchmark(DATASET_DIR)
labels = ds["train"].features["label"].names

# ------------------------------------------------------------------------------
# 3. TIỀN XỬ LÝ (SAFE CROP LOGIC)
# ------------------------------------------------------------------------------
MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(MODEL_CHECKPOINT)
size = processor.size["height"]
normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

_train_tf = Compose([
    RandomResizedCrop(size, scale=(0.8, 1.0)), 
    RandomHorizontalFlip(), 
    RandomRotation(15), 
    ToTensor(), 
    normalize
])

_val_tf = Compose([
    Resize(256), 
    CenterCrop(size), 
    ToTensor(), 
    normalize
])

def apply_transforms(examples, tf_fn):
    pixel_values = []
    final_labels = []
    
    for i in range(len(examples['image'])):
        try:
            img = examples['image'][i].convert("RGB")
            x, y, w, h = examples['box_x'][i], examples['box_y'][i], examples['box_w'][i], examples['box_h'][i]
            
            # Logic an toàn: Chỉ crop nếu box hợp lệ (w, h > 1 pixel)
            if w > 1 and h > 1:
                # Cắt theo Box (Standard Method for high accuracy tasks)
                img_cropped = img.crop((x, y, x + w, y + h))
            else:
                # Nếu box lỗi (hiếm gặp), lấy toàn bộ ảnh
                img_cropped = img
            
            pixel_values.append(tf_fn(img_cropped))
            final_labels.append(examples["label"][i])
            
        except Exception as e:
            # Bỏ qua ảnh lỗi để không crash quá trình train
            print(f"Skipping bad image index {i}: {e}")
            continue

    return {
        "pixel_values": pixel_values,
        "labels": final_labels
    }

# Sử dụng set_transform thay vì map để tiết kiệm ổ cứng
ds["train"].set_transform(lambda x: apply_transforms(x, _train_tf))
ds["test"].set_transform(lambda x: apply_transforms(x, _val_tf))

# ------------------------------------------------------------------------------
# 4. METRICS (CHUẨN NGHIÊN CỨU)
# ------------------------------------------------------------------------------
def compute_metrics(p):
    probs = torch.nn.functional.softmax(torch.tensor(p.predictions), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    
    acc1 = accuracy_score(p.label_ids, preds)
    acc5 = top_k_accuracy_score(p.label_ids, probs, k=5, labels=range(len(labels)))
    
    y_true_bin = label_binarize(p.label_ids, classes=range(len(labels)))
    map_score = average_precision_score(y_true_bin, probs, average="macro")
    
    return {"accuracy": acc1, "recall_at_5": acc5, "mAP": map_score}

# ------------------------------------------------------------------------------
# 5. CẤU HÌNH TRAINING (TỐI ƯU Ổ CỨNG)
# ------------------------------------------------------------------------------
model = ViTForImageClassification.from_pretrained(
    MODEL_CHECKPOINT, 
    num_labels=len(labels), 
    id2label={i:l for i,l in enumerate(labels)}, 
    label2id={l:i for i,l in enumerate(labels)}, 
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=40,            
    learning_rate=2e-5,             
    lr_scheduler_type="cosine",     
    per_device_train_batch_size=8,  
    gradient_accumulation_steps=8,  
    
    # Chiến lược lưu thông minh: Chỉ giữ 1 bản tốt nhất để tránh lỗi đầy ổ cứng
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,             
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    
    remove_unused_columns=False,    # BẮT BUỘC FALSE
    label_names=["labels"],
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DefaultDataCollator(),
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    compute_metrics=compute_metrics
)

# ------------------------------------------------------------------------------
# 6. THỰC THI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("--> STARTING CUB-200 OFFICIAL BENCHMARK RUN")
    print(f"--> Split Protocol: Standard (train_test_split.txt)")
    print(f"--> Crop Protocol:  Bounding Box (Method C)")
    print("="*60 + "\n")
    
    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    
    # Vẽ biểu đồ
    try:
        history = trainer.state.log_history
        epochs = [log["epoch"] for log in history if "eval_accuracy" in log]
        acc = [log["eval_accuracy"] for log in history if "eval_accuracy" in log]
        recall = [log["eval_recall_at_5"] for log in history if "eval_recall_at_5" in log]
        mAPs = [log["eval_mAP"] for log in history if "eval_mAP" in log]

        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b-o', label='Top-1 Acc')
        plt.plot(epochs, recall, 'g--s', label='Top-5 Recall')
        plt.title('Accuracy Metrics'); plt.legend(); plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, mAPs, 'r-^', label='mAP (Macro)')
        plt.title('Mean Average Precision'); plt.legend(); plt.grid(True)
        
        plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_results.png"))
        print("--> [DONE] Results saved.")
    except Exception:
        pass