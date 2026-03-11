"""
MODULE: SCIENTIFIC BENCHMARK - BIRDS BASELINE (RESNET-50)
COMPATIBILITY: Matches 'train.py' protocol exactly.
FEATURE: Official Split + Bounding Box Cropping (Critical for Fairness).
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor, ResNetForImageClassification, TrainingArguments, Trainer, DefaultDataCollator
)

# --- CONFIGURATION ---
DATA_ROOT = "dataset_birds"
OUTPUT_DIR = "D:/AI_Projects/model_birds_resnet50_baseline"
MODEL_NAME = "microsoft/resnet-50"

BATCH_SIZE = 32
EPOCHS = 40  # Khớp với ViT Birds
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ImageFile.LOAD_TRUNCATED_IMAGES = True # Fix lỗi ảnh hỏng

# --- DATASET CLASS VỚI LOGIC CẮT BOUNDING BOX ---
class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split="train", processor=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.processor = processor
        
        # Check files
        for f in ['images.txt', 'train_test_split.txt', 'bounding_boxes.txt']:
            if not os.path.exists(os.path.join(root_dir, f)):
                raise FileNotFoundError(f"[CRITICAL] Thiếu file '{f}' trong {root_dir}")

        # 1. Load Metadata
        df_imgs = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['id', 'filepath'])
        df_split = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['id', 'is_train'])
        df_labels = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['id', 'class_id'])
        df_bbox = pd.read_csv(os.path.join(root_dir, 'bounding_boxes.txt'), sep=' ', names=['id', 'x', 'y', 'w', 'h'])
        df_classes = pd.read_csv(os.path.join(root_dir, 'classes.txt'), sep=' ', names=['id', 'class_name'])

        # 2. Merge Data
        df = pd.merge(df_imgs, df_split, on='id')
        df = pd.merge(df, df_labels, on='id')
        df = pd.merge(df, df_bbox, on='id') # Thêm thông tin BBox

        target = 1 if split == "train" else 0
        self.data = df[df['is_train'] == target].reset_index(drop=True)
        self.classes = df_classes['class_name'].tolist()
        print(f"[DATA] {split.upper()} Set: {len(self.data)} images (Official Split + BBox).")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filepath'])
        label = int(row['class_id']) - 1
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # --- CRITICAL: CẮT ẢNH THEO BOUNDING BOX (Công bằng với ViT) ---
            # Tọa độ trong file là x, y, width, height
            x, y, w, h = row['x'], row['y'], row['w'], row['h']
            image = image.crop((x, y, x + w, y + h))
            # -------------------------------------------
            
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long)
            }
        return image, label

# --- METRICS ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    try: recall_5 = top_k_accuracy_score(labels, probs, k=5, labels=np.arange(probs.shape[1]))
    except: recall_5 = 0.0
    try: 
        labels_onehot = label_binarize(labels, classes=range(probs.shape[1]))
        m_ap = average_precision_score(labels_onehot, probs, average='macro')
    except: m_ap = 0.0
    return {"accuracy": acc, "recall_at_5": recall_5, "mAP": m_ap, "f1_score": f1}

# --- MAIN ---
def main():
    print(f"[INIT] TRAINING BIRDS RESNET-50 | BBOX CROP: ENABLED | EPOCHS: {EPOCHS}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    try:
        train_ds = CUB200Dataset(DATA_ROOT, split="train", processor=processor)
        test_ds = CUB200Dataset(DATA_ROOT, split="test", processor=processor)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Clean class names
    labels_clean = [c.split('.')[-1].replace('_', ' ') for c in train_ds.classes]
    id2label = {str(i): c for i, c in enumerate(labels_clean)}
    label2id = {c: str(i) for i, c in enumerate(labels_clean)}
    
    model = ResNetForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels_clean),
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR, per_device_train_batch_size=BATCH_SIZE, num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE, save_strategy="epoch", eval_strategy="epoch",
        save_total_limit=1, fp16=(DEVICE=="cuda"), report_to="none",
        load_best_model_at_end=True, metric_for_best_model="accuracy", dataloader_num_workers=0
    )
    
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
        data_collator=DefaultDataCollator(), compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("[DONE] BIRDS TRAINING FINISHED.")

if __name__ == "__main__":
    main()