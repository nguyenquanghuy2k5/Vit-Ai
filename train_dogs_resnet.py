"""
MODULE: SCIENTIFIC BENCHMARK - DOGS BASELINE (RESNET-50)
COMPATIBILITY: Matches 'train_dogs.py' protocol exactly.
FEATURE: Official Split + Bounding Box Cropping (Critical for Fairness).
"""

import os
import torch
import numpy as np
import scipy.io
import tarfile
import xml.etree.ElementTree as ET
from urllib.request import urlretrieve
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor, ResNetForImageClassification, TrainingArguments, Trainer, DefaultDataCollator
)

# --- CONFIGURATION ---
DATA_ROOT = "C:/Users/Hi/Desktop/food ai/dataset_dogs"
OUTPUT_DIR = "D:/AI_Projects/model_dogs_resnet50_baseline"
MODEL_NAME = "microsoft/resnet-50"

BATCH_SIZE = 32
EPOCHS = 20  # Khớp với ViT Dogs
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ImageFile.LOAD_TRUNCATED_IMAGES = True # Fix lỗi ảnh hỏng

# --- DATASET CLASS VỚI LOGIC CẮT BOUNDING BOX ---
class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", processor=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "Images")
        self.annotation_dir = os.path.join(root_dir, "Annotation")
        self.processor = processor
        
        # 1. Tự động tải dữ liệu nếu thiếu (Images, Annotation, Lists)
        self._check_and_download()
        
        # 2. Load Metadata chuẩn (.mat)
        mat_file = "train_list.mat" if split == "train" else "test_list.mat"
        mat = scipy.io.loadmat(os.path.join(root_dir, mat_file))
        
        self.file_list = [f[0][0] for f in mat['file_list']]
        self.labels = [l[0] - 1 for l in mat['labels']]
        self.classes = sorted(list(set([f.split('/')[0].split('-')[1] for f in self.file_list])))

    def _check_and_download(self):
        # Logic tải giống hệt train_dogs.py
        urls = {
            "lists.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
            "annotation.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
        }
        os.makedirs(self.root_dir, exist_ok=True)
        for filename, url in urls.items():
            path = os.path.join(self.root_dir, filename)
            extract_folder = "Annotation" if "annotation" in filename else ""
            # Nếu chưa có folder Annotation hoặc file .mat thì tải
            if (extract_folder and not os.path.exists(os.path.join(self.root_dir, extract_folder))) or \
               (not extract_folder and not os.path.exists(os.path.join(self.root_dir, "train_list.mat"))):
                if not os.path.exists(path):
                    print(f"--> Downloading {filename}...")
                    urlretrieve(url, path)
                print(f"--> Extracting {filename}...")
                with tarfile.open(path, "r") as tar:
                    tar.extractall(self.root_dir)

    def _get_bbox(self, idx):
        # Đọc file XML để lấy tọa độ con chó (Logic y hệt train_dogs.py)
        filename = self.file_list[idx]
        xml_path = os.path.join(self.annotation_dir, filename.split('.')[0]) # Bỏ đuôi .jpg
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            bndbox = root.find('object').find('bndbox')
            return (
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            )
        except:
            return None 

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.file_list[idx])
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # --- CRITICAL: CẮT BBOX (Công bằng với ViT) ---
            bbox = self._get_bbox(idx)
            if bbox:
                image = image.crop(bbox)
            # --------------------------------------------
            
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long)
            }
        return image, label

    def __len__(self):
        return len(self.file_list)

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
    print(f"[INIT] TRAINING DOGS RESNET-50 | BBOX CROP: ENABLED | EPOCHS: {EPOCHS}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    train_ds = StanfordDogsDataset(DATA_ROOT, split="train", processor=processor)
    test_ds = StanfordDogsDataset(DATA_ROOT, split="test", processor=processor)
    
    # Mapping
    id2label = {str(i): c for i, c in enumerate(train_ds.classes)}
    label2id = {c: str(i) for i, c in enumerate(train_ds.classes)}
    
    model = ResNetForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=len(train_ds.classes),
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
    print("[DONE] DOGS TRAINING FINISHED.")

if __name__ == "__main__":
    main()