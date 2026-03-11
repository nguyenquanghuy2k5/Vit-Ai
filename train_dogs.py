# ==============================================================================
# MODULE: STANFORD DOGS BENCHMARK TRAINING
# MODE: OFFICIAL SPLIT + BOUNDING BOX CROP + REPORT GENERATION
# FIX: FORCE SINGLE_LABEL_CLASSIFICATION
# ==============================================================================

import os
import tarfile
import scipy.io
import xml.etree.ElementTree as ET
from urllib.request import urlretrieve
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers import (
    ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, DefaultDataCollator
)
from torchvision.transforms import (
    Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor, ColorJitter
)
from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_recall_fscore_support

# ------------------------------------------------------------------------------
# SYSTEM CONFIGURATION
# ------------------------------------------------------------------------------
DATA_ROOT = "D:/AI_Projects/Stanford_Dogs_Data"
OUTPUT_DIR = "D:/AI_Projects/model_dogs_stanford_benchmark"

URL_IMAGES = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
URL_ANNOTATIONS = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
URL_LISTS = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"

MODEL_NAME = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Allow loading truncated images to prevent errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------------------------------------------------------
# DATASET IMPLEMENTATION
# ------------------------------------------------------------------------------
class StanfordDogsDataset(Dataset):
    def __init__(self, root, split="train", processor=None):
        self.root = root
        self.split = split
        self.processor = processor
        self.images_dir = os.path.join(root, "Images")
        self.annots_dir = os.path.join(root, "Annotation")
        
        # Initialize data
        self._prepare_data()
        
        print(f"[INFO] Loading {split} metadata from .mat files...")
        mat_file = "train_list.mat" if split == "train" else "test_list.mat"
        mat = scipy.io.loadmat(os.path.join(root, mat_file))
        
        # Parse metadata
        self.file_list = [f[0][0] for f in mat['file_list']]
        self.labels = [l[0] - 1 for l in mat['labels']]
        unique_classes = sorted(list(set([f.split('/')[0].split('-')[1] for f in self.file_list])))
        self.classes = unique_classes
        
        # Transforms configuration
        size = processor.size["height"]
        if split == "train":
            self.transform = Compose([
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ColorJitter(brightness=0.2, contrast=0.2),
                ToTensor(),
                Normalize(mean=processor.image_mean, std=processor.image_std),
            ])
        else:
            self.transform = Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize(mean=processor.image_mean, std=processor.image_std),
            ])

    def _prepare_data(self):
        os.makedirs(self.root, exist_ok=True)
        
        def download_extract(url, filename):
            f_path = os.path.join(self.root, filename)
            # Chỉ tải nếu file chưa tồn tại
            if not os.path.exists(f_path):
                print(f"[DOWNLOAD] Starting download: {filename}")
                with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                    urlretrieve(url, f_path, reporthook=lambda b, bsize, tsize: t.update(b*bsize - t.n))
                
                print(f"[EXTRACT] Extracting archive: {filename}")
                with tarfile.open(f_path, "r") as tar:
                    tar.extractall(path=self.root)
            else:
                # Nếu file đã tồn tại, kiểm tra xem đã giải nén chưa (cơ bản)
                # Để an toàn, chúng ta tin tưởng lần trước đã giải nén xong nếu file tar còn đó
                pass 

        download_extract(URL_IMAGES, "images.tar")
        download_extract(URL_ANNOTATIONS, "annotation.tar")
        download_extract(URL_LISTS, "lists.tar")

    def _get_bbox(self, path_wo_ext):
        xml_path = os.path.join(self.annots_dir, path_wo_ext)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            b = root.find('object').find('bndbox')
            return (
                int(b.find('xmin').text),
                int(b.find('ymin').text),
                int(b.find('xmax').text),
                int(b.find('ymax').text)
            )
        except:
            return None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        image_path = os.path.join(self.images_dir, file_path)
        
        # Load Image
        image = Image.open(image_path).convert("RGB")
        
        # Apply Bounding Box Crop
        bbox = self._get_bbox(file_path.rsplit('.', 1)[0])
        if bbox:
            image = image.crop(bbox)
            
        # [QUAN TRỌNG] Ép kiểu nhãn về Long (Interger) để tránh lỗi Loss Function
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return {
            "pixel_values": self.transform(image),
            "labels": label_tensor
        }

# ------------------------------------------------------------------------------
# METRICS & VISUALIZATION
# ------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """
    Computes standard classification metrics alongside retrieval metrics (mAP, Recall@K).
    
    Metrics:
    - Accuracy (Top-1)
    - Precision/Recall/F1 (Macro-averaged)
    - Recall@K (K=1, 3, 5): Probability that ground truth is in top K predictions.
    - mAP (Mean Average Precision): Area under Precision-Recall curve, averaged over classes.
    """
    predictions, labels = eval_pred
    
    # 1. Convert Logits to Probabilities (Softmax)
    # Tensor operation required for Top-K calculation
    probs_tensor = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1)
    probs_numpy = probs_tensor.numpy()
    
    # 2. Top-1 Predictions for Classification Metrics
    preds_top1 = np.argmax(predictions, axis=1)
    
    # --- CLASSIFICATION METRICS ---
    acc = accuracy_score(labels, preds_top1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds_top1, average='macro', zero_division=0
    )
    
    # --- RETRIEVAL METRICS (Recall@K) ---
    # Recall@1 is equivalent to Accuracy, but calculated via probability ranking
    recall_at_1 = top_k_accuracy_score(labels, probs_numpy, k=1, labels=np.arange(probs_numpy.shape[1]))
    recall_at_3 = top_k_accuracy_score(labels, probs_numpy, k=3, labels=np.arange(probs_numpy.shape[1]))
    recall_at_5 = top_k_accuracy_score(labels, probs_numpy, k=5, labels=np.arange(probs_numpy.shape[1]))
    
    # --- MEAN AVERAGE PRECISION (mAP) ---
    # mAP requires One-Hot Encoded labels for multi-class calculation
    # We infer the number of classes from the probability shape
    num_classes = probs_numpy.shape[1]
    labels_one_hot = label_binarize(labels, classes=range(num_classes))
    
    try:
        map_score = average_precision_score(labels_one_hot, probs_numpy, average="macro")
    except ValueError:
        map_score = 0.0 # Handle edge cases with insufficient class samples
        
    return {
        "accuracy": acc,
        "precision": precision,
        "recall_macro": recall,
        "f1_score": f1,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "mAP": map_score
    }

def plot_training_history(log_history, output_dir):
    print(f"[REPORT] Generating training charts...")
    
    loss_steps, loss_values = [], []
    eval_steps, eval_accs = [], []
    
    for entry in log_history:
        if 'loss' in entry:
            loss_steps.append(entry['step'])
            loss_values.append(entry['loss'])
        if 'eval_accuracy' in entry:
            eval_steps.append(entry['step'])
            eval_accs.append(entry['eval_accuracy'])

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_steps, loss_values, label='Training Loss', color='tab:red')
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(eval_steps, eval_accs, label='Validation Accuracy', color='tab:blue')
    plt.title('Validation Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "training_charts.png")
    plt.savefig(chart_path)
    print(f"[REPORT] Charts saved to: {chart_path}")
    plt.close()

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    print(f"[INIT] System initialized. Device: {DEVICE.upper()}")
    
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    train_dataset = StanfordDogsDataset(DATA_ROOT, split="train", processor=processor)
    test_dataset = StanfordDogsDataset(DATA_ROOT, split="test", processor=processor)
    
    id2label = {str(i): c for i, c in enumerate(train_dataset.classes)}
    label2id = {c: str(i) for i, c in enumerate(train_dataset.classes)}

    print(f"[DATA] Train Set: {len(train_dataset)} | Test Set: {len(test_dataset)}")
    print(f"[DATA] Classes: {len(train_dataset.classes)}")

    # [FIX CRITICAL] Thêm problem_type="single_label_classification"
    # Điều này bắt buộc Model dùng CrossEntropyLoss thay vì BCELoss
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(train_dataset.classes),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification" 
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        save_strategy="epoch",
        eval_strategy="epoch", # Đã fix lỗi tên biến
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=(DEVICE=="cuda"),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        dataloader_num_workers=2
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    print("[PROCESS] Starting training loop...")
    trainer.train()
    
    print("[PROCESS] Saving final model artifacts...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    plot_training_history(trainer.state.log_history, OUTPUT_DIR)
    
    stats = trainer.evaluate()
    print("-" * 50)
    print("FINAL EVALUATION RESULTS")
    print("-" * 50)
    print(f"Accuracy:      {stats['eval_accuracy']:.4f}")
    print(f"Top-5 Recall:  {stats['eval_top5_recall']:.4f}")
    print(f"Precision:     {stats['eval_precision']:.4f}")
    print(f"Recall:        {stats['eval_recall']:.4f}")
    print(f"F1-Score:      {stats['eval_f1_score']:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()