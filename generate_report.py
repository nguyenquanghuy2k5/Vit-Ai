"""
MODULE: BENCHMARK REPORT GENERATOR
DESCRIPTION: 
    Executes full inference on the Test set to generate:
    1. Global Metrics (mAP, Recall@K, Accuracy).
    2. Per-class Metrics (Precision, Recall, F1 for each of 120 breeds).
    3. Exports results to CSV for documentation.
"""

import torch
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, 
    top_k_accuracy_score, 
    classification_report,
    average_precision_score
)
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.utils.data import DataLoader
# Import Dataset class from your main training script
from train_dogs import StanfordDogsDataset, DATA_ROOT, OUTPUT_DIR, MODEL_NAME

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BATCH_SIZE = 32
REPORT_PATH = os.path.join(OUTPUT_DIR, "detailed_benchmark_report.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "global_metrics_summary.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"[INIT] System: {DEVICE.upper()}")
    print(f"[PROCESS] Loading Model artifacts from: {OUTPUT_DIR}")
    
    # 1. Load Resources
    try:
        processor = ViTImageProcessor.from_pretrained(OUTPUT_DIR)
        model = ViTForImageClassification.from_pretrained(OUTPUT_DIR)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Could not load model. Have you trained it yet?\nDetail: {e}")
        return

    # 2. Prepare Test Data
    test_dataset = StanfordDogsDataset(DATA_ROOT, split="test", processor=processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"[DATA] Test Set Size: {len(test_dataset)} images")
    print(f"[PROCESS] Running Inference (Forward Pass)...")

    # 3. Inference Loop
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(torch.argmax(probs, dim=-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate batches
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    # 4. Compute Global Metrics
    print("[ANALYSIS] Computing Global Metrics (mAP, Recall@K)...")
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Recall@K
    recall_1 = top_k_accuracy_score(y_true, y_prob, k=1)
    recall_3 = top_k_accuracy_score(y_true, y_prob, k=3)
    recall_5 = top_k_accuracy_score(y_true, y_prob, k=5)
    
    # mAP (Mean Average Precision)
    num_classes = len(test_dataset.classes)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    try:
        mAP = average_precision_score(y_true_bin, y_prob, average="macro")
    except:
        mAP = 0.0

    global_metrics = {
        "Accuracy_Top1": round(acc, 4),
        "Recall_at_3": round(recall_3, 4),
        "Recall_at_5": round(recall_5, 4),
        "mAP_Macro": round(mAP, 4)
    }

    # 5. Compute Per-Class Metrics (Detailed Report)
    print("[ANALYSIS] Generating Class-wise Performance Report...")
    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=test_dataset.classes, 
        output_dict=True
    )
    
    # Convert to Pandas DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    
    # Filter out macro/weighted average rows for cleaner class list
    df_classes = df_report.iloc[:-3, :].copy() # Exclude 'accuracy', 'macro avg', 'weighted avg'
    
    # 6. Save Artifacts
    print(f"[EXPORT] Saving results...")
    
    # Save CSV
    df_classes.to_csv(REPORT_PATH)
    print(f"   -> Detailed Report: {REPORT_PATH}")
    
    # Save JSON Summary
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(global_metrics, f, indent=4)
    print(f"   -> Global Summary:  {SUMMARY_PATH}")

    # 7. Print Final Summary to Console
    print("\n" + "="*50)
    print("🏆 FINAL BENCHMARK RESULTS")
    print("="*50)
    print(f"Top-1 Accuracy:  {global_metrics['Accuracy_Top1']:.2%}")
    print(f"Top-5 Recall:    {global_metrics['Recall_at_5']:.2%}")
    print(f"Mean AP (mAP):   {global_metrics['mAP_Macro']:.4f}")
    print("-" * 50)
    print("Top 3 Best Performing Classes:")
    print(df_classes.sort_values(by="f1-score", ascending=False).head(3)[['precision', 'recall', 'f1-score']])
    print("-" * 50)
    print("Top 3 Worst Performing Classes:")
    print(df_classes.sort_values(by="f1-score", ascending=True).head(3)[['precision', 'recall', 'f1-score']])
    print("="*50)

if __name__ == "__main__":
    main()