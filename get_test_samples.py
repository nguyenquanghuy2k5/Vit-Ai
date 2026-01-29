# ==============================================================================
# TOOL: OFFICIAL TEST SET SAMPLER (CUB-200-2011)
# DESCRIPTION: Extracts unseen test images based on 'train_test_split.txt'
# ==============================================================================

import os
import sys
import shutil
import pandas as pd

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
DATASET_DIR = "./dataset_birds"         # Thư mục dataset gốc
OUTPUT_DIR  = "./test_samples_official" # Thư mục chứa ảnh kết quả
NUM_SAMPLES = 10                        # Số lượng ảnh muốn lấy

def extract_test_samples():
    print(f"[INFO] Initializing sampler for dataset: {DATASET_DIR}")

    # 1. Validate Dataset Structure
    if not os.path.exists(DATASET_DIR):
        print(f"[CRITICAL ERROR] Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    try:
        # 2. Load Metadata
        # images.txt: [id, filepath]
        img_path = os.path.join(DATASET_DIR, "images.txt")
        # train_test_split.txt: [id, is_train] (0 = Test, 1 = Train)
        split_path = os.path.join(DATASET_DIR, "train_test_split.txt")

        if not os.path.exists(img_path) or not os.path.exists(split_path):
            print("[ERROR] Metadata files (images.txt, train_test_split.txt) are missing.")
            sys.exit(1)

        print("[INFO] Loading metadata files...")
        df_imgs = pd.read_csv(img_path, sep=' ', names=['id', 'filepath'])
        df_split = pd.read_csv(split_path, sep=' ', names=['id', 'is_train'])

        # 3. Merge and Filter Test Set
        # is_train == 0 means Official Test Set
        df_merged = df_imgs.merge(df_split, on='id')
        df_test = df_merged[df_merged['is_train'] == 0]
        
        total_test = len(df_test)
        print(f"[INFO] Found {total_test} images in the Official Test Set.")

        # 4. Random Sampling
        if total_test < NUM_SAMPLES:
            print(f"[WARNING] Requested {NUM_SAMPLES} samples but only found {total_test}.")
            samples = df_test
        else:
            samples = df_test.sample(n=NUM_SAMPLES, random_state=42) # Fixed seed for reproducibility

        # 5. Export Images
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR) # Clean old samples
        os.makedirs(OUTPUT_DIR)

        print(f"[PROCESS] Exporting {len(samples)} images to '{OUTPUT_DIR}'...")
        
        count = 0
        for _, row in samples.iterrows():
            # Original path in dataset
            src_path = os.path.join(DATASET_DIR, "images", row['filepath'])
            
            # Create a descriptive filename (Species_Name_ImageID.jpg)
            # Replace '/' with '_' to flatten directory structure
            new_filename = row['filepath'].replace("/", "_")
            dst_path = os.path.join(OUTPUT_DIR, new_filename)

            try:
                shutil.copy(src_path, dst_path)
                print(f"   + Copied: {new_filename}")
                count += 1
            except FileNotFoundError:
                print(f"[WARNING] File not found (Skipped): {src_path}")

        print("-" * 50)
        print(f"[SUCCESS] Extraction complete. {count} images are ready in '{OUTPUT_DIR}'.")
        print("-" * 50)

    except Exception as e:
        print(f"[CRITICAL ERROR] An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    extract_test_samples()