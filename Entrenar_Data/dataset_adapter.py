import os
import re
import pandas as pd
import numpy as np

def parse_csv_data(csv_path, images_dir, default_partition="train"):
    """
    Parses un-headered CSV files from FVL2020 dataset format.
    Format per row:
    col 0: filename (e.g., 001197_M_18_180340_5443109.jpg)
    col 1: BMI (float)
    col 2: Sex (1=Male, 0=Female)
    ...
    col -3: Age (int)
    col -2: Height in meters (float)
    col -1: Weight in kg (float)
    """
    if not os.path.exists(csv_path):
        print(f"Warning: File {csv_path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, header=None)
    records = []
    
    for idx, row in df.iterrows():
        filename = str(row[0]).strip()
        bmi = float(row[1])
        sex = int(row[2])
        height_m = float(row.iloc[-2])
        weight_kg = float(row.iloc[-1])
        
        # Build absolute path and check if file exists
        photo_path = os.path.join(images_dir, filename)
        if not os.path.exists(photo_path):
            # Check if there is a png/jpg extension mismatch
            base_name, _ = os.path.splitext(filename)
            alt_path_jpg = os.path.join(images_dir, base_name + ".jpg")
            alt_path_png = os.path.join(images_dir, base_name + ".png")
            if os.path.exists(alt_path_jpg):
                photo_path = alt_path_jpg
            elif os.path.exists(alt_path_png):
                photo_path = alt_path_png
            else:
                continue # Skip missing images
        
        # Extract individual ID (first part of filename before underscore)
        parts = filename.split('_')
        individual_id = parts[0] if len(parts) > 1 else f"ind_{idx}"
        
        records.append({
            "image_id": filename,
            "individual_id": individual_id,
            "photo_path": os.path.abspath(photo_path),
            "bmi": bmi,
            "sex": "M" if sex == 1 else "F",
            "height_m": height_m,
            "weight_kg": weight_kg,
            "partition": default_partition
        })
        
    return pd.DataFrame(records)

def create_unified_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "Image Dataset")
    
    train_csv = os.path.join(dataset_dir, "Image_train.csv")
    train_img_dir = os.path.join(dataset_dir, "Image_train")
    
    test_csv = os.path.join(dataset_dir, "Image_test.csv")
    test_img_dir = os.path.join(dataset_dir, "Image_test")
    
    print(f"Processing training data from: {train_csv}")
    df_train_raw = parse_csv_data(train_csv, train_img_dir, default_partition="train")
    print(f"   -> Found {len(df_train_raw)} valid training images.")
    
    print(f"Processing test data from: {test_csv}")
    df_test = parse_csv_data(test_csv, test_img_dir, default_partition="test")
    print(f"   -> Found {len(df_test)} valid test images.")
    
    # Split training set into train (85%) and val (15%) grouped by individual_id
    unique_individuals = df_train_raw['individual_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_individuals)
    
    val_count = int(0.15 * len(unique_individuals))
    val_individuals = set(unique_individuals[:val_count])
    
    # Assign 'val' partition for individuals in val_set
    df_train_raw.loc[df_train_raw['individual_id'].isin(val_individuals), 'partition'] = 'val'
    
    df_unified = pd.concat([df_train_raw, df_test], ignore_index=True)
    
    output_csv = os.path.join(base_dir, "unified_dataset.csv")
    df_unified.to_csv(output_csv, index=False)
    
    print("\nDataset Unified Successfully!")
    print(f"   Saved to: {output_csv}")
    print("   Partition Breakdown:")
    print(df_unified['partition'].value_counts())
    print("\n   BMI Summary Statistics:")
    print(df_unified['bmi'].describe())

if __name__ == "__main__":
    create_unified_dataset()
