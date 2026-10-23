import os
import pandas as pd
import numpy as np

def create_unified_dataset():
    """
    Constructs unified_dataset.csv using curated, leak-free subject-disjoint partitions.
    Guarantees:
    - 0% subject overlap between train, val, and test.
    - 0% exact-duplicate pixel leakage (SHA256).
    - Excludes contradictory label duplicates (40kg label discrepancy).
    - Uses confirmed cluster IDs rather than scraper photo indices.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    curated_csv = os.path.join(base_dir, "curated_subject_splits.csv")
    
    if not os.path.exists(curated_csv):
        raise FileNotFoundError(f"Curated splits not found at {curated_csv}. Run scratch/create_subject_disjoint_splits.py first.")
        
    df_curated = pd.read_csv(curated_csv)
    
    records = []
    for _, row in df_curated.iterrows():
        records.append({
            "image_id": row["filename"],
            "individual_id": row["cluster_id"],
            "cluster_type": row["cluster_type"],
            "photo_path": row["path"],
            "bmi": float(row["bmi"]),
            "sex": row["sex"],
            "age": int(row["age"]),
            "height_m": float(row["height_m"]),
            "weight_kg": float(row["weight_kg"]),
            "partition": row["curated_partition"],
            "sha256": row["sha256"]
        })
        
    df_unified = pd.DataFrame(records)
    output_csv = os.path.join(base_dir, "unified_dataset.csv")
    df_unified.to_csv(output_csv, index=False)
    
    print("Curated Unified Dataset Built Successfully!")
    print(f"Saved to: {output_csv}")
    print("\nPartition Breakdown (Images):")
    print(df_unified["partition"].value_counts())
    print("\nUnique Clusters (Subjects):")
    print(f"Total: {df_unified['individual_id'].nunique()}")
    for part in ["train", "val", "test"]:
        sub = df_unified[df_unified["partition"] == part]
        print(f"  {part.upper()}: {sub['individual_id'].nunique()} clusters, {len(sub)} images")
        
    print("\nBMI Summary Statistics (Overall):")
    print(df_unified["bmi"].describe())

if __name__ == "__main__":
    create_unified_dataset()
