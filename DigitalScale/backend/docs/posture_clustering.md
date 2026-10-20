# Posture Clustering Workflows

Two extremely simple workflows for posture analysis using keypoint clustering.

## 🎯 **Overview**

- **Workflow 1**: **Predict posture** on new data using pre-trained models
- **Workflow 2**: **Train new clustering** models with interactive cluster exploration

---

## 🚀 **Workflow 1: Posture Prediction (Inference)**

**Use case**: You have trained models and want to classify new keypoint data.

### Quick Start

```python
from src.scripts.predict_posture import predict_posture

# Your raw keypoint data (x,y coordinates for each keypoint)
keypoints_df = ...  # DataFrame with keypoint columns like 'nose-x', 'nose-y', etc.
bounding_boxes_df = ...  # DataFrame with bounding box info

# Predict postures
results = predict_posture(keypoints_df, bounding_boxes_df)

# Results
print(f"Clusters: {results['clusters']}")           # [0, 1, 2, 3, ...]
print(f"Good posture: {results['good_posture']}")   # [False, True, True, False, ...]
print(f"Confidence: {results['confidence']}")       # [0.85, 0.92, 0.78, ...]
```

### Command Line

```bash
cd src
python scripts/predict_posture.py
```

### What you get
- **Cluster labels**: 0, 1, 2, 3 (4 clusters total)
- **Good posture flags**: `True` for clusters 1 & 2, `False` for others
- **Confidence scores**: Higher = more confident prediction

---

## 🏋️ **Workflow 2: Train New Clustering**

**Use case**: Train new clustering models and identify which clusters are "good" postures.

### Option A: Command Line (Simple)

```bash
cd src
python scripts/train_posture_clustering.py
```

This will:
1. Show elbow curve to choose optimal K
2. Train clustering with K=4
3. Show cluster exploration
4. Save models to `trained_models/keypoint_models/`

### Option B: Notebook (Interactive)

```bash
jupyter notebook notebooks/posture_clustering_example.ipynb
```

This provides:
- **Interactive cluster explorer** with sliders
- **Visual analysis** of each cluster
- **Sample-by-sample inspection**

### Option C: Python Import

```python
from src.scripts.train_posture_clustering import (
    train_clustering, 
    InteractiveClusterExplorer
)

# Train models
results = train_clustering(n_clusters=4)

# Explore interactively (in notebook)
explorer = InteractiveClusterExplorer(results)
explorer.show()
```

---

## 📁 **File Structure**

```
src/
├── scripts/
│   ├── predict_posture.py          # Workflow 1: Inference
│   └── train_posture_clustering.py # Workflow 2: Training
└── helpers/
    └── keypoint_clustering.py      # Helper functions

trained_models/
└── keypoint_models/
    ├── scaler.pkl                  # Preprocessing model
    ├── pca.pkl                     # Dimensionality reduction
    └── kmeans.pkl                  # Clustering model

notebooks/
└── posture_clustering_example.ipynb # Interactive training
```

---

## 🎛️ **Key Configuration**

### Prediction Script
- **Good clusters**: Currently set to `[1, 2]` in `predict_posture.py`
- **Model path**: `trained_models/keypoint_models/`

### Training Script
- **Default clusters**: 4 (adjustable)
- **PCA variance**: 95% (adjustable)
- **Random state**: 42 (reproducible)

---

## 🔧 **Customization**

### Change which clusters are "good"
Edit `predict_posture.py`:
```python
# Change this line based on your cluster exploration
good_posture = np.isin(clusters, [1, 2])  # Update cluster IDs
```

### Adjust training parameters
```python
results = train_clustering(
    n_clusters=5,      # More clusters
    pca_variance=0.90, # Less variance retained
    random_state=123   # Different seed
)
```

---

## 📊 **Data Format**

### Input: Raw Keypoint Data
```python
# keypoints_df columns
['nose-x', 'nose-y', 'left_eye-x', 'left_eye-y', ...]  # All keypoint coordinates

# bounding_boxes_df columns  
['image_id', 'x1', 'y1', 'width', 'height']  # Bounding box info
```

### Output: Prediction Results
```python
{
    'clusters': array([0, 1, 2, 1]),      # Cluster assignments
    'good_posture': array([False, True, True, True]),  # Good posture flags
    'confidence': array([0.85, 0.92, 0.78, 0.88])     # Confidence scores
}
```

---

## 🎯 **Typical Workflow**

1. **First time setup**:
   ```bash
   # Train initial models
   python src/scripts/train_posture_clustering.py
   
   # Or use interactive notebook
   jupyter notebook notebooks/posture_clustering_example.ipynb
   ```

2. **Identify good clusters** using exploration tools

3. **Update prediction script** if good clusters changed

4. **Use for inference**:
   ```python
   from src.scripts.predict_posture import predict_posture
   results = predict_posture(new_keypoint_data, bounding_boxes)
   ```

---

## ⚠️ **Important Notes**

- **Clusters 1 & 2 are currently hardcoded as "good"** - update based on your analysis
- **Models expect normalized keypoint coordinates** - this is handled automatically
- **Bounding box data is required** for normalization
- **Interactive features require** `ipywidgets` for notebooks

---

## 🚨 **Troubleshooting**

### "No module named 'ipywidgets'"
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### "FileNotFoundError: scaler.pkl"
```bash
# Train models first
python src/scripts/train_posture_clustering.py
```

### "AttributeError: transform"
Your models might be corrupted. Retrain:
```bash
rm -rf trained_models/keypoint_models/
python src/scripts/train_posture_clustering.py
``` 