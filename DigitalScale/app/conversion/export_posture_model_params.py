import pickle
import json
import numpy as np

# Load scaler
with open("trained_models/keypoint_models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load PCA
with open("trained_models/keypoint_models/pca.pkl", "rb") as f:
    pca = pickle.load(f)

# Load KMeans
with open("trained_models/keypoint_models/kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

params = {
    "scaler": {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    },
    "pca": {
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist()
    },
    "kmeans": {
        "centers": kmeans.cluster_centers_.tolist()
    }
}

with open("posture_model_params.json", "w") as f:
    json.dump(params, f, indent=2)

print("Exported all posture model parameters to posture_model_params.json") 