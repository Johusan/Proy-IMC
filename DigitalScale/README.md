# Digital Scale: Open-Source On-Device BMI Estimation from Smartphone Camera Images Trained on a Large-Scale Real-World Dataset

**Frederik Rajiv Manichand¹³, Robin Deuber¹, Robert Jakob¹,
Steve Swerling³, Jamie Rosen³, Elgar Fleisch¹², Patrick Langer¹**

¹ Centre for Digital Health Interventions, ETH Zurich  
² Centre for Digital Health Interventions, University of St. Gallen  
³ WayBetter  

Emails:  
<rmanichand@ethz.ch>, <rdeuber@ethz.ch>, <rjakob@ethz.ch>, 
<steve@waybetter.app>, <jamie@waybetter.app>, <efleisch@ethz.ch>, <planger@ethz.ch>


https://github.com/user-attachments/assets/837967a8-4e31-49c9-b5ee-c75b84929575



https://github.com/user-attachments/assets/1c26d763-353e-4c09-9d56-0b8373f4b201



---

<!-- 
    ## Table of Contents
    1. [Abstract](#abstract)
    2. [Introduction](#introduction)
    3. [Repository Structure](#repository-structure)
    4. [Getting Started](#getting-started)
       - [Prerequisites](#prerequisites)
       - [Installation](#installation)
    5. [Data](#data)
    6. [Methods](#methods)
       - [Preprocessing](#preprocessing)
       - [Model Architecture](#model-architecture)
       - [Training](#training)
       - [Evaluation](#evaluation)
    7. [Results](#results)
    8. [Usage](#usage)
    9. [Citation](#citation)
    10. [License](#license)
    11. [Contact](#contact)
-->

This repository contains code for **Digital Scale**, a system for robust on-device estimation of body‐mass index (BMI) directly from smartphone camera images. We train on a large-scale, real-world dataset collected in collaboration with WayBetter and Centre for Digital Health Interventions.

## Abstract
>Estimating BMI from camera images enables rapid weight assessment when traditional methods are impractical, such as in telehealth or emergencies. Existing approaches have been limited to datasets of up to 14,500 images. In this study, we present a deep learning‑based BMI estimation method trained on our **WayBED dataset**—a large, proprietary collection of **84,963 smartphone images** from **25,353 individuals**. We introduce an automatic filtering method that uses posture clustering and person detection to curate the dataset by removing low‑quality images, such as those with atypical postures or incomplete views. This process retained **71,322 high‑quality images** suitable for training. We achieve a **Mean Absolute Percentage Error (MAPE) of 7.9%** on our hold‑out test set (WayBED data) using full‑body images, the lowest value in the published literature to the best of our knowledge. Further, we achieve an **MAPE of 13%** on the completely unseen (during training) VisualBodyToBMI dataset, comparable with state‑of‑the‑art approaches trained on it, demonstrating robust generalization. We deploy the full pipeline—including image filtering and BMI estimation—on Android devices using the **CLAID framework**. We release our complete code for model training, filtering, and the CLAID package for mobile deployment as open‑source contributions.


## Repository Structure

```
├── app/                              # Android app implementation for real time BMI estimation using CLAID
├── docs/                             # Documentation files
├── get_started/                      # Quick start guide with minimal setup for BMI prediction
├── src/                              # Source code
├── src/scripts                       # Scripts for training, evaluation, and deployment
└── trained_models/keypoint_models/   # Pre-trained keypoint models for posture clustering
```

## 🚀 Getting Started

For a quick start with BMI prediction, check out the **[`get_started/`](get_started/)** folder. This streamlined setup allows you to run BMI inference in minutes:

### Quick Setup
1. **Navigate to get_started folder**:
   ```bash
   cd get_started
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights and sample data**:
   - Contact: **rmanichand@ethz.ch** or **planger@ethz.ch**
   - Request: "Digital Scale model weights and sample data"

4. **Run BMI prediction**:
   ```bash
   python predict_bmi.py
   ```

### What's Included
- **Simplified model definitions** (SEDenseNet121/201)
- **Dataset handling utilities** for the Visual Body to BMI dataset
- **Inference script** with command-line options
- **Complete setup instructions** and troubleshooting guide

### Expected Performance
- **MAPE ~7.9%** on WayBED dataset using SEDenseNet121
- **Real-time inference** on modern GPUs
- **CPU compatibility** for environments without GPU access

**Note**: The `get_started/` folder excludes model weights and sample data to keep the repository lightweight. These files can be downloaded as a zip package from the authors.

---

## Scripts

The `src/scripts/` directory contains various Python scripts for different aspects of the BMI estimation pipeline. The scripts are organized into two main categories:

### Model Training, Evaluation and Deployment

| Script | Description |
|--------|-------------|
| [`train_visual_bmi_densenet.py`](src/scripts/train_visual_bmi_densenet.py) | Trains DenseNet models for BMI estimation from images, with support for parallel GPU training and different model architectures |
| [`train_densenet.py`](src/scripts/train_densenet.py) | Basic DenseNet training script for BMI estimation using the WayBetter dataset |
| [`finetune_densenet.py`](src/scripts/finetune_densenet.py) | Fine-tunes pre-trained DenseNet models on the VisualBodyToBMI dataset with various learning rate strategies and freezing configurations |
| [`hyperparameter_search.py`](src/scripts/hyperparameter_search.py) | Performs hyperparameter optimization for DenseNet training with various learning rate schedules and freezing strategies |
| [`evaluate_visual_bmi_models.py`](src/scripts/evaluate_visual_bmi_models.py) | Evaluates multiple trained BMI estimation models by finding checkpoints and running inference in parallel across GPUs |
| [`extensive_evaluation.py`](src/scripts/extensive_evaluation.py) | Runs comprehensive evaluation experiments with different model configurations and hyperparameters for systematic performance analysis |
| [`densenet_forwardpass.py`](src/scripts/densenet_forwardpass.py) | Performs forward pass inference using trained DenseNet models to generate BMI predictions from images |
| [`run_waybetter_pipeline.py`](src/scripts/run_waybetter_pipeline.py) | Executes the complete BMI estimation pipeline using the CLAID framework for on-device deployment |

### Preprocessing and Filtering

| Script | Description |
|--------|-------------|
| [`train_posture_clustering.py`](src/scripts/train_posture_clustering.py) | Trains posture clustering models using keypoint data and provides interactive tools to explore and identify good vs bad posture clusters |
| [`predict_posture.py`](src/scripts/predict_posture.py) | Predicts posture clusters from raw keypoint data using trained clustering models (only clusters 2 and 3 are considered good postures) |
| [`parse_visual_bmi_dataset.py`](src/scripts/parse_visual_bmi_dataset.py) | Parses the Visual-body-to-BMI dataset structure and extracts metadata (weight, height, gender) from filenames into a structured DataFrame |
| [`bounding_box_forwardpass.py`](src/scripts/bounding_box_forwardpass.py) | Runs bounding box detection on images to identify person regions for subsequent processing |
| [`face_bounding_box_forwardpass.py`](src/scripts/face_bounding_box_forwardpass.py) | Detects face bounding boxes in images using RetinaFace for face-based BMI estimation approaches |
| [`keypoints_forwardpass.py`](src/scripts/keypoints_forwardpass.py) | Performs keypoint detection on images to extract human pose information for posture analysis |
| [`calculate_bbox_area_ratio.py`](src/scripts/calculate_bbox_area_ratio.py) | Calculates the area ratio of detected bounding boxes relative to image size for quality assessment |

