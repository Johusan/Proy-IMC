# BMI Estimation App - Core Implementation

This repository contains the core implementation of the DigitalScale app.

## Overview

The app uses a multi-stage pipeline:
1. **Person Detection** - EfficientDet-Lite4 to detect and locate people in images
2. **Posture Analysis** - MoveNet Thunder for keypoint detection and posture classification
3. **BMI Estimation** - Custom DenseNet model to estimate BMI from processed images
<img width="1560" height="935" alt="deployed" src="https://github.com/user-attachments/assets/86a44d0c-b276-4e21-b19d-4619ebc045eb" />

## Project Structure

```
app/
├── conversion/
│   ├── convert_model_executorch_fixed.py
│   └── export_posture_model_params.py
├── android/app/src/main/
│   ├── java/com/bmiapp/estimator/
│   │   ├── MainActivity.kt
│   │   ├── CameraActivity.kt
│   │   ├── ResultActivity.kt
│   │   ├── EfficientDetPersonDetector.kt
│   │   ├── PostureDetectionClustering.kt
│   │   ├── ImagePreprocessor.kt
│   │   ├── BoundingBoxImageView.kt
│   │   └── SkeletonOverlayView.kt
│   ├── AndroidManifest.xml
│   └── build.gradle
└── README.md
```

## Setup Instructions

### Prerequisites

1. **Android Studio** with Android SDK
2. **Python 3.8+** with the following packages:
   ```bash
   pip install torch torchvision executorch numpy scikit-learn
   ```

### Model Setup

1. **Download Pre-trained Models:**
   - **EfficientDet-Lite4**: Download from [TensorFlow Hub](https://tfhub.dev/tensorflow/efficientdet/lite4/2) or use TensorFlow Lite Task Library
   - **MoveNet Thunder**: Download from [TensorFlow Hub](https://tfhub.dev/google/movenet/singlepose/thunder/4)
   - Place them in `android/app/src/main/assets/` as:
     - `efficientdet_lite4.tflite`
     - `movenet_thunder_float16.tflite`

2. **Convert Your BMI Model:**
   ```bash
   cd conversion
   python convert_model_executorch_fixed.py
   ```
   This will create `bmi_model_fixed.pte` in `android/app/src/main/assets/`

3. **Export Posture Parameters:**
   ```bash
   cd conversion
   python export_posture_model_params.py
   ```
   This will create `posture_model_params.json` in `android/app/src/main/assets/`

### Required Assets

After running the conversion scripts, ensure you have these files in `android/app/src/main/assets/`:
- `efficientdet_lite4.tflite` (person detection)
- `movenet_thunder_float16.tflite` (keypoint detection)
- `posture_model_params.json` (posture classification)
- `bmi_model_fixed.pte` (BMI estimation)

## Core Components

### Main Activities
- **MainActivity**: Entry point, handles permissions and navigation
- **CameraActivity**: Camera interface using CameraX API
- **ResultActivity**: Main processing hub and result display

### Detection Classes
- **EfficientDetPersonDetector**: Person detection with bounding box extraction
- **PostureDetectionClustering**: Keypoint detection and posture analysis
- **ImagePreprocessor**: Image preprocessing for BMI model

### UI Components
- **BoundingBoxImageView**: Displays person detection bounding box
- **SkeletonOverlayView**: Shows body keypoints and skeleton connections

## Key Features

- **Multi-stage validation**: Person detection → Posture analysis → BMI estimation
- **Quality checks**: Confidence thresholds and bounding box size validation
- **Visual feedback**: Bounding boxes and skeleton overlays
- **Error handling**: Comprehensive error messages for poor quality inputs

## Dependencies

### Android Dependencies (build.gradle)
- CameraX for camera functionality
- TensorFlow Lite for model inference
- PyTorch ExecuTorch for BMI model
- Glide for image loading
- Material Design components

### Permissions (AndroidManifest.xml)
- Camera access
- Storage read/write
- Notifications

## Usage

1. **Take Photo**: Use camera or select from gallery
2. **Person Detection**: App validates person is visible and well-positioned
3. **Posture Analysis**: Ensures person is standing upright
4. **BMI Estimation**: Runs neural network to estimate BMI
5. **Result Display**: Shows BMI score with category and visual overlays

## Model Information

- **Person Detection**: EfficientDet-Lite4 (640x640 input)
- **Keypoint Detection**: MoveNet Thunder (256x256 input, 17 COCO keypoints)
- **Posture Classification**: Custom clustering model (StandardScaler + PCA + K-Means)
- **BMI Estimation**: Custom DenseNet model (224x224 input, ImageNet normalization)

## Error Handling

The app includes validation for:
- No person detected
- Low confidence detections (< 50%)
- Poor posture (not standing upright)
- Small bounding box size (< 25% of image)
- Model loading failures
