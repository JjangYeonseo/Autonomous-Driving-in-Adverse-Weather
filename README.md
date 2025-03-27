# 🚗 Robust Object Detection in Adverse Weather Conditions

This project aims to develop a multimodal deep learning model for object detection that remains **robust under adverse weather conditions**, including fog, rain, snow, and night-time scenarios.

By fusing **camera (image)** and **LiDAR (point cloud)** data, the model can maintain high perception accuracy even in low-visibility environments—making it suitable for real-world autonomous driving applications.

## 🎯 Project Objectives

- Build a robust object detection model for various weather and lighting conditions
- Fuse image (RGB) and LiDAR (BEV) data to enhance detection reliability
- Use polygon-based annotation files for both image and LiDAR
- Align with real-world sensor configurations and environmental diversity

## 🧰 Requirements & Environment Setup

### 1. Python Version
- Python 3.9 (recommended via Anaconda)

### 2. Package Installation

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Key packages:
- `torch`, `torchvision` (CUDA 11.8 compatible)
- `open3d` (for processing `.pcd` files)
- `opencv-python`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pyyaml`, `seaborn`, `tqdm`

### 3. CUDA Support

- Tested with CUDA driver **12.8.1**
- PyTorch uses **CUDA 11.8 build**, which runs smoothly on higher versions

## 📁 Project Structure

```
project/
├── configs/
│   └── config.yaml              # Main configuration file
│
├── data/
│   ├── dataset.py               # Custom PyTorch Dataset (Image + LiDAR)
│   └── preprocessing/           # Auto-generated after preprocessing
│       ├── cleaned_data.csv     # Valid data samples
│       └── classes.txt          # List of all object classes
│
├── model/
│   └── fusion_model.py          # Fusion model (Image + LiDAR)
│
├── scripts/
│   ├── preprocess_data.py       # Cleans dataset, extracts classes
│   ├── train.py                 # Trains the fusion model
│   ├── test.py                  # Runs inference on test set
│   ├── evaluate.py              # Calculates accuracy and confusion matrix
│   └── run_pipeline.py          # Automates training → testing → evaluation → visualization
│
├── utils/
│   ├── lidar_utils.py           # Converts .pcd → BEV images
│   └── visualization.py         # Saves image predictions as visual output
│
├── checkpoints/                 # Model weights saved during training
├── outputs/                     # Predictions, confusion matrix, and visualizations
└── requirements.txt             # Python dependencies
```

## 🚀 Pipeline: Step-by-Step Guide

### Step 1: Preprocess the dataset
```bash
python scripts/preprocess_data.py
```
- Filters out invalid labels
- Creates `cleaned_data.csv` and `classes.txt`
- You must **edit `num_classes` in `config.yaml`** based on the number of lines in `classes.txt`

### Step 2: Train the model
```bash
python scripts/train.py --config configs/config.yaml
```
- Checkpoints are saved to `checkpoints/epoch_*.pt`

### Step 3: Test the model
```bash
python scripts/test.py --config configs/config.yaml
```
- Predictions are saved to `outputs/test_results.csv`

### Step 4: Evaluate performance
```bash
python scripts/evaluate.py
```
- Shows classification report and saves `confusion_matrix.png`

### Step 5: Visualize predictions
```python
from utils.visualization import visualize_predictions
import yaml
config = yaml.safe_load(open("configs/config.yaml"))
visualize_predictions(config, sample_count=10)
```
- Saves annotated images to `outputs/vis/`

### One Command for Everything (recommended!)
```bash
python scripts/run_pipeline.py
```
Runs the full pipeline: **Training → Testing → Evaluation → Visualization**

## 💡 Benefits

- Maintains high object detection performance in harsh weather
- Combines image + LiDAR input to compensate for low visibility
- Robust to noise, direction, and environmental variety
- Can be extended to real-time deployment or 3D perception models

## 🔮 Expected Outcomes

- Test accuracy target: **80%+**
- Class-wise performance analysis via confusion matrix
- Visual confirmation of predicted vs ground truth labels

## 📌 Notes

- `data/preprocessing/` and `outputs/` folders are created automatically
- It is recommended to add these folders to `.gitignore` if pushing to GitHub:

```
outputs/
data/preprocessing/
checkpoints/
```
## cf) Source Data Structure

The data is organized into two main categories:
1. **Labeling Data**: JSON files containing object and segmentation information for each image in different weather conditions.
2. **Raw Data**: Images in JPG format and LiDAR data in PCD format.

## cf) Training Data Structure

```
traindata/
├── labellingdata/
│   ├── TL_Hazy/
│   │   └── Hazy/
│   │       ├── Day/
│   │       │   ├── Back/
│   │       │   ├── Front/
│   │       │   ├── IR/
│   │       │   ├── Left/
│   │       │   ├── Lidar_Center/
│   │       │   ├── Lidar_Left/
│   │       │   ├── Lidar_Right/
│   │       │   └── Right/
│   │       └── Night/
│   ├── TL_Normal/
│   │   └── Normal/
│   │       ├── Day/
│   │       │   ├── Back/
│   │       │   ├── Front/
│   │       │   ├── IR/
│   │       │   ├── Left/
│   │       │   ├── Lidar_Center/
│   │       │   ├── Lidar_Left/
│   │       │   ├── Lidar_Right/
│   │       │   └── Right/
│   │       └── Night/
│   ├── TL_Rainy/
│   │   └── Rainy/
│   │       ├── Day/
│   │       │   ├── Back/
│   │       │   ├── Front/
│   │       │   ├── IR/
│   │       │   ├── Left/
│   │       │   ├── Lidar_Center/
│   │       │   ├── Lidar_Left/
│   │       │   ├── Lidar_Right/
│   │       │   └── Right/
│   │       └── Night/
│   └── TL_Snowy/
│       └── Snowy/
│           ├── Day/
│           │   ├── Back/
│           │   ├── Front/
│           │   ├── IR/
│           │   ├── Left/
│           │   ├── Lidar_Center/
│           │   ├── Lidar_Left/
│           │   ├── Lidar_Right/
│           │   └── Right/
│           └── Night/
└── sourcedata/
    ├── TS_Hazy/
    │   └── Hazy/
    │       ├── Day/
    │       │   ├── Back/
    │       │   ├── Front/
    │       │   ├── IR/
    │       │   ├── Left/
    │       │   ├── Lidar_Center/
    │       │   ├── Lidar_Left/
    │       │   ├── Lidar_Right/
    │       │   └── Right/
    │       └── Night/
    ├── TS_Normal/
    │   └── Normal/
    │       ├── Day/
    │       │   ├── Back/
    │       │   ├── Front/
    │       │   ├── IR/
    │       │   ├── Left/
    │       │   ├── Lidar_Center/
    │       │   ├── Lidar_Left/
    │       │   ├── Lidar_Right/
    │       │   └── Right/
    │       └── Night/
    │           ├── Lidar_Center/
    │           └── Lidar_Left/
    ├── TS_Rainy/
    │   └── Rainy/
    │       ├── Day/
    │       │   ├── Back/
    │       │   ├── Front/
    │       │   ├── IR/
    │       │   ├── Left/
    │       │   ├── Lidar_Center/
    │       │   ├── Lidar_Left/
    │       │   ├── Lidar_Right/
    │       │   └── Right/
    │       └── Night/
    │           ├── Back/
    │           ├── Front/
    │           ├── IR/
    │           ├── Left/
    │           ├── Lidar_Center/
    │           ├── Lidar_Left/
    │           ├── Lidar_Right/
    │           └── Right/
    └── TS_Snowy/
        └── Snowy/
            ├── Day/
            │   ├── Back/
            │   ├── Front/
            │   ├── IR/
            │   ├── Left/
            │   ├── Lidar_Center/
            │   ├── Lidar_Left/
            │   ├── Lidar_Right/
            │   └── Right/
            └── Night/
```
