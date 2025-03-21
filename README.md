# Weather-Robust Autonomous Driving Model

This project aims to develop a deep learning model that allows autonomous vehicles to drive robustly under various weather conditions (e.g., rain, snow, fog). The model is trained on a dataset that includes images and LiDAR point cloud data, helping the model learn to detect objects and drive safely in adverse weather scenarios.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Structure](#data-structure)
- [Goal](#goal)
- [Model Architecture](#model-architecture)
- [Usage](#usage)

---

## Project Overview

The goal of this project is to create a model that helps autonomous driving systems perform robustly even in bad weather conditions. The dataset contains images from various weather scenarios as well as LiDAR data, which will be used to train the model to identify objects and navigate safely through different weather conditions.

## Data Structure

The data is organized into two main categories:
1. **Labeling Data**: JSON files containing object and segmentation information for each image in different weather conditions.
2. **Raw Data**: Images in JPG format and LiDAR data in PCD format.

### Training Data Structure

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


## Goal

The objective of this project is to create a model that enables autonomous vehicles to maintain robust performance under various weather conditions. The model aims to:
- **Object Detection**: Detect road objects accurately even in challenging weather conditions.
- **Segmentation**: Perform pixel-level segmentation of objects.
- **Robustness**: Ensure the model performs well across different weather conditions like rain, snow, and fog.

## Model Architecture

This project utilizes Mask R-CNN for object detection and segmentation, as well as PointNet or PointNet++ for processing LiDAR data. An additional weather transformation network is implemented to ensure robustness across different weather conditions.

## Model Flow:

The model takes images and LiDAR data as input and performs object detection and segmentation.
A weather adaptation layer ensures that the model remains robust when exposed to different weather conditions.

## Usage
### Training

1. Prepare the dataset and start training:
   python train.py --dataset <path_to_dataset> --epochs 50 --batch_size 4
### Inference

1. Run inference with the trained model:
   python predict.py --image <path_to_image> --model <path_to_trained_model>
