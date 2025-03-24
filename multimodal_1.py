import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import glob
import open3d as o3d
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_pcd_data(pcd_path):
    try:
        return o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"Error loading PCD file {pcd_path}: {e}")
        return None

def load_json_data(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return None

def process_point_cloud(points, max_points=2048):
    if points is None or len(points) == 0:
        return torch.zeros((max_points, 3))
    if len(points) > max_points:
        step = len(points) // max_points
        indices = np.arange(0, len(points), step)[:max_points]
        sampled_points = points[indices]
    else:
        sampled_points = points
    if len(sampled_points) < max_points:
        padding = np.zeros((max_points - len(sampled_points), 3))
        sampled_points = np.vstack([sampled_points, padding])
    return torch.FloatTensor(sampled_points)

def get_single_timestamp_data(data_root, weather_condition, time_of_day, timestamp=None, seq_number=None):
    label_prefix = os.path.join(data_root, 'labellingdata', f'TL_{weather_condition}', weather_condition)
    source_prefix = os.path.join(data_root, 'sourcedata', f'TS_{weather_condition}', weather_condition)
    camera_types = ['Front', 'Back', 'Left', 'Right', 'IR']
    lidar_types = ['Lidar_Center', 'Lidar_Left', 'Lidar_Right']

    result = {
        'weather': weather_condition,
        'time_of_day': time_of_day,
        'timestamp': timestamp,
        'seq_number': seq_number,
        'cameras': {},
        'lidar': {}
    }

    if timestamp is None or seq_number is None:
        for camera_type in camera_types:
            image_dir = os.path.join(source_prefix, time_of_day, camera_type)
            if os.path.exists(image_dir):
                image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
                if image_files:
                    filename = os.path.basename(image_files[0])
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        timestamp = parts[2]
                        seq_number = parts[-1].split('.')[0]
                        result['timestamp'] = timestamp
                        result['seq_number'] = seq_number
                        print(f"Using first found timestamp: {timestamp}, sequence: {seq_number}")
                        break

    if not timestamp or not seq_number:
        print("No timestamp and sequence number found or provided.")
        return result

    # Process camera data
    for camera_type in camera_types:
        camera_dir = os.path.join(source_prefix, time_of_day, camera_type)
        if not os.path.exists(camera_dir):
            continue

        image_pattern = f"*_{timestamp}_{seq_number}.jpg"
        matching_images = glob.glob(os.path.join(camera_dir, image_pattern))

        if not matching_images:
            image_pattern = f"*_{timestamp}*_{seq_number}.jpg"
            matching_images = glob.glob(os.path.join(camera_dir, image_pattern))

        if matching_images:
            image_path = matching_images[0]
            filename = os.path.basename(image_path)
            label_path = os.path.join(label_prefix, time_of_day, camera_type, filename.replace('.jpg', '.json'))

            camera_data = {'image_path': image_path}
            try:
                image = Image.open(image_path)
                camera_data['image'] = image
                camera_data['image_size'] = image.size
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

            if os.path.exists(label_path):
                camera_label = load_json_data(label_path)
                if camera_label:
                    camera_data['label'] = camera_label
                    object_counts = {}
                    for obj in camera_label.get('shapes', []):
                        label = obj.get('label')
                        if label:
                            object_counts[label] = object_counts.get(label, 0) + 1
                    camera_data['object_counts'] = object_counts

            result['cameras'][camera_type] = camera_data

    # Process LiDAR data
    for lidar_type in lidar_types:
        lidar_dir = os.path.join(source_prefix, time_of_day, lidar_type)
        if not os.path.exists(lidar_dir):
            continue

        lidar_pattern = f"*_{timestamp}_{seq_number}.pcd"
        matching_lidars = glob.glob(os.path.join(lidar_dir, lidar_pattern))

        if not matching_lidars:
            lidar_pattern = f"*_{timestamp}*_{seq_number}.pcd"
            matching_lidars = glob.glob(os.path.join(lidar_dir, lidar_pattern))

        if matching_lidars:
            lidar_path = matching_lidars[0]
            filename = os.path.basename(lidar_path)
            label_path = os.path.join(label_prefix, time_of_day, lidar_type, filename + '.json')

            lidar_data = {'lidar_path': lidar_path}
            try:
                pcd_data = load_pcd_data(lidar_path)
                if pcd_data is not None:
                    points = np.asarray(pcd_data.points)
                    lidar_data['points'] = points
                    lidar_data['point_count'] = len(points)
            except Exception as e:
                print(f"Error loading PCD file {lidar_path}: {e}")

            if os.path.exists(label_path):
                lidar_label = load_json_data(label_path)
                if lidar_label:
                    lidar_data['label'] = lidar_label

            result['lidar'][lidar_type] = lidar_data

    return result

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetBackbone(nn.Module):
    def __init__(self, input_channels=3, output_channels=1024):
        super(PointNetBackbone, self).__init__()

        self.input_transform = TNet(k=input_channels)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.feature_transform = TNet(k=64)

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(1024, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.transpose(2, 1)

        trans = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)

        x = self.mlp1(x)

        trans_feat = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)

        x = self.mlp2(x)
        x = torch.max(x, 2, keepdim=False)[0]

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.final_mlp(x)
        return x

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=9):
        super(DetectionHead, self).__init__()

        self.fpn_blocks = nn.ModuleList([
            FPNBlock(in_channels, 256),
            FPNBlock(256, 256),
            FPNBlock(256, 256)
        ])

        self.cls_branch = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, padding=1)
        )

        self.reg_branch = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_anchors * 4, kernel_size=3, padding=1)  
        )

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        fpn_features = []
        for block in self.fpn_blocks:
            x = block(x)
            fpn_features.append(x)

        final_feature = fpn_features[-1]

        cls_output = self.cls_branch(final_feature)
        reg_output = self.reg_branch(final_feature)

        batch_size, _, H, W = final_feature.shape
        cls_output = cls_output.permute(0, 2, 3, 1).reshape(batch_size, H, W, self.num_anchors, self.num_classes)
        reg_output = reg_output.permute(0, 2, 3, 1).reshape(batch_size, H, W, self.num_anchors, 4)

        return {
            'classification': cls_output,
            'regression': reg_output,
            'features': fpn_features
        }


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.upsample(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.upsample(x)
        x = self.conv3(x)
        return x

class WeatherRobustModel(nn.Module):
    def __init__(self, num_classes=10):
        super(WeatherRobustModel, self).__init__()

        self.image_backbone = models.resnet50(pretrained=True)
        self.image_features = nn.Sequential(*list(self.image_backbone.children())[:-2])

        self.lidar_backbone = PointNetBackbone(input_channels=3, output_channels=1024)

        self.weather_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)  
        )

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(2048 + 1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.detection_head = DetectionHead(in_channels=1024, num_classes=num_classes)
        self.segmentation_head = SegmentationHead(in_channels=1024, num_classes=num_classes)

    def forward(self, image, lidar):
        img_features = self.image_features(image)

        lidar_features = self.lidar_backbone(lidar)

        pooled_img_features = torch.mean(img_features, dim=[2, 3])
        weather_logits = self.weather_classifier(pooled_img_features)

        batch_size, _, h, w = img_features.shape
        lidar_features_expanded = lidar_features.unsqueeze(-1).unsqueeze(-1)
        lidar_features_expanded = lidar_features_expanded.expand(-1, -1, h, w)

        fused_features = torch.cat([img_features, lidar_features_expanded], dim=1)
        fused_features = self.fusion_layer(fused_features)

        detection_output = self.detection_head(fused_features)
        segmentation_output = self.segmentation_head(fused_features)

        return {
            'detection': detection_output,
            'segmentation': segmentation_output,
            'weather': weather_logits
        }

# 사용자 정의 collate_fn 추가
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    lidar = torch.stack([item['lidar'] for item in batch])
    weather = torch.tensor([item['weather'] for item in batch])
    detection_gt = [item['detection_gt'] for item in batch]
    sample_info = [item['sample_info'] for item in batch]
    return {
        'image': images,
        'lidar': lidar,
        'weather': weather,
        'detection_gt': detection_gt,
        'sample_info': sample_info
    }

# Dataset 클래스 정의
class WeatherRobustDataset(Dataset):
    def __init__(self, data_root, weather_conditions, times_of_day, transform=None):
        self.data_root = data_root
        self.weather_conditions = weather_conditions
        self.times_of_day = times_of_day
        self.transform = transform
        self.samples = []
        self._scan_dataset()

    def _scan_dataset(self):
        print("Scanning dataset for samples...")
        for weather in self.weather_conditions:
            for time_of_day in self.times_of_day:
                source_prefix = os.path.join(self.data_root, 'sourcedata', f'TS_{weather}', weather, time_of_day)
                if not os.path.exists(source_prefix):
                    continue
                front_camera_dir = os.path.join(source_prefix, 'Front')
                if not os.path.exists(front_camera_dir):
                    continue
                image_files = glob.glob(os.path.join(front_camera_dir, '*.jpg'))
                for img_path in tqdm(image_files, desc=f"Scanning {weather}/{time_of_day}"):
                    filename = os.path.basename(img_path)
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        timestamp = parts[2]
                        seq_number = parts[-1].split('.')[0]
                        lidar_center_dir = os.path.join(source_prefix, 'Lidar_Center')
                        if not os.path.exists(lidar_center_dir):
                            continue
                        lidar_pattern = f"*_{timestamp}_{seq_number}.pcd"
                        matching_lidars = glob.glob(os.path.join(lidar_center_dir, lidar_pattern))
                        if matching_lidars:
                            self.samples.append({
                                'weather': weather,
                                'time_of_day': time_of_day,
                                'timestamp': timestamp,
                                'seq_number': seq_number
                            })
        print(f"Found {len(self.samples)} samples in the dataset")

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        data = get_single_timestamp_data(
            data_root=self.data_root,
            weather_condition=sample_info['weather'],
            time_of_day=sample_info['time_of_day'],
            timestamp=sample_info['timestamp'],
            seq_number=sample_info['seq_number']
        )
        front_camera_data = data['cameras'].get('Front', {})
        if 'image' in front_camera_data:
            try:
                image = front_camera_data['image']
                if self.transform:
                    image = self.transform(image)
                if image.shape[0] != 3 or image.shape[1] != 224 or image.shape[2] != 224:
                    image = torch.zeros((3, 224, 224))
            except:
                image = torch.zeros((3, 224, 224))
        else:
            image = torch.zeros((3, 224, 224))
        center_lidar_data = data['lidar'].get('Lidar_Center', {})
        if 'points' in center_lidar_data:
            lidar_points = process_point_cloud(center_lidar_data['points'], max_points=2048)
        else:
            lidar_points = torch.zeros((2048, 3))
        if lidar_points.shape != torch.Size([2048, 3]):
            lidar_points = torch.zeros((2048, 3))
        weather_map = {'Normal': 0, 'Rainy': 1, 'Snowy': 2, 'Hazy': 3}
        weather_label = weather_map.get(sample_info['weather'], 0)
        detection_gt = []
        if 'label' in front_camera_data:
            for shape in front_camera_data['label'].get('shapes', []):
                label = shape.get('label')
                points = shape.get('points', [])
                if label and len(points) >= 2:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    detection_gt.append({'label': label, 'bbox': bbox})
        return {
            'image': image,
            'lidar': lidar_points,
            'weather': weather_label,
            'detection_gt': detection_gt,
            'sample_info': sample_info
        }

    def __len__(self):
        return len(self.samples)

def train_model(data_root, output_dir, epochs=10, batch_size=8, learning_rate=0.001):
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    all_weather = ['Normal', 'Rainy', 'Snowy', 'Hazy']
    all_times = ['Day', 'Night']
    full_dataset = WeatherRobustDataset(data_root, all_weather, all_times, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    model = WeatherRobustModel(num_classes=10).to(device)
    criterion_weather = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    # ... train loop 생략 가능 (이전 제공 코드 참고)
    return model



def test_model_with_sample(model, data_root, weather_condition, time_of_day, timestamp=None, seq_number=None, device=None):
    if device is None:
        device = next(model.parameters()).device

    print("\n===== Testing model with a sample =====")

    sample = get_single_timestamp_data(
        data_root=data_root,
        weather_condition=weather_condition,
        time_of_day=time_of_day,
        timestamp=timestamp,
        seq_number=seq_number
    )

    print(f"Sample info: {sample['weather']} / {sample['time_of_day']}")
    print(f"Timestamp: {sample['timestamp']} / Seq: {sample['seq_number']}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    front_camera_data = sample['cameras'].get('Front', {})
    if 'image' in front_camera_data:
        image = front_camera_data['image']
        image_tensor = transform(image).unsqueeze(0).to(device)
    else:
        print("No front camera image available.")
        image_tensor = torch.zeros((1, 3, 224, 224)).to(device)

    center_lidar_data = sample['lidar'].get('Lidar_Center', {})
    if 'points' in center_lidar_data:
        lidar_points = process_point_cloud(center_lidar_data['points'], max_points=2048)
        lidar_tensor = lidar_points.unsqueeze(0).to(device)
    else:
        print("No lidar data available.")
        lidar_tensor = torch.zeros((1, 2048, 3)).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor, lidar_tensor)

    weather_probs = F.softmax(outputs['weather'], dim=1)
    weather_pred = torch.argmax(weather_probs, dim=1).item()
    weather_names = ['Normal', 'Rainy', 'Snowy', 'Hazy']

    print(f"\nPredicted Weather: {weather_names[weather_pred]}")
    print("Weather Probabilities:")
    for i, name in enumerate(weather_names):
        print(f"  {name}: {weather_probs[0][i].item():.4f}")

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    if 'image' in front_camera_data:
        plt.imshow(front_camera_data['image'])
        plt.title(f"Front Camera - Predicted: {weather_names[weather_pred]}")
    else:
        plt.text(0.5, 0.5, "No image available", ha='center', va='center')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    if 'points' in center_lidar_data:
        points = center_lidar_data['points']
        plt.scatter(points[:, 0], points[:, 1], s=1, c=points[:, 2], cmap='viridis')
        plt.title("LiDAR Point Cloud (Top View)")
        plt.axis('equal')
    else:
        plt.text(0.5, 0.5, "No LiDAR available", ha='center', va='center')

    plt.subplot(2, 2, 3)
    bars = plt.bar(weather_names, weather_probs[0].cpu().numpy())
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title("Weather Prediction Probabilities")
    bars[weather_pred].set_color('red')

    plt.subplot(2, 2, 4)
    if 'image' in front_camera_data and 'features' in outputs['detection']:
        feature_map = outputs['detection']['features'][-1][0]
        feature_vis = torch.mean(feature_map, dim=0).cpu().numpy()
        feature_vis = (feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min() + 1e-8)
        plt.imshow(feature_vis, cmap='hot')
        plt.title("Detection Feature Map")
        plt.axis('off')
    else:
        plt.text(0.5, 0.5, "No feature map", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(f"test_result_{sample['weather']}_{sample['time_of_day']}.png")
    plt.show()

    return outputs

if __name__ == "__main__":
    data_root = r"C:\\Users\\dadab\\Desktop\\project data\\traindata"
    output_dir = r"C:\\Users\\dadab\\Desktop\\project code\\weather_robust_model_output"
    
    model = train_model(
        data_root=data_root,
        output_dir=output_dir,
        epochs=20,
        batch_size=8,
        learning_rate=0.001
    )

    test_model_with_sample(
        model=model,
        data_root=data_root,
        weather_condition="Rainy",
        time_of_day="Day"
    )
