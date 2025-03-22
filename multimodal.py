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

# Data Loading Functions
def load_pcd_data(pcd_path):
    """Load PCD file using Open3D."""
    try:
        return o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"Error loading PCD file {pcd_path}: {e}")
        return None

def load_json_data(json_path):
    """Load JSON label file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return None

def process_point_cloud(points, max_points=2048):
    """
    Process point cloud data efficiently.
    
    Args:
        points (np.ndarray): Point cloud data
        max_points (int): Maximum number of points to keep
        
    Returns:
        torch.Tensor: Processed point cloud data
    """
    if points is None or len(points) == 0:
        return torch.zeros((max_points, 3))
    
    # Efficient downsampling for large point clouds
    if len(points) > max_points:
        # Use uniform sampling instead of random for better stability
        step = len(points) // max_points
        indices = np.arange(0, len(points), step)[:max_points]
        sampled_points = points[indices]
    else:
        sampled_points = points
    
    # Pad if necessary
    if len(sampled_points) < max_points:
        padding = np.zeros((max_points - len(sampled_points), 3))
        sampled_points = np.vstack([sampled_points, padding])
    
    return torch.FloatTensor(sampled_points)

def get_single_timestamp_data(data_root, weather_condition, time_of_day, timestamp=None, seq_number=None):
    """
    Extract data for a single timestamp across all sensors.
    
    Args:
        data_root (str): Root directory of the dataset
        weather_condition (str): Weather condition ('Normal', 'Rainy', 'Snowy', 'Hazy')
        time_of_day (str): Time of day ('Day' or 'Night')
        timestamp (str, optional): Specific timestamp to match (e.g., '20220117')
        seq_number (str, optional): Specific sequence number to match (e.g., '048607')
    
    Returns:
        dict: Combined data from all sensors for the specific timestamp
    """
    # Define path prefixes
    label_prefix = os.path.join(data_root, 'labellingdata', f'TL_{weather_condition}', weather_condition)
    source_prefix = os.path.join(data_root, 'sourcedata', f'TS_{weather_condition}', weather_condition)
    
    # Camera and LiDAR types
    camera_types = ['Front', 'Back', 'Left', 'Right', 'IR']
    lidar_types = ['Lidar_Center', 'Lidar_Left', 'Lidar_Right']
    
    # Result data structure
    result = {
        'weather': weather_condition,
        'time_of_day': time_of_day,
        'timestamp': timestamp,
        'seq_number': seq_number,
        'cameras': {},
        'lidar': {}
    }
    
    # If no timestamp is provided, find first available sample
    if timestamp is None or seq_number is None:
        for camera_type in camera_types:
            image_dir = os.path.join(source_prefix, time_of_day, camera_type)
            if os.path.exists(image_dir):
                image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
                if image_files:
                    filename = os.path.basename(image_files[0])
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        timestamp = parts[2]  # Extract timestamp (e.g., 20220117)
                        seq_number = parts[-1].split('.')[0]  # Extract sequence number
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
        
        # Find image with matching timestamp and sequence number
        image_pattern = f"*_{timestamp}_{seq_number}.jpg"
        matching_images = glob.glob(os.path.join(camera_dir, image_pattern))
        
        if not matching_images:
            # Try alternative pattern
            image_pattern = f"*_{timestamp}*_{seq_number}.jpg"
            matching_images = glob.glob(os.path.join(camera_dir, image_pattern))
        
        if matching_images:
            image_path = matching_images[0]
            filename = os.path.basename(image_path)
            label_path = os.path.join(label_prefix, time_of_day, camera_type, filename.replace('.jpg', '.json'))
            
            camera_data = {'image_path': image_path}
            
            # Load image with error handling
            try:
                image = Image.open(image_path)
                camera_data['image'] = image
                camera_data['image_size'] = image.size
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
            
            # Load label if exists
            if os.path.exists(label_path):
                camera_label = load_json_data(label_path)
                if camera_label:
                    camera_data['label'] = camera_label
                    
                    # Count object types
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
        
        # Find LiDAR file with matching timestamp and sequence number
        lidar_pattern = f"*_{timestamp}_{seq_number}.pcd"
        matching_lidars = glob.glob(os.path.join(lidar_dir, lidar_pattern))
        
        if not matching_lidars:
            # Try alternative pattern
            lidar_pattern = f"*_{timestamp}*_{seq_number}.pcd"
            matching_lidars = glob.glob(os.path.join(lidar_dir, lidar_pattern))
        
        if matching_lidars:
            lidar_path = matching_lidars[0]
            filename = os.path.basename(lidar_path)
            label_path = os.path.join(label_prefix, time_of_day, lidar_type, filename + '.json')
            
            lidar_data = {'lidar_path': lidar_path}
            
            # Load point cloud with memory-efficient handling
            try:
                pcd_data = load_pcd_data(lidar_path)
                if pcd_data is not None:
                    points = np.asarray(pcd_data.points)
                    lidar_data['points'] = points
                    lidar_data['point_count'] = len(points)
            except Exception as e:
                print(f"Error loading PCD file {lidar_path}: {e}")
            
            # Load label if exists
            if os.path.exists(label_path):
                lidar_label = load_json_data(label_path)
                if lidar_label:
                    lidar_data['label'] = lidar_label
            
            result['lidar'][lidar_type] = lidar_data
    
    return result

# Model Components
class TNet(nn.Module):
    """T-Net for PointNet architecture (input and feature transformation)"""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Initialize weights/bias for last fc to produce identity transform
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))
        
    def forward(self, x):
        batch_size = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k*self.k).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        
        return x

class PointNetBackbone(nn.Module):
    """PointNet backbone for processing LiDAR point clouds"""
    def __init__(self, input_channels=3, output_channels=1024):
        super(PointNetBackbone, self).__init__()
        
        # Input transformation network
        self.input_transform = TNet(k=input_channels)
        
        # MLP for initial point feature extraction
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Feature transformation network
        self.feature_transform = TNet(k=64)
        
        # MLP for feature embedding
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
        
        # Final MLP to get the output dimension
        self.final_mlp = nn.Sequential(
            nn.Linear(1024, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: [batch_size, num_points, input_channels]
        # Transpose to [batch_size, input_channels, num_points] for Conv1d
        x = x.transpose(2, 1)
        
        # Input transformation
        trans = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        
        # First MLP
        x = self.mlp1(x)
        
        # Feature transformation
        trans_feat = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
        
        # Second MLP and max pooling
        x = self.mlp2(x)
        x = torch.max(x, 2, keepdim=False)[0]
        
        # Handle dimension for proper batch normalization
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Final MLP
        x = self.final_mlp(x)
        
        return x

class FPNBlock(nn.Module):
    """Feature Pyramid Network (FPN) block"""
    def __init__(self, in_channels, out_channels):
        super(FPNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class DetectionHead(nn.Module):
    """Object detection head using Feature Pyramid Network (FPN)"""
    def __init__(self, in_channels, num_classes, num_anchors=9):
        super(DetectionHead, self).__init__()
        
        # FPN blocks
        self.fpn_blocks = nn.ModuleList([
            FPNBlock(in_channels, 256),
            FPNBlock(256, 256),
            FPNBlock(256, 256)
        ])
        
        # Classification branch
        self.cls_branch = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, padding=1)
        )
        
        # Regression branch (bbox)
        self.reg_branch = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_anchors * 4, kernel_size=3, padding=1)  # 4 values for bbox (x, y, w, h)
        )
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
    def forward(self, x):
        fpn_features = []
        
        # Apply FPN blocks
        for block in self.fpn_blocks:
            x = block(x)
            fpn_features.append(x)
        
        # Use the final feature map for detection
        final_feature = fpn_features[-1]
        
        # Apply classification and regression branches
        cls_output = self.cls_branch(final_feature)
        reg_output = self.reg_branch(final_feature)
        
        # Reshape outputs
        batch_size, _, H, W = final_feature.shape
        cls_output = cls_output.permute(0, 2, 3, 1).reshape(batch_size, H, W, self.num_anchors, self.num_classes)
        reg_output = reg_output.permute(0, 2, 3, 1).reshape(batch_size, H, W, self.num_anchors, 4)
        
        return {
            'classification': cls_output,
            'regression': reg_output,
            'features': fpn_features
        }

class SegmentationHead(nn.Module):
    """Segmentation head for pixel-wise classification"""
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=1)
        
        # Upsampling layer for higher resolution output
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.upsample(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.upsample(x)
        x = self.conv3(x)
        
        return x

class WeatherRobustModel(nn.Module):
    """Multi-modal model for robust perception in adverse weather conditions"""
    def __init__(self, num_classes=10):
        super(WeatherRobustModel, self).__init__()
        
        # Image processing backbone (ResNet50)
        self.image_backbone = models.resnet50(pretrained=True)
        # Use all layers except the final fully connected layer
        self.image_features = nn.Sequential(*list(self.image_backbone.children())[:-2])
        
        # LiDAR processing backbone
        self.lidar_backbone = PointNetBackbone(input_channels=3, output_channels=1024)
        
        # Weather condition classifier
        self.weather_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)  # 4 weather conditions: Normal, Rainy, Snowy, Hazy
        )
        
        # Multi-modal fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(2048 + 1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        # Object detection head
        self.detection_head = DetectionHead(in_channels=1024, num_classes=num_classes)
        
        # Segmentation head
        self.segmentation_head = SegmentationHead(in_channels=1024, num_classes=num_classes)
    
    def forward(self, image, lidar):
        # Image feature extraction
        img_features = self.image_features(image)
        
        # LiDAR feature extraction
        lidar_features = self.lidar_backbone(lidar)
        
        # Weather classification from image features
        # Global average pooling to get a feature vector
        pooled_img_features = torch.mean(img_features, dim=[2, 3])
        weather_logits = self.weather_classifier(pooled_img_features)
        
        # Feature fusion
        # Reshape lidar features to match image feature dimensions
        batch_size, _, h, w = img_features.shape
        lidar_features_expanded = lidar_features.unsqueeze(-1).unsqueeze(-1)
        lidar_features_expanded = lidar_features_expanded.expand(-1, -1, h, w)
        
        # Concatenate image and lidar features
        fused_features = torch.cat([img_features, lidar_features_expanded], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # Object detection
        detection_output = self.detection_head(fused_features)
        
        # Segmentation
        segmentation_output = self.segmentation_head(fused_features)
        
        return {
            'detection': detection_output,
            'segmentation': segmentation_output,
            'weather': weather_logits
        }

# Dataset class for the multi-modal data
class WeatherRobustDataset(Dataset):
    def __init__(self, data_root, weather_conditions, times_of_day, transform=None):
        """
        Dataset for Weather Robust Model
        
        Args:
            data_root (str): Root directory of the dataset
            weather_conditions (list): List of weather conditions to include
            times_of_day (list): List of time of day to include
            transform (callable, optional): Optional transform to be applied to images
        """
        self.data_root = data_root
        self.weather_conditions = weather_conditions
        self.times_of_day = times_of_day
        self.transform = transform
        
        # List to store sample information (weather, time_of_day, timestamp, seq_number)
        self.samples = []
        
        # Scan dataset to find all available samples
        self._scan_dataset()
    
    def _scan_dataset(self):
        """Scan the dataset to find all available samples"""
        print("Scanning dataset for samples...")
        
        for weather in self.weather_conditions:
            for time_of_day in self.times_of_day:
                source_prefix = os.path.join(self.data_root, 'sourcedata', f'TS_{weather}', weather, time_of_day)
                
                # Skip if the directory doesn't exist
                if not os.path.exists(source_prefix):
                    continue
                
                # Use Front camera as reference to find all timestamps
                front_camera_dir = os.path.join(source_prefix, 'Front')
                if not os.path.exists(front_camera_dir):
                    continue
                
                image_files = glob.glob(os.path.join(front_camera_dir, '*.jpg'))
                
                # Use tqdm for progress display during scanning
                for img_path in tqdm(image_files, desc=f"Scanning {weather}/{time_of_day}"):
                    filename = os.path.basename(img_path)
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        timestamp = parts[2]  # Extract timestamp (e.g., 20220117)
                        seq_number = parts[-1].split('.')[0]  # Extract sequence number
                        
                        # Check if LiDAR data is also available
                        lidar_center_dir = os.path.join(source_prefix, 'Lidar_Center')
                        if not os.path.exists(lidar_center_dir):
                            continue
                            
                        lidar_pattern = f"*_{timestamp}_{seq_number}.pcd"
                        matching_lidars = glob.glob(os.path.join(lidar_center_dir, lidar_pattern))
                        
                        if matching_lidars:  # Only add sample if both camera and LiDAR data are available
                            self.samples.append({
                                'weather': weather,
                                'time_of_day': time_of_day,
                                'timestamp': timestamp,
                                'seq_number': seq_number
                            })
        
        print(f"Found {len(self.samples)} samples in the dataset")
    

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
    
        # Get all sensor data for this sample
        data = get_single_timestamp_data(
            data_root=self.data_root,
            weather_condition=sample_info['weather'],
            time_of_day=sample_info['time_of_day'],
            timestamp=sample_info['timestamp'],
            seq_number=sample_info['seq_number']
        )

        # Process front camera image with error handling
        front_camera_data = data['cameras'].get('Front', {})
        if 'image' in front_camera_data:
            try:
                image = front_camera_data['image']
                if self.transform:
                    image = self.transform(image)
                # Ensure consistent tensor size
                if image.shape[0] != 3 or image.shape[1] != 224 or image.shape[2] != 224:
                    image = torch.zeros((3, 224, 224))
            except Exception as e:
                print(f"Error processing image: {e}")
                # Create dummy image
                image = torch.zeros((3, 224, 224))
        else:
            # Create dummy image if not available
            image = torch.zeros((3, 224, 224))

        # Process center LiDAR data with improved point cloud handling
        center_lidar_data = data['lidar'].get('Lidar_Center', {})
        if 'points' in center_lidar_data:
            lidar_points = process_point_cloud(center_lidar_data['points'], max_points=2048)
        else:
            # Create dummy point cloud if not available
            lidar_points = torch.zeros((2048, 3))
    
        # Ensure lidar_points is exactly the right shape
        if lidar_points.shape != torch.Size([2048, 3]):
            lidar_points = torch.zeros((2048, 3))

        
        # Prepare weather condition label (one-hot encoding)
        weather_map = {'Normal': 0, 'Rainy': 1, 'Snowy': 2, 'Hazy': 3}
        weather_label = weather_map.get(sample_info['weather'], 0)
        
        # Extract object detection ground truth from Front camera
        detection_gt = []
        if 'label' in front_camera_data:
            for shape in front_camera_data['label'].get('shapes', []):
                label = shape.get('label')
                points = shape.get('points', [])
                if label and len(points) >= 2:
                    # Convert points to bounding box format [x_min, y_min, x_max, y_max]
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    detection_gt.append({
                        'label': label,
                        'bbox': bbox
                    })
        
        return {
            'image': image,
            'lidar': lidar_points,
            'weather': weather_label,
            'detection_gt': detection_gt,
            'sample_info': sample_info
        }

    def __len__(self):
        """데이터셋의 샘플 수를 반환합니다"""
        return len(self.samples)

# Training functions
def train_one_epoch(model, train_loader, criterion_weather, optimizer, epoch, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    weather_correct = 0
    total_samples = 0
    
    # Use tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for i, batch in enumerate(progress_bar):
        # Get data from batch
        images = batch['image'].to(device)
        lidar_points = batch['lidar'].to(device)
        weather_labels = batch['weather'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, lidar_points)
        
        # Calculate loss (for now, just weather classification)
        loss = criterion_weather(outputs['weather'], weather_labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item() * images.size(0)
        
        # Weather accuracy
        _, predicted = torch.max(outputs['weather'], 1)
        weather_correct += (predicted == weather_labels).sum().item()
        total_samples += weather_labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / total_samples,
            'weather_acc': 100 * weather_correct / total_samples
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * weather_correct / len(train_loader.dataset)
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion_weather, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    weather_correct = 0
    weather_preds = []
    weather_true = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            # Get data from batch
            images = batch['image'].to(device)
            lidar_points = batch['lidar'].to(device)
            weather_labels = batch['weather'].to(device)
            
            # Forward pass
            outputs = model(images, lidar_points)
            
            # Calculate loss
            loss = criterion_weather(outputs['weather'], weather_labels)
            val_loss += loss.item() * images.size(0)
            
            # Weather accuracy
            _, predicted = torch.max(outputs['weather'], 1)
            weather_correct += (predicted == weather_labels).sum().item()
            
            # Store predictions for confusion matrix
            weather_preds.extend(predicted.cpu().numpy())
            weather_true.extend(weather_labels.cpu().numpy())
    
    # Calculate metrics
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100 * weather_correct / len(val_loader.dataset)
    
    return val_loss, val_acc, weather_true, weather_preds

# Model saving and loading functions
def save_checkpoint(model, optimizer, epoch, val_acc, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        val_acc = checkpoint['val_acc']
        print(f"Loaded checkpoint from epoch {epoch} with validation accuracy {val_acc:.2f}%")
        return epoch, val_acc
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, 0

# Function to test the model on a single sample
def test_model_with_sample(model, data_root, weather_condition, time_of_day, timestamp=None, seq_number=None, device=None):
    """Test the model with a single sample"""
    if device is None:
        device = next(model.parameters()).device
        
    print("\n===== Testing model with a sample =====")
    
    # Get sample data
    sample = get_single_timestamp_data(
        data_root=data_root,
        weather_condition=weather_condition,
        time_of_day=time_of_day,
        timestamp=timestamp,
        seq_number=seq_number
    )
    
    # Print sample information
    print(f"Sample information: Weather={sample['weather']}, Time={sample['time_of_day']}")
    print(f"Timestamp: {sample['timestamp']}, Sequence: {sample['seq_number']}")
    
    # Process front camera image
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
        print("No front camera image available")
        image_tensor = torch.zeros((1, 3, 224, 224)).to(device)
    
    # Process LiDAR data
    center_lidar_data = sample['lidar'].get('Lidar_Center', {})
    if 'points' in center_lidar_data:
        lidar_points = process_point_cloud(center_lidar_data['points'], max_points=2048)
        lidar_tensor = lidar_points.unsqueeze(0).to(device)
    else:
        print("No LiDAR data available")
        lidar_tensor = torch.zeros((1, 2048, 3)).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(image_tensor, lidar_tensor)
    
    # Process weather prediction
    weather_probs = F.softmax(outputs['weather'], dim=1)
    weather_pred = torch.argmax(weather_probs, dim=1).item()
    weather_names = ['Normal', 'Rainy', 'Snowy', 'Hazy']
    print(f"\nWeather Prediction: {weather_names[weather_pred]}")
    print("Weather Probabilities:")
    for i, name in enumerate(weather_names):
        print(f"  {name}: {weather_probs[0][i].item():.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 8))
    
    # Plot front camera image
    plt.subplot(2, 2, 1)
    if 'image' in front_camera_data:
        plt.imshow(front_camera_data['image'])
        plt.title(f"Front Camera - Pred: {weather_names[weather_pred]}")
    else:
        plt.text(0.5, 0.5, "No image available", ha='center', va='center')
    plt.axis('off')
    
    # Plot LiDAR point cloud
    plt.subplot(2, 2, 2)
    if 'points' in center_lidar_data:
        points = center_lidar_data['points']
        plt.scatter(points[:, 0], points[:, 1], s=1, c=points[:, 2], cmap='viridis')
        plt.title("LiDAR Point Cloud (Top View)")
        plt.axis('equal')
    else:
        plt.text(0.5, 0.5, "No LiDAR data available", ha='center', va='center')
    
    # Plot weather classification probabilities
    plt.subplot(2, 2, 3)
    bars = plt.bar(weather_names, weather_probs[0].cpu().numpy())
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title("Weather Classification")
    
    # Highlight the predicted class
    bars[weather_pred].set_color('red')
    
    # Plot detection/segmentation visualization placeholder
    plt.subplot(2, 2, 4)
    if 'image' in front_camera_data:
        # Get detection outputs
        detection_output = outputs['detection']
        
        # For simplicity, just show the feature map
        feature_map = detection_output['features'][-1][0]
        
        # Take the mean across channels to visualize
        feature_vis = torch.mean(feature_map, dim=0).cpu().numpy()
        
        # Normalize for visualization
        feature_vis = (feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min() + 1e-8)
        
        plt.imshow(feature_vis, cmap='hot')
        plt.title("Feature Map")
        plt.axis('off')
    else:
        plt.text(0.5, 0.5, "No visualization available", ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f"test_result_{sample['weather']}_{sample['time_of_day']}.png")
    plt.show()
    
    return outputs

# Main function for model training
def train_model(data_root, output_dir, epochs=10, batch_size=8, learning_rate=0.001):
    """Train the weather robust perception model"""
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset splits
    all_weather = ['Normal', 'Rainy', 'Snowy', 'Hazy']
    all_times = ['Day', 'Night']
    
    # Create full dataset
    full_dataset = WeatherRobustDataset(
        data_root=data_root,
        weather_conditions=all_weather,
        times_of_day=all_times,
        transform=transform
    )
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = WeatherRobustModel(num_classes=10).to(device)
    
    # Define loss functions and optimizer
    criterion_weather = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    # Checkpoint path
    checkpoint_path = os.path.join(output_dir, 'best_model.pth')
    
    # Load checkpoint if exists
    start_epoch = 0
    best_val_acc = 0
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_acc = load_checkpoint(model, optimizer, checkpoint_path)
    
    # Initialize metrics tracking
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion_weather=criterion_weather,
            optimizer=optimizer,
            epoch=epoch,
            device=device
        )
        
        # Validate
        val_loss, val_acc, weather_true, weather_preds = validate(
            model=model,
            val_loader=val_loader,
            criterion_weather=criterion_weather,
            device=device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Track metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{epochs-1}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save model if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            
            # Generate confusion matrix at best checkpoint
            cm = confusion_matrix(weather_true, weather_preds)
            weather_names = ['Normal', 'Rainy', 'Snowy', 'Hazy']
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(weather_names))
            plt.xticks(tick_marks, weather_names, rotation=45)
            plt.yticks(tick_marks, weather_names)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_epoch_{epoch}.png'))
            
            # Save classification report
            report = classification_report(weather_true, weather_preds, target_names=weather_names)
            with open(os.path.join(output_dir, f'classification_report_epoch_{epoch}.txt'), 'w') as f:
                f.write(report)
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    
    # Load best model
    load_checkpoint(model, None, checkpoint_path)
    
    return model

if __name__ == "__main__":
    # Set data root path and output directory
    data_root = r"C:\Users\dadab\Desktop\project data\traindata"
    output_dir = r"C:\Users\dadab\Desktop\project code\weather_robust_model_output"
    
    # Train model
    model = train_model(
        data_root=data_root,
        output_dir=output_dir,
        epochs=20,
        batch_size=8,
        learning_rate=0.001
    )
    
    # Test with a sample
    test_model_with_sample(
        model=model,
        data_root=data_root,
        weather_condition="Rainy",
        time_of_day="Day"
    )
