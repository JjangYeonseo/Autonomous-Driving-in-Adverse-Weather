import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import glob
import open3d as o3d

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

class WeatherRobustDataset(Dataset):
    def __init__(self, data_root, weather_condition, time_of_day=None, sensor_types=None, transform=None, test_single=False):
        """
        Dataset for weather-robust perception.
        
        Args:
            data_root (str): Root directory of the dataset
            weather_condition (str): Weather condition ('Normal', 'Rainy', 'Snowy', 'Hazy')
            time_of_day (str, optional): Time of day filter ('Day' or 'Night')
            sensor_types (list, optional): List of sensor types to include 
                                         (e.g., ['Front', 'Back', 'Lidar_Center'])
            transform: Optional transforms to apply to images
            test_single (bool): If True, only load a single sample for testing
        """
        self.data_root = data_root
        self.weather_condition = weather_condition
        self.time_of_day = time_of_day
        self.sensor_types = sensor_types
        self.transform = transform
        self.test_single = test_single
        
        # Define path prefixes
        self.label_prefix = os.path.join(data_root, 'labellingdata', f'TL_{weather_condition}', weather_condition)
        self.source_prefix = os.path.join(data_root, 'sourcedata', f'TS_{weather_condition}', weather_condition)
        
        # Create the data index
        self.samples = self._create_index()
        
    def _get_file_base_info(self, file_path):
        """Extract base information from a file path to help with matching."""
        filename = os.path.basename(file_path)
        timestamp_part = '_'.join(filename.split('_')[2:3])  # Extract date part (e.g., 20220117)
        seq_number = filename.split('_')[-1].split('.')[0]   # Extract sequence number (e.g., 048607)
        return {
            'timestamp': timestamp_part,
            'seq_number': seq_number,
            'full_name': filename
        }
    
    def _find_matching_files(self, base_path, time_of_day, source_type, file_ext):
        """Find all files of a certain type in the specified directory."""
        search_path = os.path.join(base_path, time_of_day, source_type)
        if not os.path.exists(search_path):
            return []
        
        return glob.glob(os.path.join(search_path, f'*.{file_ext}'))
    
    def _create_index(self):
        """Create an index matching image, LiDAR, and label files."""
        samples = []
        
        # Define the types of data to process
        time_periods = ['Day', 'Night'] if self.time_of_day is None else [self.time_of_day]
        
        # Camera directions and corresponding file prefixes
        camera_types = ['Front', 'Back', 'Left', 'Right', 'IR']
        camera_prefixes = {'Front': 'CF', 'Back': 'CB', 'Left': 'CL', 'Right': 'CR', 'IR': 'CI'}
        
        # LiDAR types and corresponding file prefixes
        lidar_types = ['Lidar_Center', 'Lidar_Left', 'Lidar_Right']
        lidar_prefixes = {'Lidar_Center': 'LC', 'Lidar_Left': 'LL', 'Lidar_Right': 'LR'}
        
        # Filter sensor types if specified
        sensor_types = []
        if self.sensor_types:
            for s_type in self.sensor_types:
                if s_type in camera_types or s_type in lidar_types:
                    sensor_types.append(s_type)
        else:
            sensor_types = camera_types + lidar_types
        
        # Process each time period
        for time_of_day in time_periods:
            # Process camera data
            for camera_type in [ct for ct in camera_types if ct in sensor_types]:
                # Find all image files for this camera type
                image_files = self._find_matching_files(
                    self.source_prefix, time_of_day, camera_type, 'jpg')
                
                for image_path in image_files:
                    # Construct the expected label path
                    filename = os.path.basename(image_path)
                    label_path = os.path.join(
                        self.label_prefix, time_of_day, camera_type, 
                        filename.replace('.jpg', '.json'))
                    
                    # Check if label exists
                    if not os.path.exists(label_path):
                        continue
                        
                    # Extract timestamp info for matching with LiDAR
                    file_info = self._get_file_base_info(image_path)
                    timestamp = file_info['timestamp']
                    seq_number = file_info['seq_number']
                    
                    # Find matching LiDAR files (if needed)
                    lidar_matches = {}
                    for lidar_type in [lt for lt in lidar_types if lt in sensor_types]:
                        prefix = lidar_prefixes[lidar_type]
                        # Construct expected LiDAR filename pattern
                        # The pattern matches the timestamp and sequence number
                        lidar_pattern = f"*{prefix}*{timestamp}*{seq_number}.pcd"
                        lidar_search_path = os.path.join(
                            self.source_prefix, time_of_day, lidar_type, lidar_pattern)
                        
                        matching_lidar_files = glob.glob(lidar_search_path)
                        if matching_lidar_files:
                            lidar_path = matching_lidar_files[0]
                            # Find corresponding LiDAR label
                            lidar_label_path = os.path.join(
                                self.label_prefix, time_of_day, lidar_type,
                                os.path.basename(lidar_path) + '.json')
                            
                            if os.path.exists(lidar_label_path):
                                lidar_matches[lidar_type] = {
                                    'lidar_path': lidar_path,
                                    'lidar_label_path': lidar_label_path
                                }
                    
                    # Create a sample entry
                    sample = {
                        'image_path': image_path,
                        'label_path': label_path,
                        'camera_type': camera_type,
                        'time_of_day': time_of_day,
                        'lidar_matches': lidar_matches,
                        'weather': self.weather_condition
                    }
                    
                    samples.append(sample)
                    
                    # If testing with a single sample, return after finding one
                    if self.test_single and samples:
                        return samples[:1]
            
            # Process LiDAR-only data (if no matching images are required)
            if not samples or (self.test_single and not samples):
                for lidar_type in [lt for lt in lidar_types if lt in sensor_types]:
                    lidar_files = self._find_matching_files(
                        self.source_prefix, time_of_day, lidar_type, 'pcd')
                    
                    for lidar_path in lidar_files:
                        # Construct the expected label path
                        filename = os.path.basename(lidar_path)
                        lidar_label_path = os.path.join(
                            self.label_prefix, time_of_day, lidar_type, 
                            filename + '.json')
                        
                        # Check if label exists
                        if not os.path.exists(lidar_label_path):
                            continue
                        
                        # Create a sample entry
                        sample = {
                            'image_path': None,
                            'label_path': None,
                            'camera_type': None,
                            'time_of_day': time_of_day,
                            'lidar_matches': {
                                lidar_type: {
                                    'lidar_path': lidar_path,
                                    'lidar_label_path': lidar_label_path
                                }
                            },
                            'weather': self.weather_condition
                        }
                        
                        samples.append(sample)
                        
                        # If testing with a single sample, return after finding one
                        if self.test_single and samples:
                            return samples[:1]
        
        if not samples:
            print(f"Warning: No matching samples found for {self.weather_condition}, {self.time_of_day}, {self.sensor_types}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data_dict = {'weather': sample['weather'], 'time_of_day': sample['time_of_day']}
        
        # Load image and camera label if available
        if sample['image_path'] and os.path.exists(sample['image_path']):
            try:
                image = Image.open(sample['image_path'])
                if self.transform:
                    image = self.transform(image)
                data_dict['image'] = image
                data_dict['camera_type'] = sample['camera_type']
                
                # Load camera label
                if sample['label_path'] and os.path.exists(sample['label_path']):
                    camera_label = load_json_data(sample['label_path'])
                    data_dict['camera_label'] = camera_label
                    
                    # Count object types in camera label
                    object_counts = {}
                    for obj in camera_label.get('shapes', []):
                        label = obj.get('label')
                        if label:
                            object_counts[label] = object_counts.get(label, 0) + 1
                    data_dict['object_counts'] = object_counts
            except Exception as e:
                print(f"Error loading image or label: {e}")
        
        # Load LiDAR data and labels
        lidar_data = {}
        for lidar_type, lidar_info in sample['lidar_matches'].items():
            try:
                if os.path.exists(lidar_info['lidar_path']):
                    pcd_data = load_pcd_data(lidar_info['lidar_path'])
                    if pcd_data is not None:
                        lidar_array = np.asarray(pcd_data.points)
                        lidar_tensor = torch.from_numpy(lidar_array).float()
                        lidar_data[lidar_type] = {
                            'data': lidar_tensor
                        }
                        
                        # Load LiDAR label
                        if os.path.exists(lidar_info['lidar_label_path']):
                            lidar_label = load_json_data(lidar_info['lidar_label_path'])
                            lidar_data[lidar_type]['label'] = lidar_label
            except Exception as e:
                print(f"Error loading LiDAR data or label: {e}")
        
        if lidar_data:
            data_dict['lidar'] = lidar_data
            
        return data_dict


# Example usage
if __name__ == "__main__":
    data_root = r"C:\Users\dadab\Desktop\project data\traindata"
    
    # Test with a single sample first
    test_dataset = WeatherRobustDataset(
        data_root=data_root,
        weather_condition='Snowy',
        time_of_day='Day',
        sensor_types=['Back', 'Lidar_Center'],
        test_single=True
    )
    
    print(f"Found {len(test_dataset)} samples")
    
    if len(test_dataset) > 0:
        # Get the first sample
        sample = test_dataset[0]
        print("\nSample contents:")
        for key, value in sample.items():
            if key == 'image':
                print(f"Image shape: {value.size}")
            elif key == 'lidar':
                for lidar_type, lidar_data in value.items():
                    print(f"LiDAR {lidar_type} shape: {lidar_data['data'].shape}")
            elif key == 'object_counts':
                print(f"Object counts: {value}")
            else:
                print(f"{key}: {value}")
        
        print("\nTest successful! Now you can process all data.")
    else:
        print("No samples found. Check paths and data structure.")
