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
            
            # Load image
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
            
            # Load point cloud
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

def print_sample_summary(sample):
    """Print a summary of the sample data."""
    print("\n===== 샘플 정보 =====")
    print(f"날씨 조건: {sample['weather']}")
    print(f"시간대: {sample['time_of_day']}")
    print(f"타임스탬프: {sample['timestamp']}")
    print(f"시퀀스 번호: {sample['seq_number']}")
    
    print("\n===== 카메라 정보 =====")
    for camera_type, camera_data in sample['cameras'].items():
        print(f"\n[{camera_type} 카메라]")
        if 'image_size' in camera_data:
            print(f"  이미지 크기: {camera_data['image_size']}")
        if 'object_counts' in camera_data:
            print(f"  감지된 객체:")
            for obj_type, count in camera_data['object_counts'].items():
                print(f"    - {obj_type}: {count}개")
    
    print("\n===== 라이다 정보 =====")
    for lidar_type, lidar_data in sample['lidar'].items():
        print(f"\n[{lidar_type}]")
        if 'point_count' in lidar_data:
            print(f"  포인트 클라우드 크기: {lidar_data['point_count']} 포인트")
            if 'points' in lidar_data:
                print(f"  포인트 클라우드 형태: {lidar_data['points'].shape}")

# Example usage
if __name__ == "__main__":
    data_root = r"C:\Users\dadab\Desktop\project data\traindata"
    
    # 기존 예제에서 사용된 것과 동일한 타임스탬프와 시퀀스 번호 사용
    sample = get_single_timestamp_data(
        data_root=data_root,
        weather_condition='Snowy',
        time_of_day='Day',
        timestamp='20220117',  # 예제에서 추출한 타임스탬프
        seq_number='048607'    # 예제에서 추출한 시퀀스 번호
    )
    
    # 또는 첫 번째 발견되는 샘플 사용 (타임스탬프/시퀀스 번호를 모르는 경우)
    # sample = get_single_timestamp_data(
    #     data_root=data_root,
    #     weather_condition='Snowy',
    #     time_of_day='Day'
    # )
    
    # 결과 출력
    print_sample_summary(sample)
    
    # 발견된 센서 수 확인
    camera_count = len(sample['cameras'])
    lidar_count = len(sample['lidar'])
    print(f"\n총 {camera_count}개의 카메라와 {lidar_count}개의 라이다 데이터를 찾았습니다.")
    
    if camera_count > 0 or lidar_count > 0:
        print("\n테스트 성공! 이제 모든 데이터를 처리할 수 있습니다.")
    else:
        print("\n데이터를 찾을 수 없습니다. 경로와 데이터 구조를 확인하세요.")
