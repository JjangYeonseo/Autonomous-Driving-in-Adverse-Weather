import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from utils.lidar_utils import load_pcd_as_bev
from utils.label_parser import parse_image_label, parse_lidar_label

class MultimodalDataset(Dataset):
    def __init__(self, csv_path, classes_path, sample_size=None, random_state=None):
        self.data = pd.read_csv(csv_path)
        self.sample_size = sample_size  
        self.random_state = random_state  

        # 클래스 정보 로드
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class2idx = {cls.lower(): i for i, cls in enumerate(self.classes)}

        # 클래스 균형 고려한 Stratified Sampling
        if sample_size and sample_size < len(self.data):
            if 'label_path' in self.data.columns:
                self.data['label_class'] = self.data['label_path'].apply(
                    lambda x: self._get_first_label_class(x)
                )
                self.data = self.data.groupby('label_class', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), sample_size // len(self.class2idx)), random_state=random_state)
                ).reset_index(drop=True)

        # 이미지 - 라이다 쌍 구성
        self.samples = self._pair_image_lidar()

    def _get_first_label_class(self, label_path):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'objects' in data and data['objects']:
                return data['objects'][0]['classTitle'].strip().lower()
            elif 'shapes' in data and data['shapes']:
                return data['shapes'][0]['label'].strip().lower()
            return 'unknown'
        except:
            return 'unknown'

    def _pair_image_lidar(self):
        paired = []
        grouped = self.data.groupby(['weather', 'time'])

        for (weather, time), group in grouped:
            images = group[group['modality'] == 'image']
            lidars = group[group['modality'] == 'lidar']

            if 'direction' not in lidars.columns:
                continue
            lidars = lidars[lidars['direction'].notnull()]

            for _, img_row in images.iterrows():
                lidar_row = lidars[lidars['direction'].str.contains('Lidar_Center', na=False)]
                if not lidar_row.empty:
                    paired.append((img_row, lidar_row.iloc[0]))

        return paired

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_row, lidar_row = self.samples[idx]

        # 이미지 로딩
        image = cv2.imread(img_row['data_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        # 라이다 BEV 변환
        bev = load_pcd_as_bev(lidar_row['data_path'])
        bev = torch.tensor(bev).unsqueeze(0).float()

        # 라벨 로딩
        image_label_path = img_row['label_path']
        lidar_label_path = lidar_row['label_path']

        image_classes = parse_image_label(image_label_path, self.class2idx)
        lidar_classes = parse_lidar_label(lidar_label_path, self.class2idx)

        # 이미지와 라이다 클래스 병합
        merged_classes = image_classes + lidar_classes
        if not merged_classes:
            merged_classes = [0]  # fallback: 'unknown'

        target_class = merged_classes[0]  # 우선 첫 번째만 사용

        return {
            'image': image,
            'bev': bev,
            'target_class': torch.tensor(target_class)
        }

