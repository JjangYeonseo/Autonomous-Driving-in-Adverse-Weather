import os
import cv2
import torch
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset import MultimodalDataset
from model.fusion_model import FusionModel

def visualize_predictions(config, sample_count=10):
    # sample_size 인자 제거 ← 여기 수정
    dataset = MultimodalDataset(
        csv_path=config['dataset']['csv_path'],
        classes_path=config['dataset']['classes_path']
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FusionModel(num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(config['test']['model_path'], map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()

    with open(config['dataset']['classes_path'], encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]

    os.makedirs("outputs/vis", exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=sample_count)):
            if i >= sample_count:
                break

            img = batch['image'][0].permute(1, 2, 0).cpu().numpy() * 255
            img = img.astype('uint8').copy()
            image_tensor = batch['image'].to(device)
            bev_tensor = batch['bev'].to(device)
            label = batch['target_class'].item()

            output = model(image_tensor, bev_tensor)
            pred = torch.argmax(output, dim=1).item()

            cv2.putText(img, f"GT: {classes[label]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f"Pred: {classes[pred]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            save_path = f"outputs/vis/sample_{i}.png"
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print("✅ 시각화 완료! outputs/vis 폴더에서 확인하세요.")

# 직접 실행용
if __name__ == "__main__":
    config = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    visualize_predictions(config, sample_count=30)
