import os
import csv
import torch
import yaml
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from model.fusion_model import FusionModel
from data.dataset import MultimodalDataset

def test(config):
    # ✅ 타임스탬프 기반 하위 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # ✅ 데이터셋 로드
    dataset = MultimodalDataset(
        csv_path=config['dataset']['csv_path'],
        classes_path=config['dataset']['classes_path'],
        sample_size=config['dataset'].get('sample_size', None)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # ✅ 모델 로드
    model = FusionModel(num_classes=config['model']['num_classes'])
    checkpoint = torch.load(config['test']['model_path'], map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []

    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Running inference"):
            image = sample['image'].to(device)
            bev = sample['bev'].to(device)

            outputs = model(image, bev)
            _, preds = torch.max(outputs, 1)

            results.append({
                'true_label': int(sample['target_class'].item()),
                'pred_label': int(preds.item())
            })

    # ✅ 결과 저장
    df = pd.DataFrame(results)
    result_path = os.path.join(output_dir, "test_results.csv")
    df.to_csv(result_path, index=False)
    print(f"✅ 테스트 결과가 저장되었습니다: {result_path}")

if __name__ == "__main__":
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    test(config)
