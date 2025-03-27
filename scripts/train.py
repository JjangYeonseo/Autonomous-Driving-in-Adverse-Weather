import os
import yaml
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from data.dataset import MultimodalDataset
from model.fusion_model import FusionModel

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

def train(config):
    # ✅ 타임스탬프 기반 하위 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_ckpt_dir = config['train']['checkpoint_dir']
    run_ckpt_dir = os.path.join(base_ckpt_dir, f"run_{timestamp}")
    os.makedirs(run_ckpt_dir, exist_ok=True)
    config['train']['checkpoint_dir'] = run_ckpt_dir  # 이후 경로에 자동 반영

    # ✅ Dataset
    dataset = MultimodalDataset(
        csv_path=config['dataset']['csv_path'],
        classes_path=config['dataset']['classes_path'],
        sample_size=config['dataset'].get('sample_size', None)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=4
    )

    # ✅ Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ✅ Model
    model = FusionModel(num_classes=config['model']['num_classes']).to(device)

    # ✅ Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    # ✅ Resume checkpoint (선택)
    resume_path = config['train'].get('resume_path', "")
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resumed training from epoch {start_epoch}")

    # ✅ Training loop
    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}"):
            image = batch['image'].to(device)
            bev = batch['bev'].to(device)
            labels = batch['target_class'].to(device)

            optimizer.zero_grad()
            outputs = model(image, bev)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"📘 Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        # ✅ Save checkpoint
        ckpt_path = os.path.join(config['train']['checkpoint_dir'], f"epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch, ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
