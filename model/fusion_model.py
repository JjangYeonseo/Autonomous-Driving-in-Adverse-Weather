import torch
import torch.nn as nn
import torchvision.models as models

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()

        # --- Image encoder (ResNet18) ---
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC
        self.image_out_dim = resnet.fc.in_features  # Usually 512

        # --- LiDAR BEV encoder (Simple CNN) ---
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.lidar_out_dim = 32

        # --- Fusion and Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(self.image_out_dim + self.lidar_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, bev):
        # image: (B, 3, 640, 480)
        # bev: (B, 1, H, W)

        image_feat = self.image_encoder(image)  # (B, 512, 1, 1)
        image_feat = image_feat.view(image_feat.size(0), -1)  # (B, 512)

        bev_feat = self.lidar_encoder(bev)  # (B, 32, 1, 1)
        bev_feat = bev_feat.view(bev_feat.size(0), -1)  # (B, 32)

        fusion = torch.cat([image_feat, bev_feat], dim=1)  # (B, 512 + 32)
        out = self.classifier(fusion)
        return out
