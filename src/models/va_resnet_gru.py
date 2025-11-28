import torch
import torch.nn as nn
import torchvision.models as models


class VAResNetGRU(nn.Module):
    def __init__(self, hidden_dim=256, pretrained=True):
        super().__init__()

        if pretrained:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            base = models.resnet18(weights=None)

        # remove final FC
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B, 512, 1, 1)
        self.feat_dim = 512

        self.gru = nn.GRU(
            input_size=self.feat_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Linear(hidden_dim * 2, 2)  # valence, arousal

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)                 # (B*T, 512, 1, 1)
        feats = feats.view(B, T, self.feat_dim)  # (B, T, 512)

        out, _ = self.gru(feats)                 # (B, T, 2*hidden)
        pooled = out.mean(dim=1)                 # (B, 2*hidden)
        va = self.head(pooled)                   # (B, 2)
        return va
