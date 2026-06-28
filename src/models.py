"""Both change-detection architectures: from-scratch and pretrained backbone."""
import torch
import torch.nn as nn

try:
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights
except ImportError:
    torchvision = None


def _combine(f1, f2):
    """Combine two branch features: concat + absolute difference."""
    return torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)


class SiameseScratchCNN(nn.Module):
    """Compact two-branch CNN with SHARED weights, built from scratch (Role C)."""

    def __init__(self, num_classes=4, feat_dim=128):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.encoder = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, feat_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 3, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, pre, post):
        return self.head(_combine(self.encoder(pre), self.encoder(post)))


class SiamesePretrainedCNN(nn.Module):
    """Two-branch model on a SHARED pretrained backbone, fine-tuned (Role D)."""

    def __init__(self, num_classes=4, backbone="resnet18", freeze_until=6):
        super().__init__()
        if torchvision is None:
            raise ImportError("torchvision is required: pip install torchvision")
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.encoder = net
        # freeze the first `freeze_until` children, fine-tune the rest
        for i, child in enumerate(self.encoder.children()):
            if i < freeze_until:
                for p in child.parameters():
                    p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 3, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, pre, post):
        return self.head(_combine(self.encoder(pre), self.encoder(post)))
