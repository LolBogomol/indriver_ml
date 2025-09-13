from torch import nn
import timm
from damage_classifier.config import BACKBONE, NUM_CLASSES


class DamageClassifier(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES, pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        if hasattr(self.backbone, "get_classifier"):
            in_ch = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0)
        else:
            try:
                in_ch = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Identity()
            except Exception:
                in_ch = 1536
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x) if hasattr(self.backbone, "forward_features") else self.backbone(x)
        logits = self.head(feats)
        return logits
