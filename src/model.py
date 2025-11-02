import torch
import torch.nn as nn
import torchvision.models as M


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained : bool =True, dropout_p: float = 0.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        # load Resnet
        self.backbone = M.resnet18(
            weights=M.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.head = nn.Linear(in_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.dropout(feats)
        logits = self.head(feats)
        return logits

    def loss(self, logits, targets):
        return self.criterion(logits, targets)