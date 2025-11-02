import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

class MetricPack:
    def __init__(self, num_classes: int, device: str = "cpu"):
        self.top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
        self.top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)

    def update(self, logits, targets):
        self.top1.update(logits, targets)
        self.top5.update(logits, targets)
        self.f1.update(logits, targets)

    def compute(self):
        return {
            "top1" : self.top1.compute().item(),
            "top5" :self.top5.compute().item(),
            "macro_f1" : self.f1.compute().item(),
        }

    def reset(self):
        for m in (self.top1, self.top5, self.f1):
            m.reset()