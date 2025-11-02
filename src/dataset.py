from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .transforms import build_transforms


def build_dataloader(root: str, batch_size: int = 64, workers: int = 8, strong_aug: bool = False):
    train_ds = ImageFolder(f"{root}/train", transform=build_transforms(True, strong_aug))
    val_ds = ImageFolder(f"{root}/valid", transform=build_transforms(False))
    test_ds = ImageFolder(f"{root}/test", transform=build_transforms(False))

    assert train_ds.classes == val_ds.classes == test_ds.classes, "Class lists differ across splits."
    
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_ld, val_ld, test_ld, train_ds.classes