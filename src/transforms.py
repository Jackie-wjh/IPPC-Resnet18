import torchvision.transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool = True, strong_aug: bool = False):
    if train:
        aug = [
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ]
        if strong_aug:
            from torchvision.transforms.autoaugment import RandAugment
            aug.append(RandAugment(num_ops=2, magnitude=7))
        aug += [T.ToTensor(), T.Normalize(MEAN, STD)]
        return T.Compose(aug)
    else:
        return T.Compose([
            T.Resize(256), T.CenterCrop(224),
            T.ToTensor(), T.Normalize(MEAN, STD)
        ])