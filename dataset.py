import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
DEFAULT_STD = (0.2470, 0.2435, 0.2616)


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
    ])

    return train_transform, test_transform


def build_dataloaders(
    data_dir="./data",
    batch_size=128,
    num_workers=2,
):
    train_transform, test_transform = build_transforms()
    data_dir = Path(data_dir)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader