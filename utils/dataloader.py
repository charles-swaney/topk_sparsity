import logging

from torchvision import datasets
from torchvision.transforms import v2, autoaugment
from torch.utils.data import DataLoader


def load_cifar10(config, root='./data', train=True, download=True, **kwargs):
    if config is None:
        raise ValueError('No config.')

    train_batch_size = config['train_batch_size']
    test_batch_size = config['test_batch_size']
    num_workers = config['num_workers']

    train_transform = v2.Compose([
        autoaugment.AutoAugment(),
        v2.ToTensor(),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    test_transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    logging.info(f"Loading CIFAR-10 dataset. Train: {train}, Download: {download}")

    if train:
        dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=train_transform
        )
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers
            )
        except Exception as e:
            logging.error(f"Failed to create DataLoader: {e}")
            return None

    else:
        dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=test_transform
        )
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        except Exception as e:
            logging.error(f"Failed to create DataLoader: {e}")
            return None
        
    logging.info("Successfully created DataLoader.")
    return dataloader
