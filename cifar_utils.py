import os
import subprocess
import tarfile
import pickle
import torch
from torchvision import datasets, transforms

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def download_cifar10(root='./data'):

    os.makedirs(root, exist_ok=True)
    cifar10_url = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"    
    tar_filepath = os.path.join(root, "cifar-10-binary.tar.gz")
    subprocess.run(["wget", cifar10_url, "-O", tar_filepath, "-q"])
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=root)

    os.remove(tar_filepath)

def load_cifar10(root='./data', train=True, download=True, batch_size=128, num_workers=4):
    if download:
        download_cifar10(root)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=test_transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    return dataloader


    