import os
import subprocess
import tarfile
import pickle
from torchvision import datasets, transforms

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def download_cifar10(root='./data'):

    os.makedirs(root, exist_ok=True)
    cifar10_url = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    subprocess.run(["wget", cifar10_url, "-P", root])

    tar_filepath = os.path.join(root, "cifar-10-binary.tar.gz")
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=root)

    os.remove(tar_filepath)

def load_cifar10(root='./data', train=True, download=True, batch_size=64, num_workers=2):

    if download:
        download_cifar10(root)

    