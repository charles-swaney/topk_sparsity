import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from topk_masked import TopkViT
from train import train, save_checkpoint
from torchvision.models.vision_transformer import vit_b_16
from cifar_utils import load_cifar10

num_classes = 10
max_k_value = 64
num_epochs = 100
train_batch_size = 128
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam()
device = 'cuda'

def main():

    models = []
    k_values = [8 * i for i in range(1, np.floor(max_k_value / 8))]
    vanilla = vit_b_16(pretrained=False)
    vanilla.heads = nn.Sequential(nn.Linear(vanilla.heads[0].in_features, num_classes))
    models.append(vanilla)
    for k_value in k_values:
        temp = copy.deepcopy(vanilla)
        masked = TopkViT(temp, k_value)
        models.append(masked)

    # code for setting up dataloader
    
    # test_loader = load_cifar10(root='./data', train=True, download=True, batch_size=train_batch_size, num_workers=2)

    for model in models:
        train(model, test_loader, num_epochs, criterion, optimizer, device)

if __name__ == '__main__':
    main()