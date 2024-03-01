import copy
import torch
import torch.nn as nn
import torch.optim as optim
from topk_masked import TopkViT
from train import train, save_checkpoint
from torchvision.models.vision_transformer import vit_b_16
from cifar_utils import load_cifar10
from torch.optim.lr_scheduler import CosineAnnealingLR
from scheduler import WarmupCosineAnnealingLR

num_classes = 10
max_k_value = 64
num_epochs = 200
train_batch_size = 128
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    models = []
    k_values = [8 * i for i in range(1, max_k_value // 8)]
    vanilla = vit_b_16(pretrained=False)
    vanilla.heads = nn.Sequential(nn.Linear(vanilla.heads[0].in_features, num_classes))
    models.append(vanilla)
    for k_value in k_values:
        temp = copy.deepcopy(vanilla)
        masked = TopkViT(temp, k_value)
        models.append(masked)

    train_loader = load_cifar10(root='./data', train=True, download=True, batch_size=128, num_workers=4)

    for model in models:
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=num_epochs, warmup_factor=.1, last_epoch=-1)
        train(model, train_loader, num_epochs, criterion, optimizer, scheduler, device)

if __name__ == '__main__':
    main()