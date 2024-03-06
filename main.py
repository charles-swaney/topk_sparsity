import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

from torchvision.models.vision_transformer import VisionTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR

import cifar_utils
from topk_masked import TopkViT, TopkGeLU, initialize_weights
from train import train, save_checkpoint, evaluate
from scheduler import WarmupCosineAnnealingLR

# ViT initialization
image_size = 32
patch_size = 4
num_layers = 12
num_heads = 12  
hidden_dim = 768
mlp_dim = 3072
num_classes = 10

# Training constants 
num_epochs = 200
train_batch_size = 128
test_interval = 5
weight_decay = 0.01
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    # Initialize vanilla models, k values
    models = []
    k_values = [2 ** k for k in range(3,8)]
    vanilla = VisionTransformer(image_size=image_size, patch_size=patch_size, num_layers=num_layers, num_heads=num_heads, 
                                hidden_dim=hidden_dim, mlp_dim=mlp_dim, num_classes=num_classes)
    initialize_weights(vanilla)
    models.append(vanilla)
    for k_value in k_values:

        # Initialize masked ViTs
        temp = copy.deepcopy(vanilla)
        masked = TopkViT(temp, k_value)
        models.append(masked)

    train_loader = cifar_utils.load_cifar10(root='./data', train=True, download=True, batch_size=128, num_workers=4)
    test_loader = cifar_utils.load_cifar10(root='./data', train=False, download=True, batch_size=512, num_workers=4)

    for model in models:
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=weight_decay)
        scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=num_epochs, warmup_factor=.1, last_epoch=-1)
        train(model, train_loader, test_loader, num_epochs, criterion, optimizer, scheduler, test_interval, device)

if __name__ == '__main__':
    main()