import yaml
import argparse

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

from torchvision.models.vision_transformer import VisionTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR

import utils.dataloader as dataloader
from models.topk_vit import TopkViT, TopkGeLU, init_weights
from models.train import train, save_checkpoint, evaluate
from utils.scheduler import CosineAnnealingWarmupRestarts
from models.model_init import initialize_model

# need to applyCutMix [42], Mixup [43], Auto Augment [6], Repeated Augment [7] to all models

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Train a ViT with Top-k masking.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    train_loader = dataloader.load_cifar10(root='./data', train=True, download=True, batch_size=128, num_workers=4)
    test_loader = dataloader.load_cifar10(root='./data', train=False, download=True, batch_size=512, num_workers=4)

    for model in models:
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=warmup)
        train(model, train_loader, test_loader, num_epochs, criterion, optimizer, scheduler, test_interval, device)

if __name__ == '__main__':
    main()
