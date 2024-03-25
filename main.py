import yaml
import argparse
import logging

import torch.optim as optim
from torchvision.transforms import v2

from utils.dataloader import load_cifar10
from models.train import train
from utils.scheduler import CosineAnnealingWarmupRestarts
from models.model_init import initialize_model


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Train a specified model using configuration from config.yaml.")
    parser.add_argument('--model_name', type=str, required=True, help="Model's name as specified in config.yaml")
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model_config = config['models'][args.model_name]

    train_loader = load_cifar10(root='./data', config=config, train=True, download=True)
    test_loader = load_cifar10(root='./data', config=config, train=False, download=True)

    model = initialize_model(model_config).to(config['device'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, 
                                              first_cycle_steps=config['first_cycle_steps'], warmup_steps=config['warmup_steps'])
    train(model, config, train_loader, test_loader, scheduler, device=config['device'])

if __name__ == '__main__':
    main()
