import yaml
import argparse
import logging
import sys

import torch.optim as optim

from utils.dataloader import load_cifar10
from models.train import train
from utils.scheduler import CosineAnnealingWarmupRestarts
from models.model_init import initialize_vit


def main():
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(
        description="Train a specified model using configuration from config.yaml.")
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help="Model's name from available options in config.yaml"
    )
    args = parser.parse_args()

    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error("'config.yaml' file not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        sys.exit(1)

    try:
        model_config = config['models'][args.model_name]
    except KeyError:
        logging.error(f"Model name '{args.model_name}' not found in 'config.yaml'.")
        sys.exit(1)

    experiment_name = model_config['experiment_name']

    train_loader = load_cifar10(
        root='./data',
        config=config,
        train=True,
        download=True,
        persistent_workers=True
    )

    test_loader = load_cifar10(
        root='./data',
        config=config,
        train=False,
        download=True,
        persistent_workers=True
    )

    model = initialize_vit(model_config).to(config['device'])
    print(f'model: {experiment_name} initialized successfully')
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer=optimizer,
        first_cycle_steps=config['first_cycle_steps'],
        warmup_steps=config['warmup_steps']
    )

    train(config, model, train_loader, test_loader, scheduler, experiment_name=experiment_name)


if __name__ == '__main__':
    main()
