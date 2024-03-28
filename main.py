import yaml
import argparse
import logging
import sys
import os

import torch.optim as optim

from utils.dataloader import load_cifar10
from models.train import train
from utils.scheduler import build_scheduler
from models.model_init import initialize_vit


def main():
    parser = argparse.ArgumentParser(
        description="Train a specified model using configuration from config.yaml.")
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help="Model's name from available options in config.yaml"
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        required=True,
        help="Which directory to output logs."
    )  

    args = parser.parse_args()

    script_directory = os.path.dirname(os.path.abspath(__file__))
    topk_farm_directory = os.path.dirname(script_directory)
    output_directory_path = os.path.join(topk_farm_directory, args.output_directory)
    os.makedirs(output_directory_path, exist_ok=True)

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

    log_file_path = os.path.join(output_directory_path, f'{experiment_name}_training.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting training")

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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    # set up scheduler
    n_iter_per_epoch = len(train_loader)
    scheduler_config = config['scheduler']
    scheduler = build_scheduler(scheduler_config, optimizer, n_iter_per_epoch)

    train(config, model, train_loader, test_loader, scheduler, experiment_name=experiment_name)


if __name__ == '__main__':
    main()
