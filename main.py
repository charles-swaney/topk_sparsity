import yaml
import argparse
import logging
import sys
import os
import torch

import torch.optim as optim

from utils.dataloader import load_cifar10
from models.train import train
from models.model_init import initialize_vit

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


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
    parser.add_argument(
        '--scheduler',
        type=str,
        required=True,
        help="Either multistep_lr or cos_annealing."
    )
    parser.add_argument(
        '--load_save_state',
        type=str,
        required=False,
        help="Which save state from models/checkpoints/ to load."
    )


    args = parser.parse_args()

    # Set up output directory
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

    # Saving logs and other info
    experiment_name = model_config['experiment_name']

    log_file_path = os.path.join(output_directory_path, f'{experiment_name}_training.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting training")

    # Set up dataloaders
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

    # set up model, optimizer
    model = initialize_vit(model_config).to(config['device'])
    print(f'model: {experiment_name} initialized successfully')

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    # set up scheduler
    if args.scheduler == 'multistep_lr':
        scheduler_config = config['multistep_lr']
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=scheduler_config['milestones'],
            gamma = scheduler_config['gamma']
        )
    elif args.scheduler == 'cos_annealing':
        scheduler_config = config['cos_annealing']
        scheduler = CosineAnnealingWarmupRestarts(
                optimizer=optimizer,
                first_cycle_steps=scheduler_config['first_cycle_steps'],
                cycle_mult=scheduler_config['cycle_mult'],
                max_lr=scheduler_config['max_lr'],
                min_lr=float(scheduler_config['min_lr']),
                warmup_steps=scheduler_config['warmup_steps'],
                gamma=scheduler_config['gamma'],
                last_epoch=-1
            )
    else:
        logging.error(f"Invalid scheduler: {args.scheduler}.")
        sys.exit(1)

    # load save state if applicable
    if args.load_save_state:
        save_state_to_load = args.load_save_state
        checkpoint_path = f"models/checkpoints/{experiment_name}/{experiment_name}_{save_state_to_load}.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info("Checkpoint loaded successfully.")
        else:
            logging.error(f"Checkpoint file does not exist at {checkpoint_path}")
    else:
        logging.info("No save state specified.")    

    train(config, model, train_loader, test_loader, optimizer, scheduler, experiment_name=experiment_name)


if __name__ == '__main__':
    main()
