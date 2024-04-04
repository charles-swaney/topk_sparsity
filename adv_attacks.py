import yaml
import argparse
import logging
import sys
import os

import torch
from utils.dataloader import load_cifar10
from utils.adv_attack_utils import adv_test
from models.model_init import initialize_vit


def adv_attacks():
    parser = argparse.ArgumentParser(
        description="Select a model to run against adversarial attacks.")
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
        '--load_save_state',
        type=str,
        required=True,
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

    epsilons = config['epsilons']
    device = config['device']
    mean = config['cifar_mean']
    std = config['cifar_std']
    experiment_name = model_config['experiment_name']

    log_file_path = os.path.join(output_directory_path, f'{experiment_name}_adversarial.log')
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Starting training")

    test_loader = load_cifar10(
        root='./data',
        config=config,
        train=False,
        download=True,
        persistent_workers=True
    )

    model = initialize_vit(model_config=model_config).to(device)
    model.eval()
    logging.info(f'model: {experiment_name} initialized successfully')

    save_state_to_load = args.load_save_state
    checkpoint_path = \
        f"models/checkpoints/{experiment_name}/{experiment_name}_{save_state_to_load}.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Checkpoint loaded successfully.")
    else:
        logging.error(f"Checkpoint file does not exist at {checkpoint_path}")

    # Run adversarial attacks with loaded model
    adv_test(
        model=model,
        device=device,
        test_loader=test_loader,
        mean=mean,
        std=std,
        epsilons=epsilons,
        experiment_name=experiment_name
    )


if __name__ == "__main__":
    adv_attacks()
