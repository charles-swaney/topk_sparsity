import os
import logging
import csv

import torch
import torch.nn.functional as F
from torchvision import transforms
device = 'cuda'


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def denorm(batch, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def adv_test(model, device, test_loader, mean, std, epsilons, experiment_name):
    model = model.to(device).eval()
    correct = 0
    logging.info("Beginning adversarial tests.")
    base_dir = 'logs'
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    adv_log_path = os.path.join(experiment_dir, 'adversarial_log.csv')

    # Set up headers for log file
    with open(adv_log_path, 'w', newline='') as adv_file:
        train_writer = csv.writer(adv_file)
        train_writer.writerow(["Epsilon", "Test_Acc"])

    for eps in epsilons:
        correct = 0
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            data.requires_grad = True

            output = model(data)
            logits = output.logits if hasattr(output, 'logits') else output[0]
            init_pred = logits.max(1, keepdim=True)[1]

            if init_pred.item() != target.item():
                continue

            loss = F.cross_entropy(logits, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            data_denorm = denorm(data)
            perturbed_data = fgsm_attack(data_denorm, eps, data_grad)
            perturbed_data_normalized = transforms.Normalize(
                mean=mean, std=std)(perturbed_data)

            output = model(perturbed_data_normalized)
            logits = output.logits if hasattr(output, 'logits') else output[0]

            final_pred = logits.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                correct += 1

        final_acc = correct / float(len(test_loader))
        with open(adv_log_path, 'a', newline='') as adv_file:
            adv_writer = csv.writer(adv_file)
            adv_writer.writerow([eps, final_acc])

        logging.info(
            f"Epsilon: {eps}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}"
        )

    logging.info("Adversarial computations complete.")
