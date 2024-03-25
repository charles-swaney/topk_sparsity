import torch
import os
import logging
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(
        model,
        optimizer,
        epoch,
        base_dir='models/checkpoints',
        experiment_name='default',
        filename='checkpoint.pth'
):
    dir_path = os.path.join(base_dir, experiment_name)
    filepath = os.path.join(dir_path, filename)
    logging.info(f"Attempting to save checkpoint to: {filepath}.")

    if not os.path.exists(dir_path):
        logging.info("Directory does not exist, creating directory.")
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create dir {dir_path}: {e}")
            return
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
        
    try:
        torch.save(checkpoint, filepath)
        logging.info("Checkpoint saved.")
    except Exception as e:
        logging.error(f"Checkpoint failed to save at {filepath}: {e}")


def train(config, model, train_dataloader, test_dataloader, scheduler, device='cuda'):
    logging.info("Starting training process.")

    # Set up strings to save model checkpoints and information.
    k_value_part = f"_masked_{config['k_value']}" if 'k_value' in config else ""
    experiment_name = f'vit{k_value_part}'
    writer = SummaryWriter(f'runs/{experiment_name}')

    num_epochs = config['num_epochs']
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])
    test_interval = config['test_interval']
    model = model.to(device).train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
        
        scheduler.step()

        epoch_loss = epoch_loss / len(train_dataloader.dataset)
        epoch_accuracy = correct / total

        # Save latest checkpoint for current model
        save_checkpoint(
            model,
            optimizer,
            epoch,
            base_dir='models/checkpoints',
            experiment_name=experiment_name,
            filename=f'{experiment_name}_latest_checkpoint.pth'
        )

        # Log to tensorboard
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_accuracy, epoch)
        logging.info(f"Epoch {epoch+1}/{num_epochs} completed.")

        # Validation
        if epoch % test_interval == 0 and epoch > 0:
            current_loss, test_accuracy = evaluate(model=model,
                                                   dataloader=test_dataloader,
                                                   criterion=criterion,
                                                   device='cuda')

            writer.add_scalar('Loss/Test', current_loss, epoch)
            writer.add_scalar('Accuracy/Test', test_accuracy, epoch)

            if current_loss < best_loss:
                best_loss = current_loss
                logging.info("Saving new best model.")

                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    base_dir='models/checkpoints',
                    experiment_name=experiment_name,
                    filename=f'{experiment_name}_best_checkpoint.pth'
                )
                
    logging.info("Training completed successfully")
    writer.close()


def evaluate(model, dataloader, criterion, device='cuda'):
    model = model.to(device).eval()
    test_losses = []
    num_correct, test_total = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = criterion(output, y)
            test_losses.append(loss.item())

            preds = torch.argmax(output, dim=1)
            num_correct += (preds == y).sum().item()
            test_total += preds.size(0)

    test_accuracy = num_correct / test_total
    mean_loss = sum(test_losses) / len(test_losses)

    print(f'Test Loss: {mean_loss}, Test Accuracy: {test_accuracy}')
    return mean_loss, test_accuracy
