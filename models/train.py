import torch
import os
import logging
import csv


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


def train(
        config,
        model,
        train_dataloader,
        test_dataloader,
        scheduler,
        experiment_name='vit_base'
    ):
    logging.info("Starting training process.")

    base_dir = 'logs'
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    training_log_path = os.path.join(experiment_dir, 'training_log.csv')
    validation_log_path = os.path.join(experiment_dir, 'validation_log.csv')

    # Initialize log files with headers outside the training loop.
    with open(training_log_path, 'w', newline='') as train_file:
        train_writer = csv.writer(train_file)
        train_writer.writerow(["Epoch", "Loss", "Accuracy"])

    with open(validation_log_path, 'w', newline='') as val_file:
        val_writer = csv.writer(val_file)
        val_writer.writerow(["Epoch", "Val_Loss", "Val_Accuracy"])

    device = config['device']

    num_epochs = config['num_epochs']
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])
    test_interval = config['test_interval']
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            logits = output.logits if hasattr(output, 'logits') else output[0]
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
        
        scheduler.step()
        
        epoch_loss = epoch_loss / len(train_dataloader.dataset)
        epoch_accuracy = correct / total
        with open(training_log_path, 'a', newline='') as train_file:
            train_writer=csv.writer(train_file)
            train_writer.writerow([epoch, epoch_loss, epoch_accuracy])

        # Save latest checkpoint for current model
        save_checkpoint(
            model,
            optimizer,
            epoch,
            base_dir='models/checkpoints',
            experiment_name=experiment_name,
            filename=f'{experiment_name}_latest_checkpoint.pth'
        )

        logging.info(f"Epoch {epoch+1}/{num_epochs} completed.")

        # Validation
        if epoch % test_interval == 0 and epoch > 0:
            current_loss, current_acc = evaluate(model=model,
                                                   dataloader=test_dataloader,
                                                   criterion=criterion,
                                                   device='cuda')
            
            with open(validation_log_path, 'a', newline='') as val_file:
                val_writer = csv.writer(val_file)
                val_writer.writerow([epoch, current_loss, current_acc])

            if current_acc > best_acc:
                best_acc = current_acc
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


def evaluate(model, dataloader, criterion, device='cuda'):
    model = model.to(device).eval()
    test_losses = []
    num_correct, test_total = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            logits = output.logits if hasattr(output, 'logits') else output[0]
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            test_losses.append(loss.item())

            num_correct += (preds == y).sum().item()
            test_total += preds.size(0)

    test_accuracy = num_correct / test_total
    mean_loss = sum(test_losses) / len(test_losses)

    return mean_loss, test_accuracy
