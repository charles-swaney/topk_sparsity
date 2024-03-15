import torch
import os 
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(model, optimizer, epoch, dir='models/checkpoints', filename='checkpoint.pth'):
    filepath = os.path.join(dir, filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    os.makedirs(dir, exist_ok=True)
    torch.save(checkpoint, filepath)


def train(model, config, train_dataloader, test_dataloader, scheduler, device='cuda'):
    
    k_value_part = f"_masked_{config['k_value']}" if 'k_value' in config else ""
    experiment_name = f'vit{k_value_part}'
    writer = SummaryWriter(f'runs/{experiment_name}')

    num_epochs = config['num_epochs']
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    test_interval = config['test_interval']
    model = model.to(device).train()
    best_loss = 10 ** 4

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

        save_checkpoint(model, optimizer, epoch, dir='models/checkpoints', filename=f'{experiment_name}_latest_checkpoint.pth')
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_accuracy, epoch)

        if epoch % test_interval == 0 and epoch > 0:

            current_loss, test_accuracy = evaluate(model=model, dataloader=test_dataloader, criterion=criterion, device='cuda')

            writer.add_scalar('Loss/Test', current_loss, epoch)
            writer.add_scalar('Accuracy/Test', test_accuracy, epoch)

            if current_loss < best_loss:
                best_loss = current_loss
                save_checkpoint(model, optimizer, epoch, dir='models/checkpoints', filename=f'{experiment_name}_best_checkpoint.pth')
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
