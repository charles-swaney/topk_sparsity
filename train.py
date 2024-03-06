import torch
import time
import os 

def save_checkpoint(model, optimizer, epoch, time_taken, loss, accuracy, lr, dir='models', filename='checkpoint.pth'):
    filepath = os.path.join(dir, filename)
    checkpoint = {
        'epoch': epoch,
        'time_taken': time_taken,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'learning_rate': lr
    }
    os.makedirs(dir, exist_ok=True)
    torch.save(checkpoint, filepath)

def train(model, train_dataloader, test_dataloader, num_epochs, criterion, optimizer, scheduler, test_interval=5, device='cuda'):
    losses = []
    trainAcc = []
    model = model.to(device).train()
    t1 = time.time()
    best_loss = 10 ** 4

    for epoch in range(num_epochs):
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(output, dim=1)
            num_correct = (preds == y).sum()
            num_samples = preds.size(0)
            accuracy = num_correct / num_samples
            trainAcc.append(accuracy)
            losses.append(loss.detach().cpu().numpy())
            current_learning_rate = optimizer.param_groups[0]['lr']
        
        scheduler.step()

        if epoch % test_interval == 0 and epoch > 0:

            t2 = time.time()
            elapsed = t2 - t1

            current_loss = evaluate(model=model, dataloader=test_dataloader, criterion=criterion, device='cuda')

            if current_loss < best_loss:
                best_loss = current_loss
                save_checkpoint(model, optimizer, epoch, elapsed, loss.item(), accuracy, current_learning_rate, dir='models',
                                filename=f'checkpoint_epoch_{epoch}.pth')

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
    return mean_loss