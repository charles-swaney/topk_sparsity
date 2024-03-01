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

def train(model, dataloader, num_epochs, criterion, optimizer, scheduler, device='cuda'):
    losses = []
    trainAcc = []
    model = model.to(device).train()
    t1 = time.time()

    for epoch in range(num_epochs):
        for x, y in dataloader:
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

        if epoch % 20 == 0 and epoch > 0:
            t2 = time.time()
            elapsed = t2 - t1
            if epoch < 200:
                save_checkpoint(model, optimizer, epoch, elapsed, loss, accuracy,
                                    current_learning_rate, dir='models', filename=f'checkpoint_epoch_{epoch}.pth')
            else:
                save_checkpoint(model, optimizer, epoch, elapsed, loss, accuracy, 
                                    current_learning_rate, dir='models', filename=f'final.pth')

