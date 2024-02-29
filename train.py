import torch
import numpy as np
import torch.optim as optim
import time

def save_checkpoint(model, optimizer, epoch, time_taken, loss, filepath='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'time_taken': time_taken,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def train(model, dataloader, num_epochs, criterion, optimizer, device='cuda'):
    losses = []
    trainAcc = []
    model = model.to(device)
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
            trainAcc.append(num_correct / num_samples)
            losses.append(loss.detach().cpu().numpy())

        if epoch % 20 == 0 and epoch > 0:
            t2 = time.time()
            elapsed = t2 - t1
            if epoch < 200:
                save_checkpoint(model, optimizer, epoch, elapsed, loss, filepath=f'checkpoint_epoch_{epoch}.pth')
            else:
                save_checkpoint(model, optimizer, epoch, elapsed, loss, filepath=f'final_checkpoint.pth')

