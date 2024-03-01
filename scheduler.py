from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineAnnealingLR(_LRScheduler):
    '''
    LR is gradually warmed up, then apply cosine annealing
    '''
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_factor=0.1, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_factor = warmup_factor
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * (self.warmup_factor + (1.0 - self.warmup_factor) * alpha) for base_lr in self.base_lrs]
        else:
            cosine = 0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            return [base_lr * cosine for base_lr in self.base_lrs]
        
