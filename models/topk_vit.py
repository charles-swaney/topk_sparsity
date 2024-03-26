import torch
import torch.nn as nn
import torch.nn.functional as F


class TopkViT(nn.Module):
    """
    A Vision Transformer with top-k masking applied to the FFN activations in each encoder block.
    The GeLU activations are replaced with TopkReLU modules.

    Attributes:
        - model: the underlying model on which the masking is applied
        - config: a config file which must contain at least the keys 'k_value' representing the
                k value in the masking, and 'num_layers', representing the model's depth
        - device: 'cuda' if possible

    Methods:
        - forward: uses the same forward pass as self.model
    """
    def __init__(self, model, config):
        super(TopkViT, self).__init__()
        self.device = config['device']
        self.model = model
        self.k_value = config['k_value']
        replace_activations(self.model, nn.ReLU, TopkReLU())
                
    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)


class TopkReLU(nn.Module):
    """
    A Module which first applies ReLU, then applies top k masking to the ReLU activation.

    Attributes:
        - k: the k-level in the masking.
    
    Methods:
        - top_k_mask: apply top k masking to input matrix
        - forward: forward pass of the module: first apply ReLU, then mask
    """
    def __init__(self, k):
        super(TopkReLU, self).__init__()
        self.k = k

    def top_k_mask(self, matrix, k):
        '''
        Inputs:
            matrix: the matrix to apply the mask to
            k: number of elements in each row to be nonzero
        Outputs:
            masked: masked version of matrix with only top k within each row nonzero
      '''
        _, indices = torch.topk(matrix, k, -1)
        mask = torch.zeros_like(matrix)
        mask.scatter_(-1, indices, 1)
        masked = matrix * mask
        return masked
    
    def forward(self, x):
        x = self.top_k_mask(F.relu(x), self.k)  # Use ReLU because it is easier to compute sparsity.
        return x
    

def replace_activations(model, old, new):
    """
    Recursively replace all activation functions specified by 'old' with 'new'.
    
    Parameters:
        model (nn.Module): The PyTorch model.
        old (activation function): The activation function to replace.
        new (activation function): The new activation function.
    """
    for name, child in model.named_children():
        if isinstance(child, old):
            setattr(model, name, new)
        else:
            replace_activations(child, old, new)
