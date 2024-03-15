'''
This module defines the class TopkViT, which is a vision transformer whose GeLUs are replaced with TopkReLU modules.
opkReLU is an activation function which first applies ReLU to the affine transformation, then zeros
out all but the largest `k` entries.Initialize weights because PyTorch's VisionTransformer initializes weights to 0 by default.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class TopkViT(nn.Module):
    def __init__(self,  model, k_value):
        super(TopkViT, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.k_value = k_value
        for i in range(12):
            model.encoder.layers[i].mlp[1] = TopkReLU(self.k_value)
    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

class TopkReLU(nn.Module):
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
        x = self.top_k_mask(F.relu(x), self.k) # Use ReLU because it is easier to compute sparsity.
        return x
    
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
