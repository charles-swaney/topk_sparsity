import torch
import torch.nn as nn
import torch.nn.functional as F

class TopkViT(nn.Module):
    '''
    masked ViT
    '''
    def __init__(self,  model, k_value):
        super(TopkViT, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.k_value = k_value
        for i in range(12):
            model.encoder.layers[i].mlp[1] = TopkGELU(self.k_value)
    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

class TopkGELU(nn.Module):
    def __init__(self, k):
        super(TopkGELU, self).__init__()
        self.k = k

    def top_k_mask(self, matrix, k):
        '''
        Inputs:
            matrix: the matrix to apply the mask to
            k: number of elements in each row (corresponding to input vectors) to be nonzero
        Outputs:
            masked: masked version of matrix with only top k within each row nonzero
      '''
        _, indices = torch.topk(matrix, k, -1)
        mask = torch.zeros_like(matrix)
        mask.scatter_(-1, indices, 1)
        masked = matrix * mask
        return masked
    
    def forward(self, x):
        x = self.top_k_mask(F.gelu(x), self.k)
        return x