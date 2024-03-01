import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer


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
            encoder_layer = self.model.encoder.layers[i]
            mlp_layer = encoder_layer.mlp
            for name, module in mlp_layer.named_children():
                if isinstance(module, nn.GELU):
                    setattr(mlp_layer, name, TopkGELU(self.k_value))
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