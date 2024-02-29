import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer


class CifarTopkViT(nn.Module):
    '''
    ViT for CIFAR10
    '''
    def __init__(self,  k_value, num_layers=12):
        super(CifarTopkViT, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.model = VisionTransformer(image_size=32, patch_size=4, num_layers=num_layers, num_heads=12, hidden_dim=768, mlp_dim=3072,
                                       dropout=.5, attention_dropout=.5, num_classes=10)
        self.k_value = k_value
        for i in range(num_layers):
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