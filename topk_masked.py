import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F

class CifarViT(models.vision_transformer.vit_b_16):
    '''
    vanilla vision transformer with output dimension=10
    '''
    def __init__(self, pretrained=False, num_classes=10):
        super(CifarViT, self).__init__(pretrained=pretrained)

        self.heads = nn.Linear(self.heads[0].in_features, num_classes)

def top_k_mask(matrix, k):
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

class TopkGELU(nn.Module):
    def __init__(self, k):
        super(TopkGELU, self).__init__()
        self.k = k
    def forward(self, x):
        x = top_k_mask(F.gelu(x), self.k)
        return x
    
class TopkViT():
    '''
    k_value: the k in Top-k
    '''
    def __init__(self, model, k_value):
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.k_value = k_value
        for i in range(len(12)):
            layer_name = f'encoder_layer_{i}'
            gelu_module = getattr(getattr(self.model.encoder.layers, layer_name), '1')
            setattr(getattr(self.model.encoder.layers, layer_name), '1', TopkGELU(self.k_value))
    
    def forward(self, x):
        x = x.to(self.device)
        x = self.model(x)
        return x