import torch


class SparsityCalc():
    """
    A class for computing the sparsity of ReLU activations in ViTs, measured as the proportion
    of positive entries out of total entries.

    Attributes:
        - model: the model for which to calculate sparsity
        - hook_handles: the forward hooks forcomputing sparsities, attached to the ReLUs
        - cls_sparsities: a dictionary whose keys are the encoder block layer, and values
            are the sparsity for that layer's activation on the CLS token
        - patch_sparsities: " " averaged across the patch tokens
        - overall_sparsities: " " averaged across all tokens
    
    Methods:
        - count_positive: counts the proportion of positive entries in a given tensor
        - forward_hook: to define the forward hooks
        - add_hook_handles: attaches the forward hooks to each ReLU layer
        - get_sparsity: computes output to compute sparsities and update the sparsity dicts
        - remove_hooks: removes the forward hooks
    """
    def __init__(self, config: dict, model):
        self.model = model
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.hook_handles = []
        self.cls_sparsities = {f'encoder.layers.encoder_layer_{i}.mlp.1': [] 
                               for i in range(config['num_layers'])}
        self.patch_sparsities = {f'encoder.layers.encoder_layer_{i}.mlp.1': [] 
                                 for i in range(config['num_layers'])}
        self.overall_sparsities = {f'encoder.layers.encoder_layer_{i}.mlp.1': [] 
                                   for i in range(config['num_layers'])}
        
    def count_pos(self, tensor) -> float: 
        return (tensor > 0).to(torch.float).mean().item()
    
    

