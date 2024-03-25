import torch.nn as nn

from models.topk_vit import TopkViT
from torchvision.models.vision_transformer import VisionTransformer


def initialize_model(config):
    model = VisionTransformer(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        hidden_dim=config['hidden_dim'],
        mlp_dim=config['mlp_dim'],
        num_classes=config['num_classes'],
    )

    if 'k_value' in config:  # then this should be a top k masked ViT.
        model = TopkViT(model, config)

    for module in model.modules():
        init_weights(module)

    return model


def init_weights(m):
    # Manually initialize the weights of a ViT using xavier normal.
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
