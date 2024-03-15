from models.topk_vit import TopkViT, init_weights
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

    if 'k_value' in config:
        model = TopkViT(model, config['k_value'])

    for module in model.modules():
        init_weights(module)

    return model
