from topk_vit import TopkViT
from transformers import ViTConfig, ViTForImageClassification


def initialize_vit(model_config):
    vit_config = ViTConfig(
        hidden_size=model_config['hidden_size'],
        num_hidden_layers=model_config['num_hidden_layers'],
        num_attention_heads=model_config['num_attention_heads'],
        intermediate_size=model_config['intermediate_size'],
        hidden_act=model_config['hidden_act'],
        image_size=model_config['image_size'],
        patch_size=model_config['patch_size'],
        num_channels=model_config['num_channels'],
        num_labels=model_config['num_labels'],
        device=model_config['device']
    )
    model = ViTForImageClassification(config=vit_config)

    if 'k_value' in model_config:  # initialize a top-k masked ViT
        model = TopkViT(model, model_config)

    return model
