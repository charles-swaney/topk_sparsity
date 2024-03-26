# topk_sparsity
comparing ReLU activation sparsity in Top-k masked ViTs compared to vanilla ViTs.

Following training regime described in https://arxiv.org/pdf/2112.13492.pdf

Code adapted in part from https://github.com/aanna0701/SPT_LSA_ViT

## Dependencies

This project depends on specific versions of `numpy`, `pillow-simd`, `torch`, and `torchvision` to ensure compatibility and performance on Compute Canada resources. Please ensure you install the exact versions listed below:

- `numpy`: 1.21.4+computecanada
- `pillow-simd`: 7.0.0.post3+computecanada
- `torch`: 1.10.0+computecanada
- `torchvision`: 0.11.1+computecanada
