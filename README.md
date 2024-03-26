# Top-k Sparsity
Comparing ReLU activation sparsity in Top-k masked ViTs compared to vanilla ViTs.

Following training regime described in: https://arxiv.org/pdf/2112.13492.pdf

Code adapted in part from: https://github.com/aanna0701/SPT_LSA_ViT

## Dependencies

This project requires Python $3.7.7$ and pip version $24.0$ or newer. For the following libraries, any versions at least as up-to-date as the following will work:

- `pip`: 24.0
- `numpy`: 1.21.4
- `Pillow-SIMD`: 7.0.0.post3
- `PyYAML`: 5.4.1
- `torch`: 1.10.0
- `torchvision`: 0.11.1
- `typing_extensions`: 4.7.1
- `transformers`: 4.20.0

When not accessing an environment hosted by Compute Canada, one can use the most up-to-date versions of each package.

To install the requirements, simply run:
```sh
pip install -r requirements.txt
```
or
```sh
pip install -r cc_requirements.txt
```
if accessing an environment hosted by Compute Canada.
