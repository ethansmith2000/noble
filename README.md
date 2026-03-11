# NOBLE: Accelerating Transformers with Nonlinear Low-Rank Branches

Official implementation of [NOBLE: Accelerating Transformers with Nonlinear Low-Rank Branches](https://arxiv.org/abs/2603.06492) (arXiv 2026).

## Overview

NOBLE (**N**onlinear l**O**w-rank **B**ranch for **L**inear **E**nhancement) is an architectural augmentation that adds nonlinear low-rank branches to transformer linear layers.

The branch computes: **σ(xW_down)W_up** where σ is a learnable nonlinearity.

## Installation

```bash
pip install torch
```

Then copy `noble.py` to your project, or:

```bash
git clone https://github.com/canva-research/noble.git
```

`noble_simple.py` is a much more slimmed down version, that supports only CosNet as the nonlinearity choice, and uses settings from the paper

## Quick Start

`NOBLELinear` is a drop-in replacement for `nn.Linear`:

```python
import torch
from noble import NOBLELinear

# Replace nn.Linear with NOBLELinear
layer = NOBLELinear(in_features=768, out_features=768, lora_rank=32)

x = torch.randn(batch_size, seq_len, 768)
y = layer(x)  # Same output shape as nn.Linear
```

### Integration with Transformers

Replace linear layers in attention and MLP blocks:

```python
# Before
self.q_proj = nn.Linear(hidden_size, hidden_size)
self.k_proj = nn.Linear(hidden_size, hidden_size)
self.v_proj = nn.Linear(hidden_size, hidden_size)

# After
self.q_proj = NOBLELinear(hidden_size, hidden_size, lora_rank=32)
self.k_proj = NOBLELinear(hidden_size, hidden_size, lora_rank=32)
self.v_proj = NOBLELinear(hidden_size, hidden_size, lora_rank=32)
```

## Components

### NOBLELinear

The main module that augments a linear layer with a nonlinear low-rank branch.

```python
NOBLELinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    lora_rank: int = 32,                      # Bottleneck dimension
    lora_middle_nonlinearity: str = "cos_net", # Nonlinearity in the branch
    lora_lr_mult_power: float = 0.2,          # LR scaling for branch weights
    lora_up_init_scale: float = 0.01,         # Init scale for up projection
    linear_init_scale: float = 0.5,           # Init scale for main linear
)
```

### CosNet (Recommended Nonlinearity)

CosNet is a two-layer cosine nonlinearity with learnable frequency and phase, with a linear projection in the bottleneck:

```
cos(ω₁x + φ₁) → Linear → cos(ω₂x + φ₂)
```

This is the default and best-performing nonlinearity in our experiments.

### Available Nonlinearities

Use `get_nonlinearity()` to create activation modules:

```python
from noble import get_nonlinearity

# Recommended
act = get_nonlinearity("cos_net", dim=32, full_dim=768)

# Alternatives
act = get_nonlinearity("cos", dim=32)           # Single cosine layer
act = get_nonlinearity("gelu_net", dim=32, full_dim=768)
act = get_nonlinearity("silu_net", dim=32, full_dim=768)
act = get_nonlinearity("gelu", dim=32)          # Standard GELU
act = get_nonlinearity("silu", dim=32)          # Standard SiLU/Swish
```

### CosActivation

Learnable cosine activation with per dimension frequency and phase:

```python
from noble import CosActivation

cos_act = CosActivation(
    dim=32,
    freq_init="harmonic_random",  # Frequency initialization strategy
    min_freq=0.8,
    max_freq=1.2,
    trainable_freq=True,
    phase_init_std=0.1,
)
```

Frequency initialization strategies:
- `"uniform"`: All dimensions use the same frequency
- `"harmonic"`: Linear spacing from min_freq to max_freq
- `"geometric"`: Log spacing (like positional encodings)
- `"harmonic_random"`: Uniform random in [min_freq, max_freq]
- `"geometric_random"`: Log uniform random

## Learning Rate Scaling

NOBLE uses per parameter learning rate multipliers (via `lr_mult` attributes) to balance optimization between the main linear layer and the low-rank branch. If your optimizer doesn't support this, you can implement it with parameter groups:

```python
def get_param_groups(model, base_lr):
    groups = []
    for name, param in model.named_parameters():
        lr_mult = getattr(param, 'lr_mult', 1.0)
        groups.append({'params': [param], 'lr': base_lr * lr_mult})
    return groups

optimizer = torch.optim.AdamW(get_param_groups(model, lr=1e-4))
```

## Citation

```bibtex
@article{smith2026noble,
  title={NOBLE: Accelerating Transformers with Nonlinear Low-Rank Branches},
  author={Smith, Ethan},
  journal={arXiv preprint arXiv:2603.06492},
  year={2026}
}
```

## License

Apache 2.0
