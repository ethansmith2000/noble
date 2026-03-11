import torch
import torch.nn as nn
import math
from typing import Optional


class CosActivation(nn.Module):
    """
    Cosine activation with learnable (or fixed) per-dimension frequency and phase.
    
    Args:
        dim: Number of dimensions
        min_freq: Minimum frequency for harmonic/geometric init (default 0.5)
        max_freq: Maximum frequency for harmonic/geometric init (default 3.0 to limit gradient magnitude)
        trainable_freq: Whether freq_scale is learnable (default True)
        phase_init_std: Std for random phase initialization (default 0.0 = all zeros)
        freq_lr_mult: LR multiplier for freq_scale (if trainable)
        freq_bias_lr_mult: LR multiplier for freq_bias
    """

    def __init__(
        self, 
        dim, 
        min_freq: float = 0.8,
        max_freq: float = 1.2,
        trainable_freq: bool = True,
        phase_init_std: float = 0.1,
        freq_lr_mult: float = 2.0, 
        freq_bias_lr_mult: float = 4.0
    ):
        super().__init__()

        # Uniform random in [min_freq, max_freq]
        freqs = torch.rand(dim) * (max_freq - min_freq) + min_freq

        if trainable_freq:
            self.freq_scale = nn.Parameter(freqs)
            setattr(self.freq_scale, "lr_mult", freq_lr_mult)
        else:
            self.register_buffer("freq_scale", freqs)
        
        # Initialize phase with optional jitter
        phase_init = torch.randn(dim) * phase_init_std
        
        # Apply modulo to keep phase in bounds
        phase_init = torch.fmod(phase_init, 2 * math.pi)
        
        self.freq_bias = nn.Parameter(phase_init)
        setattr(self.freq_bias, "lr_mult", freq_bias_lr_mult)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.freq_scale * x + self.freq_bias)


class Net(nn.Module):
    def __init__(self, act: nn.Module, dim: int, second_act: Optional[nn.Module] = None, full_dim: int = None, lr_mult_power: float = 0.5, init_scale: float = 0.5):
        super().__init__()
        self.act = act
        self.fc = nn.Linear(dim, dim)
        self.second_act = second_act if second_act is not None else act

        # Initialize fc weights
        std = init_scale / math.sqrt(dim)
        nn.init.normal_(self.fc.weight, std=std)

        lr_mult_fc = (full_dim / dim) ** lr_mult_power
        setattr(self.fc, "lr_mult", lr_mult_fc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_act(self.fc(self.act(x)))


class NOBLELinearCosNet(nn.Module):
    def __init__(self,
                in_features: int,
                out_features: int,
                bias: bool = True,
                lora_rank: int = 32,
                lora_lr_mult_power: float = 0.2,
                lora_up_init_scale: float = 0.01,
                linear_init_scale: float = 0.5,
                nonlinearity_kwargs: dict = None,
                ): 
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        
        nonlinearity_kwargs = nonlinearity_kwargs or {}

        self.cos_net = Net(
            CosActivation(lora_rank, **nonlinearity_kwargs), 
            lora_rank, 
            second_act=CosActivation(lora_rank, **nonlinearity_kwargs),
            full_dim=min(in_features, out_features)
        )
        
        # Main linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.normal_(self.linear.weight, mean=0.0, std=linear_init_scale / math.sqrt(in_features))
        if bias:
            self.linear.bias.data.zero_()

        self.lora_down = torch.nn.Linear(in_features, lora_rank, bias=True)
        nn.init.normal_(self.lora_down.weight, mean=0.0, std=1.0 / math.sqrt(in_features))
        self.lora_down.bias.data.zero_()
        
        self.lora_up = torch.nn.Linear(lora_rank, out_features, bias=False)
        nn.init.normal_(self.lora_up.weight, mean=0.0, std=lora_up_init_scale / math.sqrt(lora_rank))
        
        # LR multipliers
        lr_mult = (min(in_features, out_features) / lora_rank) ** (lora_lr_mult_power * 2)
        setattr(self.lora_up.weight, "lr_mult", lr_mult)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        linear_out = self.linear(x)
        lora_out = self.lora_up(self.cos_net(self.lora_down(x)))
        return linear_out + lora_out
