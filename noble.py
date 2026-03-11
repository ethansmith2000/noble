import torch
import torch.nn as nn
import math
import inspect
from typing import Optional
import torch.nn.functional as F


def _kwargs_for(cls, all_kwargs):
    """Filter kwargs to only those accepted by cls.__init__."""
    params = inspect.signature(cls).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(all_kwargs)
    accepted = {name for name, p in params.items()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return {k: v for k, v in all_kwargs.items() if k in accepted}


class CosActivation(nn.Module):
    """
    Cosine activation with learnable (or fixed) per-dimension frequency and phase.
    
    Args:
        dim: Number of dimensions
        freq: Base frequency (used when freq_init="uniform")
        freq_init: Initialization strategy for frequencies:
            - "uniform": All dimensions use `freq` (default)
            - "harmonic": Linear spacing [min_freq, ..., max_freq] 
            - "geometric": Log spacing [min_freq, ..., max_freq] like positional encodings
            - "harmonic_random": Uniform random in [min_freq, max_freq]
            - "geometric_random": Log-uniform random (more density at lower frequencies)
        min_freq: Minimum frequency for harmonic/geometric init (default 0.5)
        max_freq: Maximum frequency for harmonic/geometric init (default 3.0 to limit gradient magnitude)
        trainable_freq: Whether freq_scale is learnable (default True)
        phase_init_std: Std for random phase initialization (default 0.0 = all zeros)
        phase_init_dist: Distribution for phase initialization ("normal" or "uniform")
        phase_init_min: Min bound for uniform phase initialization (default 0.0)
        phase_init_max: Max bound for uniform phase initialization (default 2π)
        phase_modulo: Modulo value to keep phase in bounds (default 2π, set to None to disable)
        freq_lr_mult: LR multiplier for freq_scale (if trainable)
        freq_bias_lr_mult: LR multiplier for freq_bias
    """

    def __init__(
        self, 
        dim, 
        freq: float = 1.0, 
        freq_init: str = "harmonic_random",
        min_freq: float = 0.8,
        max_freq: float = 1.2,
        trainable_freq: bool = True,
        phase_init_std: float = 0.1,
        phase_init_dist: str = "normal",
        phase_init_min: float = 0.0,
        phase_init_max: float = 2 * math.pi,
        freq_lr_mult: float = 2.0, 
        freq_bias_lr_mult: float = 4.0
    ):
        super().__init__()
        
        # Initialize frequencies based on strategy
        if freq_init == "uniform":
            freqs = torch.ones(dim) * freq
        elif freq_init == "harmonic":
            # Linear spacing: [min_freq, ..., max_freq]
            freqs = torch.linspace(min_freq, max_freq, dim)
        elif freq_init == "geometric":
            # Log spacing: [min_freq, ..., max_freq] - like positional encodings
            freqs = torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), dim))
        elif freq_init == "harmonic_random":
            # Uniform random in [min_freq, max_freq]
            freqs = torch.rand(dim) * (max_freq - min_freq) + min_freq
        elif freq_init == "geometric_random":
            # Log-uniform random: uniform in log-space, more density at lower frequencies
            log_min, log_max = math.log(min_freq), math.log(max_freq)
            freqs = torch.exp(torch.rand(dim) * (log_max - log_min) + log_min)
        else:
            raise ValueError(f"Unknown freq_init: {freq_init}. Use 'uniform', 'harmonic', 'geometric', 'harmonic_random', or 'geometric_random'")
        
        if trainable_freq:
            self.freq_scale = nn.Parameter(freqs)
            setattr(self.freq_scale, "lr_mult", freq_lr_mult)
        else:
            self.register_buffer("freq_scale", freqs)
        
        # Initialize phase with optional jitter
        if phase_init_dist == "normal":
            if phase_init_std > 0:
                phase_init = torch.randn(dim) * phase_init_std
            else:
                phase_init = torch.zeros(dim)
        elif phase_init_dist == "uniform":
            phase_init = torch.rand(dim) * (phase_init_max - phase_init_min) + phase_init_min
        else:
            raise ValueError(f"Unknown phase_init_dist: {phase_init_dist}. Use 'normal' or 'uniform'")
        
        # Apply modulo to keep phase in bounds
        phase_init = torch.fmod(phase_init, 2 * math.pi)
        
        self.freq_bias = nn.Parameter(phase_init)
        setattr(self.freq_bias, "lr_mult", freq_bias_lr_mult)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.freq_scale * x + self.freq_bias)

class Net(nn.Module):
    def __init__(self, act: nn.Module, dim: int, second_act: Optional[nn.Module] = None, residual_after: bool = False, residual_matrix: bool = False, full_dim: Optional[int] = None, lr_mult_power: float = 0.5, init_scale: float = 0.5, swiglu: bool = False):
        super().__init__()
        self.act = act
        if swiglu:
            self.fc = nn.Linear(dim, dim * 2)
        else:
            self.fc = nn.Linear(dim, dim)
        self.second_act = second_act if second_act is not None else act
        self.residual_after = residual_after
        self.residual_matrix = residual_matrix

        # Initialize fc weights
        std = init_scale / math.sqrt(dim)
        nn.init.normal_(self.fc.weight, std=std)

        lr_mult_fc = (full_dim / dim) ** lr_mult_power
        setattr(self.fc, "lr_mult", lr_mult_fc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act0 = self.act(x)
        if self.residual_matrix:
            act0 = self.fc(act0) + x
        else:
            act0 = self.fc(act0)
        act1 = self.second_act(act0)
        if self.residual_after:
            act1 = act1 + x
        return act1


class NetDouble(nn.Module):
    def __init__(self, act: nn.Module, dim: int, second_act: Optional[nn.Module] = None, third_act: Optional[nn.Module] = None, residual_after: bool = False, residual_matrix: bool = False, full_dim: Optional[int] = None, lr_mult_power: float = 0.5):
        super().__init__()
        self.act = act
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.second_act = second_act if second_act is not None else act
        self.third_act = third_act if third_act is not None else act
        self.residual_after = residual_after
        self.residual_matrix = residual_matrix
        
        # Initialize fc weights
        nn.init.normal_(self.fc1.weight, std=1.0 / math.sqrt(dim))
        nn.init.normal_(self.fc2.weight, std=1.0 / math.sqrt(dim))
        
        lr_mult_fc = (full_dim / dim) ** lr_mult_power
        setattr(self.fc1, "lr_mult", lr_mult_fc)
        setattr(self.fc2, "lr_mult", lr_mult_fc)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act0 = self.act(x)
        if self.residual_matrix:
            act0 = self.fc1(act0) + x
        else:
            act0 = self.fc1(act0)
        act1 = self.second_act(act0)
        if self.residual_matrix:
            act1 = self.fc2(act1) + x
        else:
            act1 = self.fc2(act1)
        act2 = self.third_act(act1)
        if self.residual_after:
            act2 = act2 + x
        return act2


class ActScaleShift(nn.Module):
    def __init__(self, act: nn.Module, dim: int, do_scale: bool = True, do_shift: bool = True):
        super().__init__()
        self.act = act
        self.do_scale = do_scale
        self.do_shift = do_shift
        if do_scale:    
            self.scale = nn.Parameter(torch.ones(dim))
            # lr mult the scale 5.0
            setattr(self.scale, "lr_mult", 5.0)
        else:
            self.scale = None
        if do_shift:
            self.shift = nn.Parameter(torch.zeros(dim))
            # lr mult the shift 3.0
            setattr(self.shift, "lr_mult", 3.0)
        else:
            self.shift = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(x)
        if self.do_scale:
            x = x * self.scale
        if self.do_shift:
            x = x + self.shift
        return x

class NegAct(nn.Module):
    def __init__(self, act: nn.Module):
        super().__init__()
        self.act = act
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.act(x)

class FlipAct(nn.Module):
    def __init__(self, act: nn.Module):
        super().__init__()
        self.act = act
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.act(-x)

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, x = x.chunk(2, dim=-1)
        return F.silu(gate) * x

def get_nonlinearity(name: str, dim: int, **kwargs) -> nn.Module:
    """
    Get a nonlinearity module by name.
    
    All kwargs are auto-routed to the relevant component classes based on their
    __init__ signatures. Unused kwargs are printed as a warning.
    
    Common kwargs for CosActivation: freq_init, min_freq, max_freq, trainable_freq,
        phase_init_std, freq, freq_lr_mult, freq_bias_lr_mult
    Common kwargs for Net/NetDouble: full_dim, lr_mult_power, init_scale, swiglu,
        residual_after, residual_matrix
    Common kwargs for ActScaleShift: do_scale, do_shift
    """
    consumed = set()

    def kw(cls, exclude=()):
        filtered = _kwargs_for(cls, kwargs)
        for e in exclude:
            filtered.pop(e, None)
        consumed.update(filtered)
        return filtered

    if name in ("none", "identity", "linear", None):
        result = nn.Identity()
    elif name == "tanh":
        result = nn.Tanh()
    elif name == "leakyrelu":
        result = nn.LeakyReLU(negative_slope=0.01)
    elif name == "gelu":
        result = nn.GELU()
    elif name == "silu":
        result = nn.SiLU()
    elif name == "gelu_net":
        result = Net(nn.GELU(), dim, **kw(Net))
    elif name == "leakyrelu_net":
        result = Net(nn.LeakyReLU(negative_slope=0.01), dim, **kw(Net))
    elif name == "silu_net":
        result = Net(nn.SiLU(), dim, **kw(Net))
    elif name == "tanh_net":
        result = Net(nn.Tanh(), dim, **kw(Net))
    elif name == "cos_net":
        cos_kw = kw(CosActivation)
        net_kw = kw(Net)
        result = Net(CosActivation(dim, **cos_kw), dim, second_act=CosActivation(dim, **cos_kw), **net_kw)
    elif name == "cos_net_double":
        cos_kw = kw(CosActivation)
        net_kw = kw(NetDouble)
        result = NetDouble(CosActivation(dim, **cos_kw), dim, second_act=CosActivation(dim, **cos_kw), third_act=CosActivation(dim, **cos_kw), **net_kw)
    elif name == "gelu_scale":
        result = ActScaleShift(nn.GELU(), dim, **kw(ActScaleShift))
    elif name == "leakyrelu_scale":
        result = ActScaleShift(nn.LeakyReLU(negative_slope=0.01), dim, **kw(ActScaleShift))
    elif name == "silu_scale":
        result = ActScaleShift(nn.SiLU(), dim, **kw(ActScaleShift))
    elif name == "silu_shift":
        result = ActScaleShift(nn.SiLU(), dim, do_shift=True, do_scale=False, **kw(ActScaleShift, exclude=("do_shift", "do_scale")))
    elif name == "tanh_scale":
        result = ActScaleShift(nn.Tanh(), dim, **kw(ActScaleShift))
    elif name == "cos":
        result = CosActivation(dim, **kw(CosActivation))
    elif name == "neg_silu":
        result = NegAct(nn.SiLU())
    elif name == "flip_silu":
        result = FlipAct(nn.SiLU())
    elif name == "neg_gelu":
        result = NegAct(nn.GELU())
    elif name == "flip_gelu":
        result = FlipAct(nn.GELU())
    elif name == "swiglu":
        result = SwiGLU()
    elif name == "swiglu_net":
        result = Net(SwiGLU(), dim, swiglu=True, **kw(Net, exclude=("swiglu",)))
    else:
        raise ValueError(f"Unknown nonlinearity: {name}")

    unused = set(kwargs) - consumed
    if unused:
        print(f"get_nonlinearity('{name}'): unused kwargs: {sorted(unused)}")

    return result


class NOBLELinear(nn.Module):
    def __init__(self,
                in_features: int,
                out_features: int,
                bias: bool = True,
                lora_rank: int = 32,
                lora_middle_nonlinearity: str = "cos_net",
                lora_lr_mult_power: float = 0.2,
                lora_up_init_scale: float = 0.01,
                linear_init_scale: float = 0.5,
                onesided_lr_mult: bool = True,
                nonlinearity_kwargs: dict = None,
                ): 
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        
        nonlinearity_kwargs = nonlinearity_kwargs or {}
        
        # Check if middle activation needs bias
        self.use_lora_down_bias = lora_middle_nonlinearity not in ("none", "identity", None)

        self.lora_middle_nonlinearity = get_nonlinearity(lora_middle_nonlinearity, lora_rank, full_dim=min(out_features, in_features), **nonlinearity_kwargs)
        
        # Main linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.normal_(self.linear.weight, mean=0.0, std=linear_init_scale / math.sqrt(in_features))
        if bias:
            self.linear.bias.data.zero_()

        self.lora_down = torch.nn.Linear(in_features, lora_rank, bias=self.use_lora_down_bias)
        nn.init.normal_(self.lora_down.weight, mean=0.0, std=1.0 / math.sqrt(in_features))
        if self.use_lora_down_bias:
            self.lora_down.bias.data.zero_()
        
        self.lora_up = torch.nn.Linear(lora_rank, out_features, bias=False)
        nn.init.normal_(self.lora_up.weight, mean=0.0, std=lora_up_init_scale / math.sqrt(lora_rank))
        
        # LR multipliers
        if onesided_lr_mult:
            lr_mult = (min(in_features, out_features) / lora_rank) ** (lora_lr_mult_power * 2)
            setattr(self.lora_up.weight, "lr_mult", lr_mult)
        else:
            lr_mult = (min(in_features, out_features) / lora_rank) ** lora_lr_mult_power
            setattr(self.lora_down.weight, "lr_mult", lr_mult)
            setattr(self.lora_up.weight, "lr_mult", lr_mult)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        linear_out = self.linear(x)

        lora_hidden = self.lora_down(x)
        lora_hidden = self.lora_middle_nonlinearity(lora_hidden)
        if lora_hidden.dtype != linear_out.dtype:
            lora_hidden = lora_hidden.to(linear_out.dtype)
        lora_out = self.lora_up(lora_hidden)
        
        return linear_out + lora_out
