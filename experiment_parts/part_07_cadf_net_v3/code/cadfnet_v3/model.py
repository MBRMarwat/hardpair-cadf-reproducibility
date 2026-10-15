from __future__ import annotations

import math

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def modify_first_conv_to_5channels(model: nn.Module) -> nn.Module:
    old_conv = model.features[0][0]
    if old_conv.in_channels == 5:
        return model
    new_conv = nn.Conv2d(
        in_channels=5,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight[:, :3].copy_(old_conv.weight)
        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:].copy_(mean_weight.expand(-1, 2, -1, -1))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.features[0][0] = new_conv
    return model


class CADFNet(nn.Module):
    feature_dim = 576

    def __init__(
        self,
        pretrained: bool = True,
        generalist_hidden: int = 256,
        specialist_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = modify_first_conv_to_5channels(mobilenet_v3_small(weights=weights))
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.generalist_head = nn.Sequential(
            nn.Linear(self.feature_dim, generalist_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(generalist_hidden, 7),
        )
        self.specialist_head = nn.Sequential(
            nn.Linear(self.feature_dim, specialist_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(specialist_hidden, 2),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 5:
            raise ValueError(f"CADF encoder requires [B,5,H,W]; got {tuple(images.shape)}")
        values = self.features(images)
        values = self.avgpool(values)
        return self.feature_norm(torch.flatten(values, 1))

    def generalist_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.generalist_head(features)

    def specialist_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.specialist_head(features)

    def forward_views(
        self, generalist_view: torch.Tensor, specialist_view: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generalist_features = self.encode(generalist_view)
        specialist_features = self.encode(specialist_view)
        return (
            self.generalist_from_features(generalist_features),
            self.specialist_from_features(specialist_features),
        )

    def freeze_encoder(self) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        for parameter in self.feature_norm.parameters():
            parameter.requires_grad = False

    def unfreeze_last_blocks(self, count: int = 3) -> None:
        self.freeze_encoder()
        for module in self.features[-int(count) :]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for parameter in self.feature_norm.parameters():
            parameter.requires_grad = True

    def freeze_batchnorm_statistics(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False


class FeasibilityPredictor(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 32, outputs: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, outputs),
        )

    def forward(self, color_stats: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        logits = self.net(color_stats)
        return logits if return_logits else torch.sigmoid(logits)


class MLPGate(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        initial_alpha_cap: float = 0.75,
    ):
        super().__init__()
        if not 0.0 < initial_alpha_cap < 1.0:
            raise ValueError("initial_alpha_cap must be strictly between zero and one")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        initial_logit = math.log(initial_alpha_cap / (1.0 - initial_alpha_cap))
        self.alpha_cap_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

    def forward(self, gate_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if gate_inputs.ndim != 2 or gate_inputs.shape[1] != 8:
            raise ValueError(f"Gate requires [B,8] inputs; got {tuple(gate_inputs.shape)}")
        values = self.net(gate_inputs)
        alpha_cap = torch.sigmoid(self.alpha_cap_logit)
        alpha = torch.sigmoid(values[:, 0]) * alpha_cap
        beta = torch.sigmoid(values[:, 1])
        return alpha, beta
