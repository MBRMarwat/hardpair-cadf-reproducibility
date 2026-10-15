from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class HardPairNet(nn.Module):
    feature_dim = 576

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.norm = nn.LayerNorm(self.feature_dim)
        self.generalist = nn.Linear(self.feature_dim, 7)
        self.specialist = nn.Linear(self.feature_dim, 2)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        values = self.features(images)
        values = self.avgpool(values)
        return torch.flatten(values, 1)

    def heads_from_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm(features)
        return self.generalist(normalized), self.specialist(normalized)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.heads_from_features(self.extract_features(images))

    def freeze_for_partial_finetuning(self, last_modules: int = 3) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        for module in self.features[-last_modules:]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for parameter in self.norm.parameters():
            parameter.requires_grad = True
        for parameter in self.generalist.parameters():
            parameter.requires_grad = True
        for parameter in self.specialist.parameters():
            parameter.requires_grad = True

    def freeze_batchnorm_statistics(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False


class LogisticGate(nn.Module):
    def __init__(self, max_alpha: float = 0.75, initial_alpha: float = 0.5):
        super().__init__()
        self.max_alpha = float(max_alpha)
        self.linear = nn.Linear(5, 1)
        nn.init.zeros_(self.linear.weight)
        ratio = torch.tensor(initial_alpha / max_alpha).clamp(1e-6, 1 - 1e-6)
        nn.init.constant_(self.linear.bias, float(torch.logit(ratio)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.max_alpha * torch.sigmoid(self.linear(values))

