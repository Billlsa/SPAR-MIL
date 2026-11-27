"""SPAR-MIL model implementation."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SPARMILOutput(NamedTuple):
    """Outputs returned by :class:`SPARMIL`."""

    logits: Tensor
    attention: Tensor
    instance_logits: Tensor


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=1)
        scale = self.expand(F.gelu(self.reduce(scale))).sigmoid()
        return x * scale


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=stride),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SqueezeExcitation(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.body(x) + self.skip(x))


class SpectralSpatialPatchEncoder(nn.Module):
    """Encode one hyperspectral patch with spatial and spectral branches."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            ConvNormAct(input_channels, base_channels),
            ResidualSEBlock(base_channels, base_channels * 2, stride=2),
            ResidualSEBlock(base_channels * 2, base_channels * 4, stride=2),
        )
        self.spectral = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
        )
        fused_dim = base_channels * 8 + base_channels * 2
        self.projection = nn.Sequential(
            nn.Linear(fused_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, patches: Tensor) -> Tensor:
        spatial = self.spatial(patches)
        spatial = torch.cat(
            [
                F.adaptive_avg_pool2d(spatial, 1).flatten(1),
                F.adaptive_max_pool2d(spatial, 1).flatten(1),
            ],
            dim=1,
        )
        spectral = F.adaptive_avg_pool2d(self.spectral(patches), 1).flatten(1)
        return self.projection(torch.cat([spatial, spectral], dim=1))


class CoordinateAwareMILPool(nn.Module):
    """Aggregate patch embeddings with coordinates and gated attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.attention_v = nn.Linear(embedding_dim, embedding_dim)
        self.attention_u = nn.Linear(embedding_dim, embedding_dim)
        self.attention_w = nn.Linear(embedding_dim, 1)
        self.instance_head = nn.Linear(embedding_dim, 1)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        embeddings: Tensor,
        coordinates: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_patches, _ = embeddings.shape
        embeddings = embeddings + self.coordinate_projection(coordinates)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, embeddings], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=valid_mask.device),
                ~valid_mask,
            ],
            dim=1,
        )
        tokens = self.norm(self.context(tokens, src_key_padding_mask=padding_mask))
        cls_embedding, patch_embeddings = tokens[:, 0], tokens[:, 1:]

        attention_logits = self.attention_w(
            torch.tanh(self.attention_v(patch_embeddings))
            * torch.sigmoid(self.attention_u(patch_embeddings))
        ).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~valid_mask, torch.finfo(attention_logits.dtype).min)
        attention = torch.softmax(attention_logits, dim=1)

        instance_logits = self.instance_head(patch_embeddings).squeeze(-1)
        instance_logits = instance_logits.masked_fill(~valid_mask, torch.finfo(instance_logits.dtype).min)
        weighted = torch.sum(patch_embeddings * attention.unsqueeze(-1), dim=1)
        critical_index = instance_logits.argmax(dim=1)
        critical = patch_embeddings[torch.arange(batch_size, device=embeddings.device), critical_index]
        masked_embeddings = patch_embeddings.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        mean = masked_embeddings.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        bag_embedding = self.fusion(torch.cat([cls_embedding, weighted, critical, mean], dim=1))
        return bag_embedding, attention, instance_logits


class SPARMIL(nn.Module):
    """Case-level SPAR-MIL classifier.

    Default values are provided for interface demonstration.
    """

    def __init__(
        self,
        input_channels: int,
        base_channels: int = 24,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.20,
        microbatch_size: int = 128,
        critical_logit_weight: float = 0.20,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.patch_encoder = SpectralSpatialPatchEncoder(
            input_channels=input_channels,
            base_channels=base_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.mil_pool = CoordinateAwareMILPool(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.microbatch_size = int(microbatch_size)
        self.critical_logit_weight = float(critical_logit_weight)

    def forward(
        self,
        patches: Tensor,
        coordinates: Tensor,
        valid_mask: Tensor | None = None,
    ) -> SPARMILOutput:
        if patches.ndim != 5:
            raise ValueError("patches must have shape [B, N, C, H, W]")
        if coordinates.shape != patches.shape[:2] + (2,):
            raise ValueError("coordinates must have shape [B, N, 2]")
        batch_size, num_patches, channels, height, width = patches.shape
        if valid_mask is None:
            valid_mask = torch.ones(
                batch_size,
                num_patches,
                dtype=torch.bool,
                device=patches.device,
            )
        if valid_mask.shape != patches.shape[:2]:
            raise ValueError("valid_mask must have shape [B, N]")
        if (~valid_mask).all(dim=1).any():
            raise ValueError("each bag must contain at least one valid patch")

        flat = patches.reshape(batch_size * num_patches, channels, height, width)
        chunks = [
            self.patch_encoder(flat[start : start + self.microbatch_size])
            for start in range(0, flat.shape[0], self.microbatch_size)
        ]
        embeddings = torch.cat(chunks, dim=0).reshape(batch_size, num_patches, -1)
        bag_embedding, attention, instance_logits = self.mil_pool(
            embeddings,
            coordinates,
            valid_mask,
        )
        bag_logits = self.classifier(bag_embedding).squeeze(-1)
        critical_logits = instance_logits.max(dim=1).values
        logits = bag_logits + self.critical_logit_weight * critical_logits
        return SPARMILOutput(logits, attention, instance_logits)

