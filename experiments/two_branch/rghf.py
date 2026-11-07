
import torch
import torch.nn as nn


class ResidualAttentionFusion(nn.Module):
    """
    WASF-inspired residual feature fusion.

    Input:
        Fattention: [B,128,T,F]
        Fresidual:  [B,128,T,F]

    Output:
        Fout:       [B,128,T,F]
    """

    def __init__(self, channels=128):
        super().__init__()

        # W_out: 256 -> 128
        self.projection = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, Fattention, Fresidual):

        # [B,256,T,F]
        x = torch.cat(
            [Fattention, Fresidual],
            dim=1
        )

        # [B,128,T,F]
        x = self.projection(x)

        # LayerNorm trên chiều channel
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

import torch
import torch.nn as nn


class HierarchicalAttention(nn.Module):
    """Channel → Temporal → Feature-axis attention."""

    def __init__(self, channels=128, reduction=16):
        super().__init__()

        # 1. Channel Attention
        hidden = max(channels // reduction, 1)

        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

        # 2. Temporal Attention
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 3. Feature-axis Attention
        self.feature = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B,128,T,F]

        # Channel: [B,128,1,1]
        ac = self.channel(x)
        x = x * ac

        # Temporal: [B,1,T,1]
        at = self.temporal(
            x.mean(dim=(1, 3)).unsqueeze(1)
        ).unsqueeze(-1)
        x = x * at

        # Feature-axis: [B,1,1,F]
        af = self.feature(
            x.mean(dim=(1, 2)).unsqueeze(1)
        ).unsqueeze(2)
        x = x * af

        return x


class RGHF(nn.Module):
    """Reliability-Guided Hierarchical Fusion."""

    def __init__(self, channels=128, beta=0.5):
        super().__init__()

        self.beta = beta

        self.attention = HierarchicalAttention(channels)
        self.fusion = ResidualAttentionFusion(channels)

    def forward(self, fw, fm, rw, rm):
        # Đặc trưng gốc dùng làm residual
        fbase = torch.cat([fw, fm], dim=1)

        # Reliability weighting
        rw = rw.reshape(-1, 1, 1, 1)
        rm = rm.reshape(-1, 1, 1, 1)

        fw = (1 - self.beta * (1 - rw)) * fw
        fm = (1 - self.beta * (1 - rm)) * fm

        fweighted = torch.cat([fw, fm], dim=1)

        # Hierarchical attention
        fattention = self.attention(fweighted)

        # Công thức (16) WASF
        fout = self.fusion(fattention, fbase)

        return fout
