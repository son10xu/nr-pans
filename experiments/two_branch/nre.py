"""NRE độc lập cho hai nhánh pretrained PANNs."""
import torch
from torch import nn
import torch.nn.functional as F


class NoiseReliabilityEstimator(nn.Module):
    """Fw, Fm: [B,64,T,F] -> rw, rm: [B,1] (Sigmoid độc lập)."""
    def __init__(self, channels=64, hidden=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),
        )

    def forward(self, fw, fm):
        x = torch.cat([self.pool(fw).flatten(1),
                       self.pool(fm).flatten(1)], dim=1)
        r = self.net(x)
        return r[:, :1], r[:, 1:]


@torch.no_grad()
def reliability_targets(fw_clean, fm_clean, fw_noisy, fm_noisy, gamma=5.0):
    """Pseudo-label từ độ ổn định cosine; KHÔNG phải nhãn nhiễu thực."""
    def target(clean, noisy):
        similarity = F.cosine_similarity(clean.flatten(1), noisy.flatten(1), dim=1)
        return torch.exp(-gamma * (1 - similarity.clamp(-1, 1))).unsqueeze(1)
    return target(fw_clean, fw_noisy), target(fm_clean, fm_noisy)

import torch
import torch.nn as nn


class NoiseReliabilityEstimatorV2(nn.Module):
    """
    NRE V2: GAP + GMP.

    Input:
        Fw, Fm: [B,64,250,32]

    Output:
        rw, rm: [B,1]
    """

    def __init__(self, channels=64, hidden=64):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid()
        )

    def forward(self, fw, fm):

        zw = torch.cat([
            self.gap(fw).flatten(1),
            self.gmp(fw).flatten(1)
        ], dim=1)

        zm = torch.cat([
            self.gap(fm).flatten(1),
            self.gmp(fm).flatten(1)
        ], dim=1)

        z = torch.cat([zw, zm], dim=1)

        reliability = self.mlp(z)

        rw = reliability[:, :1]
        rm = reliability[:, 1:]

        return rw, rm

import torch
import torch.nn as nn


class NoiseReliabilityEstimatorV3(nn.Module):
    """
    NRE-V3: Global Average Pooling + Standard Deviation.

    Input:
        fw, fm: [B, 64, 250, 32]

    Output:
        rw, rm: [B, 1]
    """

    def __init__(self, channels=64, hidden=64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(channels * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid()
        )

    def forward(self, fw, fm):

        # 1. Wavegram: mean + standard deviation
        mean_w = fw.mean(dim=(2, 3))
        std_w = fw.std(dim=(2, 3), unbiased=False)

        # 2. Log-Mel: mean + standard deviation
        mean_m = fm.mean(dim=(2, 3))
        std_m = fm.std(dim=(2, 3), unbiased=False)

        # 3. Feature concatenation: [B, 256]
        z = torch.cat(
            [mean_w, std_w, mean_m, std_m],
            dim=1
        )

        # 4. Reliability prediction
        r = self.mlp(z)

        return r[:, :1], r[:, 1:2]
