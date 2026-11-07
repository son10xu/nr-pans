"""NR-PANNs-MR: pretrained two-branch PANNs + multi-resolution transient stream.

Input waveform [B, samples] at 32 kHz. Fusion output has 128 channels,
so the original pretrained PANNs conv_block2..6 and fc1 remain reusable.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MultiResolutionTransient(nn.Module):
    """Two log-magnitude STFTs -> lightweight 32-channel transient map."""
    def __init__(self, out_channels=32):
        super().__init__()
        self.register_buffer('short_window', torch.hann_window(256), persistent=False)
        self.register_buffer('long_window', torch.hann_window(1024), persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels), nn.SiLU(),
        )

    @staticmethod
    def _normalize(x):
        # Per-sample log-energy normalization; avoid sample-level zero variance.
        mu = x.mean(dim=(2, 3), keepdim=True)
        sigma = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-5)
        return (x - mu) / sigma

    def _spectrogram(self, waveform, n_fft, hop, window, target_size):
        # STFT kept in FP32 even when CNN mixed precision is enabled.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            stft = torch.stft(waveform.float(), n_fft=n_fft, hop_length=hop,
                              window=window.float(), center=True, return_complex=True)
            x = torch.log1p(stft.abs()).unsqueeze(1)
            x = self._normalize(x)
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    def forward(self, waveform, target_size):
        short = self._spectrogram(waveform, 256, 128, self.short_window, target_size)
        long = self._spectrogram(waveform, 1024, 320, self.long_window, target_size)
        return self.net(torch.cat([short, long], dim=1))


class NoiseReliabilityEstimatorV4(nn.Module):
    """Joint mean+std descriptor from Wavegram(64), LogMel(64), Transient(32)."""
    def __init__(self, channels=(64, 64, 32), hidden=96):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * sum(channels), hidden), nn.ReLU(),
            nn.Linear(hidden, 3), nn.Sigmoid(),
        )

    def forward(self, fw, fm, ft):
        descriptors = []
        for x in (fw, fm, ft):
            descriptors.extend((x.mean((2, 3)), x.std((2, 3), unbiased=False)))
        r = self.mlp(torch.cat(descriptors, 1))
        return r[:, 0:1], r[:, 1:2], r[:, 2:3]


class ReliabilityGuidedFusionMR(nn.Module):
    """Reliability gating -> channel/time/frequency attention -> 128-channel delta."""
    def __init__(self, channels=160, out_channels=128, beta=0.5):
        super().__init__()
        self.beta = float(beta)
        hidden = max(channels // 16, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        self.temporal = nn.Sequential(nn.Conv1d(1, 1, 7, padding=3), nn.Sigmoid())
        self.frequency = nn.Sequential(nn.Conv1d(1, 1, 3, padding=1), nn.Sigmoid())
        self.projection = nn.Conv2d(channels * 2, out_channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, out_channels)
        # Small residual at initialization: keep pretrained PANNs input similar.
        nn.init.normal_(self.projection.weight, mean=0.0, std=1e-3)

    def forward(self, fw, fm, ft, rw, rm, rt):
        weights = [1.0 - self.beta * (1.0 - r.reshape(-1, 1, 1, 1))
                   for r in (rw, rm, rt)]
        merged = torch.cat([x * w for x, w in zip((fw, fm, ft), weights)], 1)
        attention = merged * self.channel(merged)
        attention = attention * self.temporal(attention.mean((1, 3)).unsqueeze(1)).unsqueeze(-1)
        attention = attention * self.frequency(attention.mean((1, 2)).unsqueeze(1)).unsqueeze(2)
        # No post-projection normalization: would undo the small-weight initialization.
        return self.projection(torch.cat((attention, merged), 1))


class NRPANNsMR(nn.Module):
    def __init__(self, encoder, num_classes=50, alpha_init=0.1,
                 conv_dropout=0.2, fc_dropout=0.5):
        super().__init__()
        if not hasattr(encoder, 'panns'):
            raise TypeError('Expected a TwoBranchPANNs encoder exposing .panns')
        self.encoder = encoder
        self.transient = MultiResolutionTransient(32)
        self.nre = NoiseReliabilityEstimatorV4()
        self.fusion = ReliabilityGuidedFusionMR()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.conv_dropout = float(conv_dropout)
        self.fc_dropout = float(fc_dropout)
        p = encoder.panns
        if p.conv_block2.conv1.in_channels != 128:
            raise RuntimeError('Pretrained conv_block2 must accept 128 channels')
        self.classifier = nn.Linear(p.fc1.out_features, num_classes)
        self.requires_grad_(True)
        # Pretrained AudioSet head is disconnected from the ESC-50 forward.
        for param in p.fc_audioset.parameters():
            param.requires_grad_(False)

    def forward_features(self, waveform, return_reliability=False):
        fw, fm, fbase = self.encoder(waveform)
        if fw.ndim != 4 or fw.shape[1] != 64 or fm.shape != fw.shape:
            raise RuntimeError(f'Expected [B,64,T,F] from both PANNs streams; got {fw.shape}, {fm.shape}')
        ft = self.transient(waveform, fw.shape[-2:])
        rw, rm, rt = self.nre(fw, fm, ft)
        delta = self.fusion(fw, fm, ft, rw, rm, rt)
        if delta.shape != fbase.shape:
            raise RuntimeError(f'Fusion mismatch: {delta.shape} vs {fbase.shape}')
        out = fbase + self.alpha * delta
        if return_reliability:
            return out, (rw, rm, rt)
        return out

    def forward_backbone(self, x):
        p = self.encoder.panns
        x = F.dropout(x, p=self.conv_dropout, training=self.training)
        for index in range(2, 7):
            pool = (1, 1) if index == 6 else (2, 2)
            x = getattr(p, f'conv_block{index}')(x, pool_size=pool, pool_type='avg')
            x = F.dropout(x, p=self.conv_dropout, training=self.training)
        x = x.mean(dim=3)
        x = x.max(dim=2).values + x.mean(dim=2)
        x = F.dropout(x, p=self.fc_dropout, training=self.training)
        x = F.relu(p.fc1(x))
        x = F.dropout(x, p=self.fc_dropout, training=self.training)
        return x

    def forward(self, waveform):
        return self.classifier(self.forward_backbone(self.forward_features(waveform)))
