"""NR-PANNs with a single registration of pretrained PANNs parameters.

This expects TwoBranchPANNs (two_branch_panns_fixed.py), NRE-V3 (nre.py)
and RGHF (rghf.py). Waveform input is mono 32-kHz [B, samples].
"""
import torch
from torch import nn
from torch.nn import functional as F


class NRPANNs(nn.Module):
    def __init__(self, encoder: nn.Module, nre: nn.Module, rghf: nn.Module,
                 num_classes: int = 50, alpha_init: float = 0.01,
                 conv_dropout: float = 0.2, fc_dropout: float = 0.5):
        super().__init__()
        if not hasattr(encoder, 'panns'):
            raise TypeError('encoder must expose .panns (TwoBranchPANNs)')
        self.encoder = encoder
        self.nre = nre
        self.rghf = rghf
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.conv_dropout = conv_dropout
        self.fc_dropout = fc_dropout
        panns = self.encoder.panns  # local reference ONLY; no duplicate registration
        if panns.conv_block2.conv1.in_channels != 128:
            raise ValueError('Expected Wavegram_Logmel_Cnn14 conv_block2 input=128')
        self.classifier = nn.Linear(panns.fc1.out_features, num_classes)
        self.requires_grad_(True)  # Undo earlier encoder.freeze(), if used.
        # AudioSet 527-way head is unused in the ESC-50 forward pass.
        # Do not claim or attempt to fine-tune a disconnected classifier.
        for parameter in panns.fc_audioset.parameters():
            parameter.requires_grad_(False)

    def forward_features(self, waveform):
        fw, fm, _ = self.encoder(waveform)
        if fw.shape != fm.shape or fw.ndim != 4 or fw.shape[1] != 64:
            raise RuntimeError(f'Expected matched [B,64,T,F] maps; got {fw.shape}, {fm.shape}')
        rw, rm = self.nre(fw, fm)
        fbase = torch.cat((fw, fm), dim=1)
        delta = self.rghf(fw, fm, rw, rm)
        if delta.shape != fbase.shape:
            raise RuntimeError(f'Fusion shape mismatch: {delta.shape} vs {fbase.shape}')
        return fbase + self.alpha * delta

    def forward_backbone(self, x):
        panns = self.encoder.panns
        # Original PANNs applies dropout after concatenating the two streams,
        # and after each conv_block2..conv_block6.
        x = F.dropout(x, p=self.conv_dropout, training=self.training)
        for block_no in range(2, 7):
            block = getattr(panns, f'conv_block{block_no}')
            pool = (1, 1) if block_no == 6 else (2, 2)
            x = block(x, pool_size=pool, pool_type='avg')
            x = F.dropout(x, p=self.conv_dropout, training=self.training)
        x = x.mean(dim=3)
        x = x.max(dim=2).values + x.mean(dim=2)
        x = F.dropout(x, p=self.fc_dropout, training=self.training)
        x = F.relu(panns.fc1(x))
        x = F.dropout(x, p=self.fc_dropout, training=self.training)
        return x

    def forward(self, waveform):
        return self.classifier(self.forward_backbone(self.forward_features(waveform)))
