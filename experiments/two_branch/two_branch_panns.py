"""Two-branch feature extraction using pretrained PANNs Wavegram_Logmel_Cnn14.

Place this file at <PROJECT_ROOT>/two_branch_panns.py or import it by path.
The project must contain audioset_tagging_cnn/pytorch/models.py and
checkpoints/Wavegram_Logmel_Cnn14_mAP=0.439.pth.
"""

from pathlib import Path
import importlib.util
import sys

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_ROOT = Path('/home/robinson/robotic/PANNS')
DEFAULT_CHECKPOINT_NAME = 'Wavegram_Logmel_Cnn14_mAP=0.439.pth'


def import_panns_model(root=DEFAULT_ROOT):
    """Import PANNs class directly from models.py, avoiding local models/ conflicts."""
    source_dir = Path(root) / 'audioset_tagging_cnn' / 'pytorch'
    models_file = source_dir / 'models.py'
    if not models_file.is_file():
        raise FileNotFoundError(f'PANNs source not found: {models_file}')
    source_str = str(source_dir)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)
    # Give each models.py location a stable, nonconflicting module name.
    module_name = 'panns_original_models'
    existing = sys.modules.get(module_name)
    if existing is not None and Path(getattr(existing, '__file__', '')).resolve() == models_file.resolve():
        return existing.Wavegram_Logmel_Cnn14
    spec = importlib.util.spec_from_file_location(module_name, models_file)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import PANNs source: {models_file}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module.Wavegram_Logmel_Cnn14


class TwoBranchPANNs(nn.Module):
    """Return (Fw, Fm, Fbase) before conv_block2 in pretrained PANNs.

    Input: audio [B, samples] at 32 kHz.
    Typical 5-second outputs: Fw/Fm [B,64,250,32], Fbase [B,128,250,32].
    """

    def __init__(self, checkpoint_path, root=DEFAULT_ROOT, freeze=True):
        super().__init__()
        PannsClass = import_panns_model(root)
        self.panns = PannsClass(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.panns.load_state_dict(state_dict, strict=True)
        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze pretrained weights and use eval mode for batch normalization."""
        for parameter in self.panns.parameters():
            parameter.requires_grad_(False)
        self.eval()
        return self

    def forward(self, audio):
        if audio.ndim != 2:
            raise ValueError(f'Expected audio [B, samples], received {tuple(audio.shape)}')
        m = self.panns

        # Wavegram stream
        fw = F.relu_(m.pre_bn0(m.pre_conv0(audio.unsqueeze(1))))
        fw = m.pre_block1(fw, pool_size=4)
        fw = m.pre_block2(fw, pool_size=4)
        fw = m.pre_block3(fw, pool_size=4)
        b, c, t = fw.shape
        if c % 32 != 0:
            raise RuntimeError(f'Cannot reshape Wavegram with {c} channels into groups of 32')
        fw = fw.reshape(b, c // 32, 32, t).transpose(2, 3)
        fw = m.pre_block4(fw, pool_size=(2, 1), pool_type='avg')

        # Log-Mel stream
        fm = m.spectrogram_extractor(audio)
        fm = m.logmel_extractor(fm)
        fm = fm.transpose(1, 3)
        fm = m.bn0(fm)
        fm = fm.transpose(1, 3)
        if self.training:
            fm = m.spec_augmenter(fm)
        fm = m.conv_block1(fm, pool_size=(2, 2), pool_type='avg')

        if fw.shape[2:] != fm.shape[2:]:
            raise RuntimeError(f'Feature size mismatch: Wavegram {tuple(fw.shape)} vs Log-Mel {tuple(fm.shape)}')
        return fw, fm, torch.cat((fw, fm), dim=1)


def create_two_branch_model(root=DEFAULT_ROOT, checkpoint_path=None, device=None, freeze=True):
    """Build two-branch pretrained model on the requested device."""
    root = Path(root)
    checkpoint_path = (root / 'checkpoints' / DEFAULT_CHECKPOINT_NAME) if checkpoint_path is None else Path(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
    model = TwoBranchPANNs(checkpoint_path, root=root, freeze=freeze).to(device)
    if freeze:
        model.eval()
    return model


def create_esc50_data(root=DEFAULT_ROOT, batch_size=8, test_fold=5, val_fold=4, num_workers=0):
    """Create train, val and test loaders using existing datasets/esc50_dataset.py."""
    from datasets.esc50_dataset import create_esc50_loaders

    return create_esc50_loaders(
        root=Path(root) / 'data' / 'ESC-50',
        batch_size=batch_size,
        test_fold=test_fold,
        val_fold=val_fold,
        num_workers=num_workers,
    )


@torch.no_grad()
def extract_two_branch_features(model, batch, device=None):
    """Extract feature maps from DataLoader batch; return (Fw, Fm, Fbase, labels)."""
    if device is None:
        device = next(model.parameters()).device
    audio = batch['waveform'].to(device)
    labels = batch['label'].to(device)
    model.eval()
    fw, fm, fbase = model(audio)
    return fw, fm, fbase, labels


if __name__ == '__main__':
    model = create_two_branch_model()
    train, val, test = create_esc50_data()
    fw, fm, fbase, labels = extract_two_branch_features(model, next(iter(train)))
    print('Wavegram:', tuple(fw.shape))
    print('Log-Mel:', tuple(fm.shape))
    print('Fbase:', tuple(fbase.shape))
    print('Labels:', tuple(labels.shape))
