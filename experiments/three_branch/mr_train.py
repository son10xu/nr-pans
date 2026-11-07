"""Train NR-PANNs from a fresh ESC-50 run using AudioSet pretrained weights.

Requirements: torch numpy scipy soundfile (plus original PANNs dependencies).
Examples:
  python train.py --root /path/to/PANNS --fold 5 --epochs 40
  python train.py --root /path/to/PANNS --all-folds --epochs 40
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from mr_encoder import TwoBranchPANNs, DEFAULT_CHECKPOINT_NAME
from mr_model import NRPANNsMR


class ESC50Dataset(Dataset):
    """ESC-50 official metadata, resampled to 32 kHz, exactly 5 seconds."""

    def __init__(self, esc_root: Path, folds: set[int], augment: bool = False):
        self.root = Path(esc_root)
        manifest = self.root / 'meta' / 'esc50.csv'
        if not manifest.is_file():
            raise FileNotFoundError(f'ESC-50 metadata missing: {manifest}')
        with manifest.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.rows = [row for row in rows if int(row['fold']) in folds]
        self.augment = augment
        if not self.rows:
            raise RuntimeError(f'No ESC-50 examples for folds {folds}')
        for row in self.rows:
            if not (self.root / 'audio' / row['filename']).is_file():
                raise FileNotFoundError(self.root / 'audio' / row['filename'])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio, sr = sf.read(self.root / 'audio' / row['filename'], dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 32000:
            from math import gcd
            g = gcd(sr, 32000)
            audio = resample_poly(audio, 32000 // g, sr // g).astype(np.float32)
        # Standard ESC-50 clips: 5 seconds, 160000 samples at 32k.
        target_length = 160000
        if len(audio) >= target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        x = torch.from_numpy(audio)
        if self.augment:
            # Waveform augmentations are only applied to train folds.
            if torch.rand(()) < 0.5:
                x = x * (0.8 + 0.4 * torch.rand(()))
            if torch.rand(()) < 0.5:
                x = x.roll(int(torch.randint(0, len(x), (1,)).item()))
        return {'waveform': x, 'label': torch.tensor(int(row['target']), dtype=torch.long)}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(root: Path, checkpoint_path: Path, nre_path: Path | None, alpha: float):
    if nre_path is not None:
        raise ValueError('NRE-V3 checkpoint is incompatible with three-output NRE-V4; use None')
    encoder = TwoBranchPANNs(checkpoint_path=checkpoint_path, root=root, freeze=False)
    model = NRPANNsMR(encoder, num_classes=50, alpha_init=alpha)
    assert all(p.requires_grad for name, p in model.named_parameters() if 'fc_audioset' not in name)
    return model


def optimizer_for(model: NRPANNsMR, lr_pretrained: float, lr_new: float, weight_decay: float):
    pretrained = [p for p in model.encoder.parameters() if p.requires_grad]
    new = (list(model.transient.parameters()) + list(model.nre.parameters())
           + list(model.fusion.parameters()) + [model.alpha] + list(model.classifier.parameters()))
    assert len(set(map(id, pretrained + new))) == len(pretrained + new), 'Duplicate optimizer parameters'
    assert {id(p) for p in model.parameters() if p.requires_grad} == set(map(id, pretrained + new))
    return torch.optim.AdamW([
        {'params': pretrained, 'lr': lr_pretrained},
        {'params': new, 'lr': lr_new},
    ], weight_decay=weight_decay)


def gradient_diagnostics(model: NRPANNsMR):
    p = model.encoder.panns
    groups = {
        'wavegram': p.pre_conv0.weight,
        'logmel': p.conv_block1.conv1.weight,
        'conv_block2': p.conv_block2.conv1.weight,
        'fc1': p.fc1.weight,
        'transient': model.transient.net[0].weight,
        'nre_v4': model.nre.mlp[0].weight,
        'fusion': model.fusion.projection.weight,
        'alpha': model.alpha,
        'classifier': model.classifier.weight,
    }
    return {name: None if parameter.grad is None else float(parameter.grad.detach().float().norm())
            for name, parameter in groups.items()}


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False, clip=5.0,
              gradient_check=False):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total = 0.0, 0, 0
    for batch_idx, batch in enumerate(loader):
        x = batch['waveform'].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if gradient_check and batch_idx == 0:
                    report = gradient_diagnostics(model)
                    print('First-batch gradient norms:', json.dumps(report))
                    missing = [name for name, norm in report.items() if norm is None or not np.isfinite(norm)]
                    if missing:
                        raise RuntimeError(f'Missing/non-finite gradients: {missing}')
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.detach().argmax(1) == y).sum().item()
        total += y.numel()
    if total == 0:
        raise RuntimeError('Empty DataLoader')
    return {'loss': total_loss / total, 'accuracy': total_correct / total, 'n': total}


def loaders_for(esc_root, fold, batch_size, workers, seed):
    val_fold = fold % 5 + 1
    training_folds = set(range(1, 6)) - {fold, val_fold}
    generator = torch.Generator().manual_seed(seed)
    def loader(folds, train):
        return DataLoader(
            ESC50Dataset(esc_root, folds, augment=train), batch_size=batch_size,
            shuffle=train, drop_last=False, num_workers=workers,
            pin_memory=False, generator=generator if train else None)
    return loader(training_folds, True), loader({val_fold}, False), loader({fold}, False), val_fold


def train_fold(args, fold, device):
    seed_everything(args.seed + fold)
    out = args.output / f'fold_{fold}'
    out.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, val_fold = loaders_for(
        args.root / 'data' / 'ESC-50', fold, args.batch_size, args.workers, args.seed + fold)
    model = build_model(args.root, args.checkpoint, args.nre_checkpoint, args.alpha).to(device)
    optimizer = optimizer_for(model, args.lr_backbone, args.lr_new, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best_acc, best_loss, best_epoch, stale = -1.0, float('inf'), 0, 0
    history = []
    save_path = out / 'best.pt'
    print(f'Fold={fold}, val_fold={val_fold}, train_folds={sorted(set(range(1, 6)) - {fold, val_fold})}')
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer, scaler, use_amp, args.clip,
                       gradient_check=(epoch == 1))
        va = run_epoch(model, val_loader, device, amp=use_amp)
        history.append({'epoch': epoch, 'train_loss': tr['loss'], 'train_accuracy': tr['accuracy'],
                        'val_loss': va['loss'], 'val_accuracy': va['accuracy'],
                        'alpha': float(model.alpha.detach().cpu())})
        better = va['accuracy'] > best_acc or (va['accuracy'] == best_acc and va['loss'] < best_loss)
        if better:
            best_acc, best_loss, best_epoch, stale = va['accuracy'], va['loss'], epoch, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_accuracy': best_acc,
                        'val_loss': best_loss, 'test_fold': fold, 'val_fold': val_fold}, save_path)
        else:
            stale += 1
        scheduler.step()
        print(f'Fold {fold} | Epoch {epoch:03d} | train {tr["accuracy"]:.2%}/{tr["loss"]:.4f} '
              f'| val {va["accuracy"]:.2%}/{va["loss"]:.4f} | alpha {float(model.alpha):.5f}')
        if stale >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break
    with (out / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    best = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(best['model'])
    te = run_epoch(model, test_loader, device, amp=use_amp)
    result = {'test_fold': fold, 'val_fold': val_fold, 'best_epoch': best_epoch,
              'best_val_accuracy': best_acc, 'test_accuracy': te['accuracy'], 'test_loss': te['loss'],
              'test_n': te['n']}
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print('TEST:', json.dumps(result))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='PANNs project root')
    parser.add_argument('--checkpoint', type=Path, default=None, help='AudioSet pretrained PANNs checkpoint')
    parser.add_argument('--nre-checkpoint', type=Path, default=None, help='Optional pretrained NRE-V3 weights')
    folds = parser.add_mutually_exclusive_group()
    folds.add_argument('--fold', type=int, choices=range(1, 6), default=5)
    folds.add_argument('--all-folds', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr-backbone', type=float, default=2e-5)
    parser.add_argument('--lr-new', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--output', type=Path, default=Path('runs/nrpanns_mr'))
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or (args.root / 'checkpoints' / DEFAULT_CHECKPOINT_NAME)
    return args


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'PANNs checkpoint not found: {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = [train_fold(args, fold, device) for fold in (range(1, 6) if args.all_folds else [args.fold])]
    if args.all_folds:
        summary = {'folds': results, 'mean_test_accuracy': float(np.mean([r['test_accuracy'] for r in results])),
                   'std_test_accuracy': float(np.std([r['test_accuracy'] for r in results]))}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('5-fold mean accuracy:', f'{summary["mean_test_accuracy"]:.2%}')


if __name__ == '__main__':
    main()
