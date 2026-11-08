"""Evaluate a saved scratch checkpoint on the held-out ESC-50 fold."""
import argparse
import json
from pathlib import Path
import torch
from train import build_model, loaders_for, run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.01)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.root, args.alpha).to(device)
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    if state.get('test_fold') != args.fold:
        raise ValueError('Checkpoint test_fold does not match --fold')
    model.load_state_dict(state['model'], strict=True)
    _, _, test_loader, _ = loaders_for(args.root / 'data' / 'ESC-50', args.fold,
                                      args.batch_size, args.workers, seed=42 + args.fold)
    metrics = run_epoch(model, test_loader, device, amp=False)
    print(json.dumps({'test_fold': args.fold, 'checkpoint_epoch': state['epoch'],
                      'test_accuracy': metrics['accuracy'], 'test_loss': metrics['loss'],
                      'test_n': metrics['n']}, indent=2))


if __name__ == '__main__':
    main()
