# NR-PANNs Full Fine-Tuning on ESC-50

This package restarts training from AudioSet-pretrained `Wavegram_Logmel_Cnn14`, **not from random PANNs weights**. NRE-V3/RGHF/classifier start random by default. Pass `--nre-checkpoint` to initialize NRE from a standalone NRE-V3 training checkpoint.

Required project layout:

```text
PANNS/
  audioset_tagging_cnn/pytorch/models.py
  checkpoints/Wavegram_Logmel_Cnn14_mAP=0.439.pth
  data/ESC-50/meta/esc50.csv
  data/ESC-50/audio/*.wav
```

Install packages required by the original PANNs `models.py` and:

```bash
pip install torch numpy scipy soundfile
```

Single test fold (fold 5), with fold 1 for validation and folds 2,3,4 for training:

```bash
python train.py --root /path/to/PANNS --fold 5 --epochs 40 --batch-size 8
```

Official five held-out folds (one independent training run per test fold):

```bash
python train.py --root /path/to/PANNS --all-folds --epochs 40 --batch-size 8
```

Results: `runs/nrpanns/fold_*/best.pt`, `history.csv`, `result.json`; aggregated `summary.json` when `--all-folds` is used.

Architecture: 32 kHz waveform -> pretrained wavegram + logmel branches -> NRE-V3 -> RGHF -> alpha residual -> pretrained conv_block2..6 -> pretrained fc1 -> random 50-class classifier. Original PANNs SpecAugmentation and dropout are enabled in training. All feature-extraction PANNs parameters remain trainable (the unused original 527-way AudioSet head is frozen), with learning rate 1e-5 for pretrained PANNs and 1e-4 for newly added modules.

**Validation protocol:** test folds are not used for early stopping, scheduler decisions, or checkpoint selection; validation fold is the next fold modulo 5. ESC-50 original `target` indices are used. This is a train/val/test 3/1/1 split on each run, not training on all four non-test folds. Reporting should state this distinction. `--all-folds` gives a 5-fold held-out evaluation under this protocol.

**Caveats:** The original pretrained PANNs checkpoint, original source code, and ESC-50 audio were not provided here, so full training has not been run. Check source dependencies and ensure the original AudioSet checkpoint is compatible. 32 kHz resampling is handled by `scipy.signal.resample_poly` and may differ slightly from other audio resamplers. Models may be large: adjust batch size if CUDA OOM occurs.


## Jupyter Notebook

Open `NR_PANNs_ESC50_Training.ipynb` from this directory in JupyterLab or VS Code. Set `ROOT` to your PANNS installation in the configuration cell, then run cells in order. Training and evaluation are called as Python functions (not terminal commands). Run the optional 5-fold cell only when required.
