# NR-PANNs — ESC-50 experiments

Research workspace containing the two-branch and multi-resolution three-branch NR-PANNs experiments.

## Folder layout

- `experiments/two_branch/` — Wavegram + Log-Mel, NRE-V3, RGHF, training code, Jupyter notebook.
- `experiments/three_branch/` — Wavegram + Log-Mel + multi-resolution transient branch, NRE-V4, fusion, training code, Jupyter notebook.
- `configs/` — recorded proposed hyperparameters (documentation only; notebooks have their own editable settings).
- `scripts/` — local Git bootstrap helper.

Each experiment keeps its Python files next to its notebook because the original implementations use local imports such as `from mr_model import ...`.

## Requirements (external to this repository)

The upstream [PANNs AudioSet tagging implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn) and an appropriate `Wavegram_Logmel_Cnn14` AudioSet checkpoint are required. Also obtain ESC-50 separately. The repository intentionally excludes data, weights, and training artifacts.

Expected paths under your existing PANNs workspace (edit ROOT in each notebook):

```text
PANNS/
  audioset_tagging_cnn/pytorch/models.py
  checkpoints/Wavegram_Logmel_Cnn14_mAP=0.439.pth
  data/ESC-50/meta/esc50.csv
  data/ESC-50/audio/*.wav
```

These notebooks currently default to `/home/robinson/robotic/PANNS`. Update `ROOT`, `OUTPUT`, and related paths for Kaggle/Colab. If using Kaggle, downloaded ESC-50 may live in `/kaggle/input` instead of `ROOT/data/ESC-50`; adjust dataset-root references accordingly.

## Run locally

```bash
python -m pip install -r requirements.txt
cd experiments/three_branch
jupyter lab NR_PANNs_MR_3branch_FIXED.ipynb
```

Use `Kernel → Restart Kernel and Run All` after changes that affect module imports or architecture. Run the forward test before a long training run. GPU availability is not guaranteed simply because `torch.cuda.is_available()` is true.

## Reproducibility

Use official ESC-50 folds and avoid accessing test Fold 5 during tuning. For the Fold 5 experiment, train on folds 2/3/4 and validate on 1. Record Python/PyTorch/CUDA versions, GPU, seed, config and Git commit (`git rev-parse HEAD`) per run. Checkpoints should be stored outside Git or in a separately managed artifact store.

**Status:** Existing source and notebooks have been preserved and Python/notebook syntax checked; complete training with the original PANNs checkpoint + ESC-50 has not been validated in this environment. Do not interpret this scaffold as a verified accuracy benchmark.

## GitHub setup

Create an empty *private* GitHub repository, then from this folder:

```bash
git init -b main
git add .
git status
git commit -m "Initial NR-PANNs research project"
git remote add origin https://github.com/YOUR_USERNAME/NR-PANNs.git
git push -u origin main
```

Use a personal access token/SSH/GitHub CLI to authenticate; do **not** paste tokens into notebooks or source code.
# nr-pans
