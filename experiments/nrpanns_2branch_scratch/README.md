# NR-PANNs 2-branch — training from scratch

**Same two-branch PANNs architecture (Wavegram + Log-Mel + NRE-V3 + RGHF + Conv Blocks 2–6 + FC1 + ESC-50 classifier), but no pretrained weights are ever loaded.**

This project still imports the **architecture definition** `Wavegram_Logmel_Cnn14` from the original [PANNs repository](https://github.com/qiuqiangkong/audioset_tagging_cnn). It does **not** eliminate PANNs architectural dependency; it eliminates pretrained weights only.

## Files

- `two_branch_panns.py`: original PANNs architecture import + random initialization, no checkpoint loading
- `nr_panns.py`, `nre.py`, `rghf.py`: same model implementation as the 2-branch baseline
- `train.py`: ESC-50 split, optimization, best-checkpoint selection, train-from-scratch
- `evaluate.py`: independent evaluation of saved scratch checkpoint
- `train_kaggle.ipynb`: Kaggle setup and training commands

## Kaggle requirements

Add ESC-50 as a Kaggle Notebook input, with `meta/esc50.csv` and `audio/*.wav`.
The notebook clones the original PANNs **source code** and creates this layout:

```
/kaggle/working/PANNS/
  audioset_tagging_cnn/pytorch/models.py
  data/ESC-50/meta/esc50.csv
  data/ESC-50/audio/*.wav
```

No AudioSet checkpoint is needed. The model starts with fresh random PyTorch weights.

## Run (after installing dependencies and setting up root)

```bash
python -u train.py --root /kaggle/working/PANNS --fold 5 --epochs 150 \
  --patience 25 --batch-size 4 --workers 0 --lr-backbone 3e-4 \
  --lr-new 3e-4 --no-amp --output /kaggle/working/results/scratch_fold5
```

Train folds 2,3,4 (1200 samples), validation fold 1 (400), test fold 5 (400).
Keep test fold untouched when deciding hyperparameters. `best.pt` is picked by val accuracy; validation loss breaks ties.

**Expected performance:** scratch training may achieve substantially worse accuracy than AudioSet-pretrained baseline on ESC-50, due to only 1200 training clips. Accuracy is not guaranteed.

## Known behavior

- BatchNorm runs in train mode by default. `--no-amp` avoids previously observed AMP NaN gradients.
- `history.csv` is updated every epoch; `best.pt` is saved whenever validation improves.
- Kaggle `/kaggle/working` is ephemeral; save the completed notebook version and copy checkpoint outputs.
