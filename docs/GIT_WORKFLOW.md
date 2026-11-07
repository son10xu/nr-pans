# Workflow

## Local initialization

```bash
cd /path/to/NR-PANNs-GitHub-Starter
git init -b main
git add .
git diff --cached --stat
git commit -m "Initial project structure"
git remote add origin https://github.com/YOUR_USERNAME/NR-PANNs.git
git push -u origin main
```

## Work on a new architecture

```bash
git switch -c experiment/mr-fusion-v2
# edit code
git status
git add experiments/three_branch/ configs/
git commit -m "Experiment with MR fusion v2"
git push -u origin experiment/mr-fusion-v2
```

## Update code on Kaggle (public repo only)

```python
!git clone https://github.com/YOUR_USERNAME/NR-PANNs.git /kaggle/working/NR-PANNs
```

Private repos require credentials; use Kaggle secrets or securely configured Git credentials. Never embed credentials in a public notebook.

## Checklist

- Verify `git status` before pushing.
- Store datasets/checkpoints outside Git.
- Save the Git commit hash in experiment metadata.
- Do not tune against held-out test folds.
- Create one branch per architecture experiment, and merge after checks.
