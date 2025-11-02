# Sports‑100 Image Classification (ResNet‑18)

A reproducible PyTorch pipeline for **100‑class sports image classification** built on **ResNet‑18**.  
The project supports configuration‑driven experiments, warm‑up + cosine LR scheduling, **AMP** mixed precision,
and compact error analysis via a **worst‑K + other** confusion matrix.

> **Common defaults across all four experiments**: AMP enabled, **8** DataLoader workers, **batch size = 64**, and **weight decay = 0.05**.

---

## 📂 Project structure

```
IPPC_Resnet18/
├─ configs/                 # YAML configs for experiments
│  ├─ baseline.yaml
│  ├─ pretrained.yaml
│  ├─ robust.yaml
│  ├─ scratch_100.yaml
│  └─ compare.yaml
├─ data/                    # ImageFolder-style dataset
│  ├─ train/<class>/*.jpg
│  ├─ valid/<class>/*.jpg
│  └─ test/<class>/*.jpg
├─ figures/                 # Plots & per-class reports (generated)
├─ runs/                    # Checkpoints & logs for each experiment
│  ├─ baseline/     (best.pt, log.csv)
│  ├─ pretrained/   (best.pt, log.csv)
│  ├─ robust/       (best.pt, log.csv)
│  └─ scratch_100/  (best.pt, log.csv)
├─ src/
│  ├─ train.py              # training loop (Linear warm-up → Cosine annealing)
│  ├─ eval_test.py          # test-time eval + worstK+other confusion plot
│  ├─ dataset.py            # ImageFolder loaders & transforms
│  ├─ transforms.py         # train/val transforms (RandAugment, etc.)
│  ├─ model.py              # ResNet‑18 head (Identity → Dropout? → Linear)
│  ├─ metrics.py            # TorchMetrics pack (top1/top5/F1)
│  ├─ utils.py              # seeding, AverageMeter, helpers
│  ├─ plot_loss.py          # (optional) loss curve plotting from log.csv
│  └─ compare_val_acc.py    # (optional) compare val acc across runs
└─ requirements.txt
```

---

## ✨ Features

- **ResNet‑18** backbone with optional **ImageNet pretraining**
- Strong augmentation (**RandAugment**), optional **label smoothing** and **dropout**
- **AdamW** optimizer; **Linear warm‑up → Cosine annealing** LR, with **linear LR scaling** by batch size
- **AMP** mixed precision for speed & memory (`torch.amp.autocast('cuda')`, `amp.GradScaler('cuda')`)
- Metrics at validation/test: **Acc@1**, **Macro‑F1**, **Macro‑Recall**
- Diagnostic plots: training curves, worst‑K confusion matrices

---

## 📦 Setup

```bash
# (optional) new env
# conda create -n r18 python=3.10 -y && conda activate r18

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` (excerpt):
```
torch>=2.1.0
torchvision>=0.16.0
torchmetrics>=1.3.0
tqdm>=4.64
pyyaml>=6.0
numpy>=1.23
pandas>=1.5
matplotlib>=3.7
scikit-learn>=1.2
```

---

## 🚀 Train

**Pretrained fine‑tuning**
```bash
python -m src.train --config configs/pretrained.yaml
```

**From scratch (baseline)**
```bash
python -m src.train --config configs/baseline.yaml
```

**Robust (pretraining + strong aug + smoothing + dropout)**
```bash
python -m src.train --config configs/robust.yaml
```

**Longer from‑scratch schedule**
```bash
python -m src.train --config configs/scratch_100.yaml
```

Each run writes to `runs/<exp>/`:
- `best.pt` — checkpoint dict with `model.state_dict()`, `classes`, `cfg`
- `log.csv` — columns: `epoch,train_loss,val_loss,val_top1,val_top5,val_f1,lr`

---

## 🧪 Evaluate & plot (test set)

Generate **Acc@1(precision) / Macro‑F1 / Macro‑Recall** and a **worst‑K + other** confusion matrix:

```bash
python -m src.eval_test \
  --cfg  configs/pretrained.yaml \
  --ckpt runs/pretrained/best.pt \
  --out_dir figures \
  --k_worst 20 \
  --suffix pretrained_basic

# Example outputs:
#   figures/per_class_report_pretrained_basic.csv
#   figures/confusion_worst20_pretrained_basic.png
```

Optional helpers (if you choose to use them):

```bash
# Plot loss curves from a log.csv
python -m src.plot_loss --csv runs/baseline/log.csv --out figures/loss_curves_baseline.png --label baseline

# Compare validation accuracy across runs (see configs/compare.yaml if used)
python -m src.compare_val_acc --cfg configs/compare.yaml
```

---

## 🔧 Configs (YAML)

All configs share the same schema:

```yaml
seed: 312
data_root: data
out_dir: runs/<name>
batch_size: 64
num_workers: 8
pretrained: true|false
strong_aug: true|false
label_smoothing: 0.0|0.1
dropout: 0.0|0.2
lr: 3.0e-4 or 3.0e-3  # base LR; peak scales by (batch_size/64)
weight_decay: 0.05
epochs: 50|100
warmup: 10% epochs
amp: true
```

**Experiment matrix used in the report**

| Experiment (`out_dir`) | pretrained | strong_aug | label_smoothing | dropout | lr | weight_decay | epochs | warmup |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| `runs/baseline`     | ✗ | ✔ | 0.1 | 0.0 | 3e‑3  | 0.05 | 50  | 5  |
| `runs/pretrained`   | ✔ | ✗ | 0.0 | 0.0 | 3e‑4  | 0.05 | 50  | 5  |
| `runs/robust`       | ✔ | ✔ | 0.1 | 0.2 | 3e‑4  | 0.05 | 50  | 5  |
| `runs/scratch_100`  | ✗ | ✔ | 0.1 | 0.0 | 3e‑3  | 0.05 | 100 | 10 |

> LR linear scaling rule: `base_lr = cfg['lr'] * (batch_size / 64)`.

---

## 🧰 Tips & troubleshooting

- **AMP deprecations**: prefer `from torch import amp; amp.autocast('cuda')` and `amp.GradScaler('cuda')`.
- **Throughput**: if GPU is under‑utilized, increase `num_workers` (up to CPU cores), enable `pin_memory=True` and `persistent_workers=True` in loaders.
- **Permission denied**: run with `python src/train.py ...` (or add a shebang and `chmod +x`).
- **Reproducibility**: fixed seeds for Python/NumPy/PyTorch/CUDA; set CuDNN deterministic for final report runs.

---

## 🙏 Acknowledgments
Built with PyTorch and torchvision. ResNet‑18 follows the torchvision reference; ImageNet statistics are used for normalization.
