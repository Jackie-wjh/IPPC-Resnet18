import os, yaml, torch, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from .dataset import build_dataloader
from .model import ResNet18Classifier

def _worstK_plus_other_confusion(cm: np.ndarray, classes: list, per_class_df: pd.DataFrame, k: int = 20):
    k = max(1, min(k, len(classes)))
    df = per_class_df.copy()
    if "f1-score" not in df.columns:
        df["f1-score"] = 0.0
    worst = df.sort_values("f1-score", ascending=True).head(k)
    worst_names = worst.index.tolist()

    name_to_idx = {name: i for i, name in enumerate(worst_names)}
    size_new = k + 1
    new_cm = np.zeros((size_new, size_new), dtype=np.int64)

    for i_old in range(len(classes)):
        ti = name_to_idx.get(classes[i_old], k)
        for j_old in range(len(classes)):
            pj = name_to_idx.get(classes[j_old], k)
            new_cm[ti, pj] += cm[i_old, j_old]

    row_sums = new_cm.sum(axis=1, keepdims=True)
    new_cm_norm = np.divide(new_cm.astype(np.float32), row_sums,
                            out=np.zeros_like(new_cm, dtype=np.float32), where=row_sums != 0)
    labels_new = worst_names + ["other"]
    return new_cm_norm, labels_new

@torch.no_grad()
def main(cfg_path: str, ckpt_path: str, out_dir: str = "figures", k_worst: int = 10, suffix: str = ""):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_ld, classes = build_dataloader(
        cfg["data_root"], cfg["batch_size"], cfg["num_workers"], cfg["strong_aug"]
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    model = ResNet18Classifier(
        num_classes=len(classes), pretrained=False,
        dropout_p=cfg.get("dropout", 0.0), label_smoothing=cfg.get("label_smoothing", 0.0)
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_logits, all_labels = [], []
    for imgs, labels in test_ld:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu()); all_labels.append(labels)
    if not all_logits:
        raise ValueError("Empty test dataloader.")

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds  = logits.argmax(1)

    acc_top1 = (preds == labels).float().mean().item()

    label_idx = list(range(len(classes)))
    report = classification_report(labels.numpy(), preds.numpy(),
                                  labels=label_idx, target_names=classes,
                                  digits=4, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    prec_macro = float(df.loc["macro avg", "precision"])
    f1_macro = float(df.loc["macro avg", "f1-score"])
    recall_macro = float(df.loc["macro avg", "recall"])    
    print(
    f"Acc@1: {acc_top1:.4f} | "
    f"Macro(P/R/F1): {prec_macro:.4f}/{recall_macro:.4f}/{f1_macro:.4f} | "
    )

    tag = f"_{suffix}" if suffix else ""
    csv_path = os.path.join(out_dir, f"per_class_report{tag}.csv")
    df.to_csv(csv_path); print(f"Saved: {csv_path}")

    #  Confusion Matrix（Worst-K + other）
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=label_idx)
    per_class_df = df.loc[[c for c in classes if c in df.index]]
    cm_small, labels_small = _worstK_plus_other_confusion(cm, classes, per_class_df, k=k_worst)

    fig_path = os.path.join(out_dir, f"confusion_worst{k_worst}{tag}.png")
    plt.figure(figsize=(7.4, 7.4))
    im = plt.imshow(cm_small, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0) 
    title_extra = f" – {suffix}" if suffix else ""
    plt.title(f"Normalized Confusion Matrix (Worst {k_worst} + other){title_extra}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(labels_small)), labels_small, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels_small)), labels_small, fontsize=8)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row-normalized", rotation=270, labelpad=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()
    print(f"Saved: {fig_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--k_worst", type=int, default=20)
    ap.add_argument("--suffix", default="")  
    args = ap.parse_args()
    main(args.cfg, args.ckpt, args.out_dir, args.k_worst, args.suffix)
