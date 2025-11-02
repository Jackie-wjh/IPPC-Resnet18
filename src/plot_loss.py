import os, argparse, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main(csv_path: str, out_path: str, label: str | None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    need = {"epoch", "train_loss", "val_loss"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns in {csv_path}. Need {need}, have {set(df.columns)}")

    # if no label given, use parent folder name (e.g., "baseline")
    if not label:
        p = Path(csv_path)
        label = p.parent.name or p.stem

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["val_loss"],   label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title(f"{label} — Train vs Val Loss")  # <-- show "baseline" on top
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="figures/loss_curves.png")
    ap.add_argument("--label", default=None, help="Text to prefix in the title (e.g., 'baseline')")
    args = ap.parse_args()
    main(args.csv, args.out, args.label)
