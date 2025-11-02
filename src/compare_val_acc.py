# compare_val_acc.py
import os, yaml, pandas as pd
import matplotlib.pyplot as plt


def _load_runs_from_cfg(cfg):
    """
    Expect:
      metrics_csv_multi:
        - {label: baseline, path: logs/base.csv}
        - {label: augmix,   path: logs/augmix.csv}
        - {label: ls0.1,    path: logs/ls01.csv}
        - {label: drop0.3,  path: logs/drop03.csv}
    """
    runs = cfg.get("metrics_csv_multi", [])
    clean = []
    for r in runs:
        label = r.get("label")
        path  = r.get("path")
        if not label or not path:
            print(f"[WARN] Skip invalid run spec (need label/path): {r}")
            continue
        clean.append({"label": label, "path": path})
    return clean


def _pick_acc_column(df):
    """
    Support either 'val_top1' or 'val_acc' as accuracy column.
    Returns the column name if found, else None.
    """
    if "val_top1" in df.columns:
        return "val_top1"
    if "val_acc" in df.columns:
        return "val_acc"
    return None


def plot_val_acc_compare(runs):
    """
    Multi-run comparison figure of validation accuracy.
    """
    if not runs:
        print("[WARN] No runs provided.")
        return

    plt.figure()
    any_plotted = False
    for r in runs:
        label, path = r["label"], r["path"]
        if not os.path.exists(path):
            print(f"[WARN] CSV not found: {path}. Skip.")
            continue
        df = pd.read_csv(path)
        if "epoch" not in df.columns:
            print(f"[WARN] Missing 'epoch' in {path}")
            continue
        acc_col = _pick_acc_column(df)
        if acc_col is None:
            print(f"[WARN] Missing val acc column in {path} (need 'val_top1' or 'val_acc').")
            continue
        plt.plot(df["epoch"], df[acc_col], label=label)
        any_plotted = True

    if any_plotted:
        plt.xlabel("epoch"); plt.ylabel("val accuracy")
        plt.title("Val Accuracy Comparison")
        plt.legend(); plt.tight_layout()
        os.makedirs("figures", exist_ok=True)
        plt.savefig("figures/val_acc_compare.png", dpi=220)
        plt.close()
        print("Saved: figures/val_acc_compare.png")
    else:
        plt.close()


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    runs = _load_runs_from_cfg(cfg)
    plot_val_acc_compare(runs)


if __name__ == "__main__":
    # Usage:
    #   python -m compare_val_acc --cfg configs/compare.yaml
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)
