"""Create the ROC figure used in the report.

Usage: python make_figures.py --config config.yaml
Output: report/figs/roc.pdf (validation ROC curves of the seed 0 classifiers).
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

CURVES = [("image_a", "Image A (CLIP)", "-", "0.0"),
          ("image_b", "Image B (DINOv2)", "--", "0.35"),
          ("text_a", "Text A (MPNet)", "-.", "0.0"),
          ("text_b", "Text B (RoBERTa)", ":", "0.35")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    runs = yaml.safe_load(open(a.config))["train"]["out_dir"]
    os.makedirs("report/figs", exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for enc, name, ls, color in CURVES:
        z = np.load(os.path.join(runs, enc, "val_scores_seed0.npz"))
        fpr, tpr, _ = roc_curve(z["labels"], z["scores"])
        auc = roc_auc_score(z["labels"], z["scores"])
        ax.plot(fpr, tpr, ls=ls, color=color, lw=1.2, label=f"{name}, {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("report/figs/roc.pdf", metadata={"CreationDate": None})
    print("wrote report/figs/roc.pdf")


if __name__ == "__main__":
    main()
