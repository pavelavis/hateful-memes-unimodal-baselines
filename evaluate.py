"""Collect results, write the LaTeX tables, and select error analysis examples.

Usage: python evaluate.py --config config.yaml
Writes report/results_table.tex and report/error_candidates.csv.
"""
import argparse, csv, json, os
import numpy as np
import yaml

NAMES = {"image_a": "Image A", "image_b": "Image B", "text_a": "Text A", "text_b": "Text B"}


def tex(s):
    return s.replace("_", r"\_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    runs_dir, out = cfg["train"]["out_dir"], "report"
    os.makedirs(out, exist_ok=True)

    rows = []
    for k, label in NAMES.items():
        p = os.path.join(runs_dir, k, "summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        ckpt = tex(cfg["encoders"][k]["checkpoint"])
        te = f"{s['test_mean']:.3f} $\\pm$ {s['test_std']:.3f}" if "test_mean" in s else "n/a"
        rows.append(f"{label} ({ckpt}) & {s['val_mean']:.3f} $\\pm$ {s['val_std']:.3f} & {te} \\\\")
    with open(os.path.join(out, "results_table.tex"), "w") as f:
        f.write("\\begin{tabular}{lcc}\n\\toprule\nBaseline (checkpoint) & Val AUROC & Test AUROC \\\\\n\\midrule\n")
        f.write("\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # error analysis: validation examples, threshold fixed at 0.5, seed 0 classifiers
    an = cfg["analysis"]
    thr = an["threshold"]
    si = np.load(os.path.join(runs_dir, an["best_image"], "val_scores_seed0.npz"))
    st = np.load(os.path.join(runs_dir, an["best_text"], "val_scores_seed0.npz"))
    assert (si["ids"] == st["ids"]).all()
    pi, pt = si["scores"], st["scores"]
    disagree = np.where((pi >= thr) != (pt >= thr))[0]
    # prioritise disagreements, largest score gap first, with both true classes represented:
    # ceil(n/2) hateful and floor(n/2) non-hateful examples
    y = si["labels"]
    order = disagree[np.argsort(-np.abs(pi - pt)[disagree])]
    n = an["n_examples"]
    pos = [i for i in order if y[i] == 1][: (n + 1) // 2]
    neg = [i for i in order if y[i] == 0][: n - len(pos)]
    pick = np.array(sorted(pos + neg, key=lambda i: -abs(pi[i] - pt[i])), dtype=int)
    assert len(pick) == n, "not enough disagreement examples"
    texts = {}
    d = cfg["data"]
    for line in open(os.path.join(d["data_dir"], d["splits"]["val"]), encoding="utf-8"):
        r = json.loads(line)
        texts[str(r[d["id_field"]])] = (r.get(d["text_field"], ""), r.get(d["image_field"], ""))
    with open(os.path.join(out, "error_candidates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "image_score", "text_score", "text", "image_path"])
        for i in pick:
            sid = str(si["ids"][i])
            w.writerow([sid, int(si["labels"][i]), f"{pi[i]:.3f}", f"{pt[i]:.3f}", texts[sid][0], texts[sid][1]])
    print("selected", [str(si["ids"][i]) for i in pick], "threshold", thr)


if __name__ == "__main__":
    main()
