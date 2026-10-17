"""Train a single linear layer (one logit) on frozen embeddings.

Usage:
  python train_classifier.py --config config.yaml --encoder image_a            # val only
  python train_classifier.py --config config.yaml --encoder image_a --eval_test # also test AUROC

Hyperparameters are chosen on validation AUROC using selection_seed; the chosen setting is then
trained with every seed in train.seeds. For each run the epoch with best validation AUROC is kept.
Test AUROC is computed only with --eval_test, once the course protocol allows it.
"""
import argparse, copy, itertools, json, os
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score


def load(enc_dir, split):
    z = np.load(os.path.join(enc_dir, f"{split}.npz"))
    return z["ids"], z["features"].astype(np.float32), z["labels"]


def fit(Xtr, ytr, Xva, yva, lr, wd, epochs, bs, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = torch.nn.Linear(Xtr.shape[1], 1)                 # one output logit
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()                      # BCE on logits
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr).float()
    Xv = torch.from_numpy(Xva)
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(Xt[idx]).squeeze(-1), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(Xv).squeeze(-1)).numpy()  # hateful-class score
        auc = roc_auc_score(yva, s)
        if auc > best:
            best, best_state, best_ep = auc, copy.deepcopy(model.state_dict()), ep + 1
    model.load_state_dict(best_state)
    return model, best, best_ep


def scores(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X)).squeeze(-1)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--eval_test", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t = cfg["train"]
    enc_dir = os.path.join(cfg["extract"]["out_dir"], a.encoder)
    _, Xtr, ytr = load(enc_dir, "train")
    vid, Xva, yva = load(enc_dir, "val")
    # standardise with training statistics (same for all encoders)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    print(f"{a.encoder}: train {len(ytr)} (pos {int(ytr.sum())}, neg {int((ytr == 0).sum())}), "
          f"val {len(yva)} (pos {int(yva.sum())}, neg {int((yva == 0).sum())})")

    # hyperparameter search on validation AUROC (single selection seed)
    search = []
    for lr, wd in itertools.product(t["grid"]["lr"], t["grid"]["weight_decay"]):
        _, auc, ep = fit(Xtr, ytr, Xva, yva, lr, wd, t["epochs"], t["batch_size"], t["selection_seed"])
        search.append({"lr": lr, "weight_decay": wd, "val_auroc": auc, "best_epoch": ep})
        print(f"  lr={lr:g} wd={wd:g} val AUROC={auc:.4f} (epoch {ep})")
    best = max(search, key=lambda r: r["val_auroc"])

    out = os.path.join(t["out_dir"], a.encoder)
    os.makedirs(out, exist_ok=True)
    runs = []
    if a.eval_test:
        tid, Xte, yte = load(enc_dir, "test")
        Xte = (Xte - mu) / sd
        assert (yte >= 0).all(), "test labels are not available in test.jsonl"
    for seed in t["seeds"]:
        model, auc, ep = fit(Xtr, ytr, Xva, yva, best["lr"], best["weight_decay"],
                             t["epochs"], t["batch_size"], seed)
        run = {"seed": seed, "val_auroc": auc, "best_epoch": ep}
        if a.eval_test:
            run["test_auroc"] = float(roc_auc_score(yte, scores(model, Xte)))
        runs.append(run)
        torch.save({"state_dict": model.state_dict(), "mean": mu, "std": sd, "hparams": best, "seed": seed},
                   os.path.join(out, f"classifier_seed{seed}.pt"))
        np.savez(os.path.join(out, f"val_scores_seed{seed}.npz"), ids=vid, labels=yva, scores=scores(model, Xva))
        print(f"  seed {seed}: {run}")
    summary = {"encoder": a.encoder, "optimizer": t["optimizer"], "epochs": t["epochs"],
               "batch_size": t["batch_size"], "selected": best, "search": search, "runs": runs,
               "val_mean": float(np.mean([r["val_auroc"] for r in runs])),
               "val_std": float(np.std([r["val_auroc"] for r in runs], ddof=1))}
    if a.eval_test:
        ta = [r["test_auroc"] for r in runs]
        summary.update(test_mean=float(np.mean(ta)), test_std=float(np.std(ta, ddof=1)))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
