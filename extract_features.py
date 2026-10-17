"""Extract frozen embeddings for one encoder on train/val/test.

Usage: python extract_features.py --config config.yaml --encoder image_a [--time_budget 150]
Output: features/<encoder>/<split>.npz with ids, features, labels (labels = -1 if absent).

Extraction is resumable: embeddings are written in chunks to features/<encoder>/parts/, so the
script can be stopped and restarted (or run with --time_budget on a machine with short job limits).
Rerun the same command until it prints ALL SPLITS DONE.
"""
import argparse, json, os, time
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, CLIPModel

CHUNK = 256


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_tensor(x):
    # get_image_features can return a model output object instead of a tensor
    if isinstance(x, torch.Tensor):
        return x
    return x.pooler_output if getattr(x, "pooler_output", None) is not None else x[0]


def build(enc, device):
    ckpt = enc["checkpoint"]
    if enc["family"] == "clip":
        model = CLIPModel.from_pretrained(ckpt).to(device).eval()
        fn = lambda px: as_tensor(model.get_image_features(pixel_values=px))
        proc = AutoImageProcessor.from_pretrained(ckpt)
    elif enc["family"] == "dinov2":
        model = AutoModel.from_pretrained(ckpt).to(device).eval()
        fn = lambda px: model(pixel_values=px).pooler_output  # CLS token after final layer norm
        proc = AutoImageProcessor.from_pretrained(ckpt)
    else:
        model = AutoModel.from_pretrained(ckpt).to(device).eval()
        fn, proc = None, AutoTokenizer.from_pretrained(ckpt)
    commit = getattr(model.config, "_commit_hash", None)
    if commit is None:
        try:
            from huggingface_hub import model_info
            commit = model_info(ckpt).sha
        except Exception:
            pass
    return model, fn, proc, commit


@torch.no_grad()
def embed_images(rows, cfg, fn, proc, device):
    d, bs, out = cfg["data"], cfg["extract"]["batch_size"], []
    for i in range(0, len(rows), bs):
        # pixels only, encoder specific resizing and normalisation, no transcript
        imgs = [Image.open(os.path.join(d["image_root"], r[d["image_field"]])).convert("RGB")
                for r in rows[i:i + bs]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        out.append(fn(px).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_text(rows, cfg, enc, model, tok, device):
    texts = [r[cfg["data"]["text_field"]] for r in rows]
    bs, out = cfg["extract"]["batch_size"], []
    for i in range(0, len(texts), bs):
        t = tok(texts[i:i + bs], padding=True, truncation=True,
                max_length=enc["max_length"], return_tensors="pt").to(device)
        h = model(**t).last_hidden_state
        m = t["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--time_budget", type=float, default=0, help="seconds; 0 means no limit")
    a = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(open(a.config))
    enc = cfg["encoders"][a.encoder]
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = os.path.join(cfg["extract"]["out_dir"], a.encoder)
    part_dir = os.path.join(out_dir, "parts")
    os.makedirs(part_dir, exist_ok=True)
    d = cfg["data"]
    model = None
    all_done = True
    for split, fname in d["splits"].items():
        final = os.path.join(out_dir, f"{split}.npz")
        if os.path.exists(final):
            continue
        rows = read_jsonl(os.path.join(d["data_dir"], fname))
        n_chunks = (len(rows) + CHUNK - 1) // CHUNK
        for c in range(n_chunks):
            p = os.path.join(part_dir, f"{split}_{c:04d}.npy")
            if os.path.exists(p):
                continue
            if a.time_budget and time.time() - t0 > a.time_budget:
                all_done = False
                break
            if model is None:
                model, fn, proc, commit = build(enc, device)
                print("device:", device, "commit:", commit, flush=True)
                json.dump({"encoder": a.encoder, **enc, "resolved_commit": commit},
                          open(os.path.join(out_dir, "encoder_info.json"), "w"), indent=2)
            chunk = rows[c * CHUNK:(c + 1) * CHUNK]
            f = (embed_images(chunk, cfg, fn, proc, device) if enc["modality"] == "image"
                 else embed_text(chunk, cfg, enc, model, proc, device))
            assert f.shape[1] == enc["dim"], (f.shape, enc["dim"])
            np.save(p + ".tmp.npy", f)
            os.replace(p + ".tmp.npy", p)
            print(f"{a.encoder} {split} chunk {c + 1}/{n_chunks} ({time.time() - t0:.0f}s)", flush=True)
        if not all(os.path.exists(os.path.join(part_dir, f"{split}_{c:04d}.npy")) for c in range(n_chunks)):
            all_done = False
            break
        feats = np.concatenate([np.load(os.path.join(part_dir, f"{split}_{c:04d}.npy")) for c in range(n_chunks)])
        ids = np.array([str(r[d["id_field"]]) for r in rows])
        labels = np.array([int(r.get(d["label_field"], -1)) for r in rows])
        assert len(feats) == len(rows)
        np.savez(final, ids=ids, features=feats, labels=labels)
        print(a.encoder, split, feats.shape, "pos", int((labels == 1).sum()), "neg", int((labels == 0).sum()),
              "unlabelled", int((labels == -1).sum()), flush=True)
    print("ALL SPLITS DONE" if all_done else "NOT FINISHED, rerun the same command", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 1)
    main()
