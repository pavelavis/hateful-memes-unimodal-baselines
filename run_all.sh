#!/usr/bin/env bash
# Runs the whole assignment. Usage: bash run_all.sh   (from inside hateful_memes_baselines/)
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
ENC="image_a image_b text_a text_b"
for e in $ENC; do python extract_features.py --config config.yaml --encoder "$e"; done
EVAL=""
if python - <<'EOF'
import json, yaml
c = yaml.safe_load(open("config.yaml"))["data"]
rows = [json.loads(l) for l in open(f"{c['data_dir']}/{c['splits']['test']}") if l.strip()]
raise SystemExit(0 if all(c["label_field"] in r for r in rows) else 1)
EOF
then EVAL="--eval_test"; echo "test labels found, computing test AUROC"; else echo "no test labels, skipping test AUROC"; fi
for e in $ENC; do python train_classifier.py --config config.yaml --encoder "$e" $EVAL | tee "runs_$e.log"; done
python evaluate.py --config config.yaml
python make_figures.py --config config.yaml
zip -qr results.zip runs/*/summary.json features/*/encoder_info.json report runs_*.log
echo "DONE. Summary written to results.zip"
