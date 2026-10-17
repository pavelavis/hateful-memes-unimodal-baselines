# Unimodal baselines for hateful memes

Repository: https://github.com/pavelavis/hateful-memes-unimodal-baselines

Four independent experiments on frozen embeddings (Image A, Image B, Text A, Text B), each with a
single linear layer (one logit) trained with binary cross-entropy on logits. Hateful = 1,
non-hateful = 0. Primary metric: AUROC from sigmoid scores.

| Name | Checkpoint (Hugging Face) | Commit | Dim | Pooling |
|---|---|---|---|---|
| image_a | openai/clip-vit-base-patch16 (image tower) | 57c216476eefef5ab752ec549e440a49ae4ae5f3 | 512 | projected pooled output |
| image_b | facebook/dinov2-base | f9e44c814b77203eaa57a6bdbbd535f21ede1415 | 768 | CLS token after final layer norm |
| text_a | sentence-transformers/all-mpnet-base-v2 | e8c3b32edf5434bc2275fc9bab85f82640a19130 | 768 | masked mean pooling, max length 128 |
| text_b | FacebookAI/roberta-base | e2da8e2f811d1448a5b465c236feacd80ffbac7b | 768 | masked mean pooling, max length 128 |

To pin the exact weights, pass `revision=<commit>` to `from_pretrained`.

## Folder layout

    data/                      course data (train.jsonl, dev.jsonl, test.jsonl, img/), next to this folder
    hateful_memes_baselines/
      config.yaml              all settings (paths, encoders, grid, seeds)
      extract_features.py      frozen encoder embeddings -> features/<encoder>/<split>.npz
      train_classifier.py      grid search + 3 seeds -> runs/<encoder>/
      evaluate.py              results table + error analysis candidates -> report/
      make_figures.py          ROC figure -> report/figs/roc.pdf
      run_all.sh               runs everything
      features/                extracted embeddings and encoder_info.json
      runs/                    classifier checkpoints, validation scores, summary.json
      report/                  2 page report in the Interspeech template (report.tex, refs.bib, Interspeech.cls)

## Setup

    pip install -r requirements.txt

Python 3.10 or newer. Feature extraction runs on CPU or GPU.
matplotlib is only needed for `make_figures.py`.

## Commands

    # 1. feature extraction (resumable; rerun until it prints ALL SPLITS DONE)
    for e in image_a image_b text_a text_b; do python extract_features.py --config config.yaml --encoder $e; done

    # 2. classifier training: 12 point grid on validation AUROC (seed 0), then seeds 0, 1, 2
    for e in image_a image_b text_a text_b; do python train_classifier.py --config config.yaml --encoder $e; done

    # 3. test AUROC (only possible if test.jsonl has labels and the course protocol allows it)
    for e in image_a image_b text_a text_b; do python train_classifier.py --config config.yaml --encoder $e --eval_test; done

    # 4. results table, error analysis candidates and ROC figure
    python evaluate.py --config config.yaml
    python make_figures.py --config config.yaml

    # 5. report (Interspeech template)
    cd report && pdflatex report && bibtex report && pdflatex report && pdflatex report

Or simply `bash run_all.sh`. `--time_budget <seconds>` on step 1 stops extraction cleanly after
that time, for machines with short job limits.

## Settings

Adam, batch size 256, up to 100 epochs with the epoch of best validation AUROC kept, L2 weight decay,
features standardised with training statistics. Grid (identical for all encoders): learning rate in
{1e-4, 1e-3, 1e-2} and weight decay in {0, 1e-4, 1e-2, 1e-1}. Selection seed 0, final seeds 0, 1, 2.
Selected for all four encoders: lr 1e-2, weight decay 1e-1.

Each `runs/<encoder>/classifier_seed<k>.pt` holds the linear layer weights, the standardisation
mean and std, and the selected hyperparameters. `runs/<encoder>/summary.json` holds the full grid
search, per seed validation AUROC and best epochs.

## Results (validation, mean and sample std over seeds 0, 1, 2)

| Baseline | Val AUROC | Test AUROC |
|---|---|---|
| Image A (CLIP ViT-B/16) | 0.657 ± 0.004 | n/a (test labels not provided) |
| Image B (DINOv2-base) | 0.571 ± 0.002 | n/a |
| Text A (all-mpnet-base-v2) | 0.650 ± 0.002 | n/a |
| Text B (roberta-base) | 0.621 ± 0.002 | n/a |

Error analysis uses threshold 0.5 on seed 0 scores of Image A and Text A (see `report/error_candidates.csv`).

## Acknowledgements

Model weights from Hugging Face (OpenAI CLIP, Meta DINOv2, sentence-transformers, FacebookAI RoBERTa).
Code uses PyTorch, Transformers and scikit-learn. The pipeline scripts were written for this project.
