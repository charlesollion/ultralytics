# Multi-hot (yolo30) — Train → Sweep → Thresholds → Stats workflow

The `yolo30` model decomposes the 17 flat litter classes into **15 multi-hot labels**
(8 objects incl. `autre`, 7 materials). Labels on disk stay 0–16; the model predicts
the 15 labels and a group-argmax decoder maps them back to the 17 classes at inference.
See `ultralytics/cfg/models/26/yolo30-seg.yaml` for the `class_map` / `decode_groups` /
`decode_rules`.

All post-training analysis lives in **`sweep_multihot.py`** (repo root). It reuses the
multi-hot inference of `eval_fp_multihot.py` and the FP-benchmark idea of
`datasetManipulation/evalFalsePositives.py`.

> Activate the env first (every command below assumes it):
> ```bash
> source ~/Programs/.venv/bin/activate
> ```

---

## 0. Train

```python
from ultralytics import YOLO
model = YOLO("yolo30n-seg.yaml")            # 15-label multi-hot + autre
model.train(
    data="/home/charles/Programs/datasetManipulation/datasets/Dataset-ViPARE-33-split/data.yaml",
    epochs=100, batch=16,
    augment=True, shear=10, degrees=30, crop_fraction=0.9, scale=0.5,
    cos_lr=True,
    close_mosaic=10,     # mosaic OFF for the last 10 epochs (texture calibration → lower FP)
    save_period=1,       # keep every checkpoint for the sweep
)
```

Notes:
- **No `class_weights`** for the baseline. If used, they are now **per-label (15)**, not
  per-class, and must be mean≈1 to avoid inflating cls-loss / overfit
  (e.g. `[0.8,1,1.3,1.2,1,0.8,1.3,1, 1,1,1,1,1,1.3,1.3]`, objects then materials).
- In-training validation is **loss-only** (`val/cls_loss` is the overfit signal). Real
  evaluation is the sweep below.

Output: `runs/segment/train-N/weights/epoch{0..99}.pt` (+ `best.pt` = lowest val loss).

---

## 1. Sweep checkpoints → pick the best epoch

Ranks each checkpoint by 17-class macro-F1 + recall + NoLitter FP-rate (at a fixed global
conf). Only the post-mosaic-close epochs are usually worth ranking (here 90–99):

```bash
python sweep_multihot.py --mode sweep \
    --run runs/segment/train-5 \
    --epochs 90-99 \
    --conf 0.25
```

- `--epochs` accepts a range (`90-99`) or list (`0,50,99`); omit for **all** epochs.
- Writes `runs/segment/train-5/sweep_results.csv` and prints the top-5.
- **Pick the epoch** trading macro-F1 against `negFP%` (don't just take max F1 — check FP).

> Runtime ≈ ~2 min/epoch (full val + NoLitter). For all 100 epochs run it in the
> background / overnight, or stride the early epochs.

---

## 2. Optimize per-class thresholds (val **and** FP)

For the chosen epoch, find a per-17-class confidence threshold that maximizes **F-β**
of recall vs **combined precision** — precision penalizes both val mis-classifications
and NoLitter false positives (weighted by `--w-neg`). `β<1` favors precision.

```bash
python sweep_multihot.py --mode thresholds \
    --ckpt runs/segment/train-5/weights/epoch95.pt \
    --beta 0.5 --w-neg 1.0
```

- Writes `runs/segment/train-5/thresholds_epoch95.json` (17 id-aligned values).
- Classes with no usable signal auto-suppress to `thr = 1.0`.
- Tune `--beta` (lower = more precision/less FP) and `--w-neg` (higher = punish NoLitter
  FPs harder) to move along the precision/recall trade-off.

---

## 3. Build stats + confusion matrices

Runs full val + NoLitter and emits the three confusion matrices and per-granularity
P/R/F1, using the per-class thresholds from step 2:

```bash
python sweep_multihot.py --mode eval \
    --ckpt runs/segment/train-5/weights/epoch95.pt \
    --thresholds runs/segment/train-5/thresholds_epoch95.json
```

Outputs to `runs/detect/sweep_eval_epoch95/`:
- `cm_17class.png`  — 17×17 (+background) confusion matrix
- `cm_object.png`   — 8 objects (autre…sac) +background
- `cm_material.png` — 7 materials + none + background
- console: **per-17-class**, **per-object**, **per-material** P/R/F1 (+ macro-F1)
- console: **NoLitter FP rate** and per-class FP counts

Drop `--thresholds` to use a single global `--conf` instead (quick look).

**Read the object vs material CM together:** if 17-class errors are mostly *within-material*
(e.g. canette↔autre-metal) the material CM stays clean while the object CM smears — and
vice-versa. That tells you which axis is the bottleneck and which axis the inference
decoder should arbitrate (see step 4).

---

## 4. (Optional) Choose the inference / dead-angle strategy

When two overlapping predictions decode to different classes for one object, class-agnostic
NMS would keep one box and drop the other's evidence. The decoder instead aggregates the
overlapping cluster's 15D vectors before group-argmax. Compare the three aggregations by
re-running step 3 with `--strategy`:

```bash
for S in leader max mean; do
  python sweep_multihot.py --mode eval \
      --ckpt runs/segment/train-5/weights/epoch95.pt \
      --thresholds runs/segment/train-5/thresholds_epoch95.json \
      --strategy $S --out runs/detect/eval_$S
done
```

- `leader` — decode the top box only (no info sharing)
- `max`    — element-wise max over the cluster (can manufacture false conjunctions)
- `mean`   — confidence-weighted mean over the cluster (**default**, rewards agreement)

Use the same strategy in steps 1–3 that you intend to deploy.

---

## Inference on a few images (visual check)

Run the deployment decode on a handful of images (e.g. some NoLitter false-positive
cases + a few from validation), with the per-class thresholds, and save annotated copies:

```bash
python sweep_multihot.py --mode predict \
    --ckpt runs/segment/train-5/weights/epoch95.pt \
    --thresholds runs/segment/train-5/thresholds_epoch95.json \
    --images \
      /home/charles/Programs/datasetManipulation/NoLitter-3/train/split/images \
      /home/charles/Programs/datasetManipulation/datasets/Dataset-ViPARE-33-split/valid/images \
    --n 6 \
    --out runs/detect/predict_check
```

- `--images` takes any mix of **folders and/or single image files**; for a folder the
  first `--n` images are used. Omit `--images` to default to NoLitter + valid.
- Drop `--thresholds` to use a single global `--conf` instead.
- `--strategy {leader,max,mean}` picks the cluster-aggregation (default `mean`).

Each detection is shown at **both granularities** — the 15D heads and the decoded class:

```
  <image>.jpg  (2 dets)
      bouteille:0.77 + plastique:0.87  ->  bouteille-en-plastique 0.77
      autre:0.39 + plastique:0.36      ->  autre-plastique-fragments 0.36
```

Annotated images saved to `--out`: orange line = `object:score material:score`,
red line = `decoded-17class conf`. (For standalone objects megot/encombrant the material
score is meaningless — they decode with no material.)

---

## General order after a training run

1. Let training finish (100 epochs, `save_period=1`).
2. **Sweep** post-close epochs → choose best epoch (F1 vs FP%).  → `--mode sweep`
3. **Thresholds** on that epoch (tune `beta`/`w_neg`).            → `--mode thresholds`
4. **Eval** with thresholds → CMs + per-granularity P/R/F1 + FP.  → `--mode eval`
5. Inspect object vs material CM → pick inference strategy.        → `--mode eval --strategy`
6. Ship the chosen `epochN.pt` + `thresholds_epochN.json`; deployment decode = the same
   group-argmax + per-class thresholds (`eval_fp_multihot.py` is the reference impl).

## Quick reference

| step | mode | key flags | output |
|---|---|---|---|
| select best epoch | `sweep` | `--run --epochs --conf` | `sweep_results.csv` |
| per-class thresholds | `thresholds` | `--ckpt --beta --w-neg` | `thresholds_*.json` |
| stats + matrices | `eval` | `--ckpt --thresholds --strategy` | `runs/detect/sweep_eval_*/` |
| visual inference | `predict` | `--ckpt --thresholds --images --n` | annotated imgs + console |
| plumbing check | `smoke` | `--ckpt --n` | console |

Default paths (override with flags): run `runs/segment/train-5`, val
`Dataset-ViPARE-33-split/valid/images`, NoLitter `NoLitter-3/train/split/images`.
