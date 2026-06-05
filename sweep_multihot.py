"""Multi-hot (yolo30) checkpoint sweep + threshold/metric/inference-strategy analysis.

Foundation layer: run a checkpoint ONCE at a low conf floor, decode candidate
detections with a pluggable cluster-aggregation strategy, match to GT, and cache
the rows. All downstream analysis (selection, per-class thresholds, 17/object/material
confusion matrices, strategy comparison) reads the cache offline.

Inspired by datasetManipulation/evalFalsePositives.py (eval_fp / threshold_analysis)
and eval_fp_multihot.py (the multi-hot inference pipeline).
"""

import sys
sys.path.insert(0, "/home/charles/Programs/datasetManipulation")

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.multihot import decode_multihot_groups

# --- Defaults ---
DATA_ROOT = "/home/charles/Programs/datasetManipulation/datasets/Dataset-ViPARE-33-split"
VAL_IMAGES = f"{DATA_ROOT}/valid/images"
NEG_IMAGES = "/home/charles/Programs/datasetManipulation/NoLitter-3/train/split/images"
RUN_DIR = "/home/charles/Programs/ultralytics-fork/runs/segment/train-5"

NAMES_17 = [
    "alimentaire-papier", "alimentaire-plastique", "autre-bois", "autre-carton",
    "autre-metal", "autre-papier-carton", "autre-plastique-fragments", "autre-polystyrene",
    "bouteille-en-plastique", "bouteille-en-verre", "canette", "encombrant", "megot",
    "paquet_cigarette", "sac-ordures-menageres", "textile", "verre",
]


# ----------------------------------------------------------------------------
# Config + projection tables (17-class -> object label / material label)
# ----------------------------------------------------------------------------
def get_decode_cfg(model):
    """Pull multi-hot decode config from the model yaml.

    Returns class_map (17,15) tensor, decode_groups, decode_rules, names15 dict,
    and projection arrays cls17->object_label and cls17->material_label (-1 = none).
    """
    y = model.model.yaml
    cm = torch.tensor(y["class_map"], dtype=torch.float)
    dg = y["decode_groups"]
    dr = dict(y.get("decode_rules", {}))
    if "material_priority" in dr:  # JSON/yaml may stringify keys
        dr["material_priority"] = {int(k): v for k, v in dr["material_priority"].items()}
    names15 = y["names"]

    obj_idx = dg["object"]
    mat_idx = dg["material"]
    n_cls = cm.shape[0]
    cls2obj = np.full(n_cls, -1, dtype=int)
    cls2mat = np.full(n_cls, -1, dtype=int)
    for c in range(n_cls):
        row = cm[c]
        objs = [o for o in obj_idx if row[o] > 0]
        mats = [m for m in mat_idx if row[m] > 0]
        if objs:
            cls2obj[c] = objs[0]
        if mats:
            cls2mat[c] = mats[0]
    return cm, dg, dr, names15, cls2obj, cls2mat


# ----------------------------------------------------------------------------
# Inference: decode candidate detections with pluggable cluster aggregation
# ----------------------------------------------------------------------------
def infer_candidates(model, img_path, cm, dg, dr, *, strategy="mean",
                     merge_iou=0.6, nms_iou=0.5, conf_floor=0.05,
                     max_pre=1000, max_det=300):
    """Decode candidate detections for one image.

    strategy controls how the 15D vectors of an overlapping (class-agnostic)
    cluster are aggregated before group-argmax decode:
      - "leader": use the top box's own vector (no info sharing)
      - "max":    element-wise max over the cluster (manufactures conjunctions)
      - "mean":   confidence-weighted mean over the cluster (rewards agreement)

    Returns dict of numpy arrays: boxes (N,4 xyxy orig px), cls (N,), conf (N,),
    scores15 (N,15).
    """
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        return _empty_cands()
    lb = LetterBox(new_shape=640, auto=True, stride=32)
    img = lb(image=img0).transpose(2, 0, 1)[::-1]
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).unsqueeze(0).float().to(model.device) / 255.0

    inner = model.model
    inner.eval()
    with torch.no_grad():
        preds = inner(img_t)
    raw = preds[1]["one2one"]
    scores_raw = raw["scores"]            # (1, 15, anchors)
    head = inner.model[-1]
    dbox = head._get_decode_boxes(raw)    # (1, 4, anchors)

    boxes = dbox[0].T                     # (A, 4)
    scores15 = scores_raw[0].T.sigmoid()  # (A, 15)
    cm = cm.to(boxes.device)

    # Pre-filter to top-K by max sigmoid
    maxs = scores15.max(dim=1).values
    k = min(max_pre, boxes.shape[0])
    top = maxs.topk(k).indices
    boxes_k, scores_k, max_k = boxes[top], scores15[top], maxs[top]

    # Class-agnostic cluster leaders
    leaders = nms(boxes_k, max_k, merge_iou)
    lead_boxes = boxes_k[leaders]
    if lead_boxes.shape[0] == 0:
        return _empty_cands()

    if strategy == "leader":
        lead_scores = scores_k[leaders]
    else:
        iou = box_iou(lead_boxes, boxes_k)        # (L, K)
        overlap = iou > merge_iou                 # (L, K)
        if strategy == "max":
            exp = scores_k.unsqueeze(0) * overlap.unsqueeze(-1)
            lead_scores = exp.max(dim=1).values
        elif strategy == "mean":
            w = (max_k.unsqueeze(0) * overlap).unsqueeze(-1)  # conf-weighted
            lead_scores = (scores_k.unsqueeze(0) * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
        else:
            raise ValueError(f"unknown strategy {strategy}")

    # Group-argmax decode -> 17-class
    cls = decode_multihot_groups(lead_scores, cm, dg, dr, obj_threshold=0.3)

    # Confidence = min sigmoid of active labels for predicted class
    safe = cls.clamp(min=0)
    active = cm[safe]
    conf = (lead_scores * active + (1 - active) * 2.0).min(dim=1).values
    conf[cls < 0] = 0.0

    keep = (cls >= 0) & (conf >= conf_floor)
    lead_boxes, lead_scores, cls, conf = lead_boxes[keep], lead_scores[keep], cls[keep], conf[keep]
    if lead_boxes.shape[0] == 0:
        return _empty_cands()

    # Class-aware NMS (offset boxes per class)
    off = lead_boxes + cls.float().unsqueeze(1) * 4096
    nk = nms(off, conf, nms_iou)[:max_det]
    lead_boxes, lead_scores, cls, conf = lead_boxes[nk], lead_scores[nk], cls[nk], conf[nk]

    # Scale boxes back to original image
    h0, w0 = img0.shape[:2]
    h1, w1 = img_t.shape[2:]
    gain = min(h1 / h0, w1 / w0)
    px, py = (w1 - w0 * gain) / 2, (h1 - h0 * gain) / 2
    b = lead_boxes.clone()
    b[:, [0, 2]] = ((b[:, [0, 2]] - px) / gain).clamp(0, w0)
    b[:, [1, 3]] = ((b[:, [1, 3]] - py) / gain).clamp(0, h0)

    # Raw group argmaxes (what the object / material heads said for each detection)
    obj_idx = torch.tensor(dg["object"], device=lead_scores.device)
    mat_idx = torch.tensor(dg["material"], device=lead_scores.device)
    os_ = lead_scores[:, obj_idx]
    ms_ = lead_scores[:, mat_idx]
    obj_lbl = obj_idx[os_.argmax(1)]
    mat_lbl = mat_idx[ms_.argmax(1)]

    return {
        "boxes": b.cpu().numpy(),
        "cls": cls.cpu().numpy().astype(int),
        "conf": conf.cpu().numpy(),
        "scores15": lead_scores.cpu().numpy(),
        "obj": obj_lbl.cpu().numpy().astype(int),
        "obj_conf": os_.max(1).values.cpu().numpy(),
        "mat": mat_lbl.cpu().numpy().astype(int),
        "mat_conf": ms_.max(1).values.cpu().numpy(),
    }


def _empty_cands():
    return {"boxes": np.zeros((0, 4)), "cls": np.zeros(0, int), "conf": np.zeros(0),
            "scores15": np.zeros((0, 15)), "obj": np.zeros(0, int), "obj_conf": np.zeros(0),
            "mat": np.zeros(0, int), "mat_conf": np.zeros(0)}


# ----------------------------------------------------------------------------
# Ground-truth loading + class-agnostic matching
# ----------------------------------------------------------------------------
def load_gt(label_path, w, h):
    """Load YOLO GT as xyxy pixel boxes + class ids. Handles detection (4 coords)
    and segmentation polygon (>=6 coords -> bbox) label formats."""
    p = Path(label_path)
    if not p.exists():
        return np.zeros((0, 4)), np.zeros(0, int)
    boxes, cls = [], []
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
        if len(coords) == 4:
            xc, yc, bw, bh = coords
            x1, y1, x2, y2 = (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h
        else:
            xs, ys = coords[0::2], coords[1::2]
            x1, y1, x2, y2 = min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h
        boxes.append([x1, y1, x2, y2])
        cls.append(c)
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(cls, dtype=int)


def match(cand_boxes, gt_boxes, iou_thr=0.5):
    """Greedy class-agnostic IoU matching. Returns matched_gt_for_cand (len N_cand,
    gt index or -1) and set of matched gt indices."""
    n_c, n_g = len(cand_boxes), len(gt_boxes)
    matched = np.full(n_c, -1, dtype=int)
    if n_c == 0 or n_g == 0:
        return matched, set()
    iou = box_iou(torch.tensor(cand_boxes, dtype=torch.float),
                  torch.tensor(gt_boxes, dtype=torch.float)).numpy()  # (N_c, N_g)
    pairs = np.argwhere(iou >= iou_thr)
    if len(pairs):
        order = iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]
        used_c, used_g = set(), set()
        for ci, gi in pairs[order]:
            if ci not in used_c and gi not in used_g:
                used_c.add(ci); used_g.add(gi); matched[ci] = gi
        return matched, used_g
    return matched, set()


# ----------------------------------------------------------------------------
# Candidate collection (shared by sweep / thresholds)
# ----------------------------------------------------------------------------
def collect(model, images_dir, cm, dg, dr, *, strategy="mean", conf_floor=0.05, with_gt=True, limit=None):
    """Run a split and return flat per-candidate arrays.

    Returns dict: cls (N,), conf (N,), correct (N,) [1 if matched GT of same class],
    gt_count (17,), n_images, img_with_det.
    """
    cls_all, conf_all, corr_all = [], [], []
    gt_count = np.zeros(17, dtype=int)
    n_images = img_with_det = 0
    imgs = sorted(Path(images_dir).glob("*.jpg"))
    if limit:
        imgs = imgs[:limit]
    for ip in imgs:
        n_images += 1
        c = infer_candidates(model, ip, cm, dg, dr, strategy=strategy, conf_floor=conf_floor)
        nc = len(c["cls"])
        if nc:
            img_with_det += 1
        if with_gt:
            img0 = cv2.imread(str(ip)); h, w = img0.shape[:2]
            lp = str(ip).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            gtb, gtc = load_gt(lp, w, h)
            for g in gtc:
                gt_count[g] += 1
            m, _ = match(c["boxes"], gtb)
            corr = np.array([1 if (m[i] >= 0 and c["cls"][i] == gtc[m[i]]) else 0 for i in range(nc)], int)
        else:
            corr = np.zeros(nc, int)
        cls_all.append(c["cls"]); conf_all.append(c["conf"]); corr_all.append(corr)
    cat = lambda L: np.concatenate(L) if L else np.zeros(0)
    return {"cls": cat(cls_all).astype(int), "conf": cat(conf_all), "correct": cat(corr_all).astype(int),
            "gt_count": gt_count, "n_images": n_images, "img_with_det": img_with_det}


def macro_f1(val, thr=None):
    """Per-17-class P/R/F1 and macro-F1 from collected val candidates.
    thr: optional (17,) per-class conf thresholds; else use all collected candidates."""
    cls, conf, corr, gt = val["cls"], val["conf"], val["correct"], val["gt_count"]
    f1s, rec_sum, tp_sum = [], 0, 0
    for c in range(17):
        sel = cls == c
        if thr is not None:
            sel = sel & (conf >= thr[c])
        preds = int(sel.sum()); tp = int(corr[sel].sum())
        p = tp / preds if preds else 0.0
        r = tp / gt[c] if gt[c] else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if gt[c] > 0:
            f1s.append(f1)
        tp_sum += tp; rec_sum += gt[c]
    return (sum(f1s) / len(f1s) if f1s else 0.0), (tp_sum / rec_sum if rec_sum else 0.0)


# ----------------------------------------------------------------------------
# Sweep: rank checkpoints (best epoch selection)
# ----------------------------------------------------------------------------
def sweep(run_dir, epochs=None, strategy="mean", conf=0.25, out_csv=None):
    """Run every checkpoint on val+neg, report macro-F1 / recall / NoLitter FP-rate, rank."""
    wdir = Path(run_dir) / "weights"
    if epochs is None:
        epochs = sorted(int(p.stem[5:]) for p in wdir.glob("epoch*.pt"))
    out_csv = Path(out_csv or Path(run_dir) / "sweep_results.csv")
    rows = []
    print(f"{'epoch':>5s}{'macroF1':>9s}{'recall':>8s}{'negFP%':>8s}")
    for e in epochs:
        ck = wdir / f"epoch{e}.pt"
        if not ck.exists():
            continue
        model = YOLO(str(ck))
        cm, dg, dr, *_ = get_decode_cfg(model)
        val = collect(model, VAL_IMAGES, cm, dg, dr, strategy=strategy, conf_floor=conf, with_gt=True)
        neg = collect(model, NEG_IMAGES, cm, dg, dr, strategy=strategy, conf_floor=conf, with_gt=False)
        mf1, rec = macro_f1(val)
        fpr = 100 * neg["img_with_det"] / max(neg["n_images"], 1)
        rows.append((e, mf1, rec, fpr))
        print(f"{e:>5d}{mf1:>9.3f}{rec:>8.3f}{fpr:>8.1f}")
        del model
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "macroF1", "recall", "negFP_pct"]); w.writerows(rows)
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\nsaved {out_csv}")
    print("Top-5 by macro-F1 (then inspect FP%):")
    for e, mf1, rec, fpr in rows[:5]:
        print(f"  epoch{e:<3d}  F1={mf1:.3f}  R={rec:.3f}  negFP={fpr:.1f}%")
    return rows


# ----------------------------------------------------------------------------
# Thresholds: per-17-class conf threshold, F-beta on combined precision (val + neg)
# ----------------------------------------------------------------------------
def optimize_thresholds(ckpt, strategy="mean", beta=0.5, w_neg=1.0, conf_floor=0.03, out=None):
    """Per-class threshold maximizing F-beta of recall vs combined precision
    (precision penalizes both val mis-classes and NoLitter FPs, weighted by w_neg)."""
    import json
    model = YOLO(ckpt)
    cm, dg, dr, *_ = get_decode_cfg(model)
    val = collect(model, VAL_IMAGES, cm, dg, dr, strategy=strategy, conf_floor=conf_floor, with_gt=True)
    neg = collect(model, NEG_IMAGES, cm, dg, dr, strategy=strategy, conf_floor=conf_floor, with_gt=False)

    thr = np.full(17, 1.0)  # default: suppress classes with no usable signal
    print(f"{'class':<28s}{'thr':>7s}{'P':>7s}{'R':>7s}{'Fb':>7s}{'GT':>6s}")
    for c in range(17):
        vsel = val["cls"] == c
        vconf, vcorr = val["conf"][vsel], val["correct"][vsel]
        nconf = neg["conf"][neg["cls"] == c]
        gt = val["gt_count"][c]
        if gt == 0 or len(vconf) == 0:
            print(f"{NAMES_17[c]:<28s}{1.0:>7.3f}{'-':>7s}{'-':>7s}{'-':>7s}{int(gt):>6d}")
            continue
        cand = np.unique(np.concatenate([vconf, [0.0]]))
        best = (-1.0, 1.0, 0.0, 0.0)  # fb, thr, p, r
        for t in cand:
            tp = int(vcorr[vconf >= t].sum())
            fp_val = int((vconf >= t).sum()) - tp
            fp_neg = int((nconf >= t).sum())
            denom = tp + fp_val + w_neg * fp_neg
            p = tp / denom if denom > 0 else 0.0
            r = tp / gt
            fb = (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if (beta ** 2 * p + r) > 0 else 0.0
            if fb > best[0]:
                best = (fb, float(t), p, r)
        thr[c] = best[1]
        print(f"{NAMES_17[c]:<28s}{best[1]:>7.3f}{best[3]:>7.3f}{best[2]:>7.3f}{best[0]:>7.3f}{int(gt):>6d}")

    out = Path(out or Path(ckpt).parent.parent / f"thresholds_{Path(ckpt).stem}.json")
    json.dump({"names": NAMES_17, "thresholds": thr.tolist(), "strategy": strategy,
               "beta": beta, "w_neg": w_neg}, open(out, "w"), indent=2)
    print(f"\nsaved {out}")
    return thr


# ----------------------------------------------------------------------------
# Confusion matrices (17-class / object / material) + per-granularity P/R/F1
# ----------------------------------------------------------------------------
def _plot_cm(C, row_labels, col_labels, title, path, normalize=True):
    """Save a confusion-matrix heatmap. Rows = true, cols = pred (last = background)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = C.astype(float)
    if normalize:
        M = M / M.sum(axis=1, keepdims=True).clip(min=1e-9)  # row-normalised (recall view)
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.7), max(5, len(row_labels) * 0.6)))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1 if normalize else M.max())
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            v = int(C[i, j])
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  saved {path}")


def _prf_table(C, names, title, skip=()):
    """Print per-row P/R/F1 from a confusion matrix (last row/col = background)."""
    n = len(names)
    print(f"\n{title}")
    print(f"{'class':<26s}{'TP':>6s}{'FP':>6s}{'FN':>6s}{'P':>8s}{'R':>8s}{'F1':>8s}{'GT':>7s}")
    macro = []
    for c in range(n):
        if c in skip:
            continue
        tp = C[c, c]
        gt = C[c, :].sum()           # all true-c (incl. predicted-bg = missed)
        pred = C[:, c].sum()         # all predicted-c (incl. true-bg = FP)
        fn = gt - tp; fp = pred - tp
        p = tp / pred if pred else 0.0
        r = tp / gt if gt else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if gt > 0:
            macro.append(f1)
        print(f"{names[c]:<26s}{int(tp):>6d}{int(fp):>6d}{int(fn):>6d}{p:>8.3f}{r:>8.3f}{f1:>8.3f}{int(gt):>7d}")
    print(f"{'MACRO-F1 (GT>0)':<26s}{'':>6s}{'':>6s}{'':>6s}{'':>8s}{'':>8s}{(sum(macro)/len(macro) if macro else 0):>8.3f}")


def evaluate(ckpt, strategy="mean", conf=0.10, n_val=None, out_dir=None, thresholds=None):
    """Run full val + neg, build 17/object/material confusion matrices, P/R/F1, plots.

    thresholds: optional path to thresholds_*.json (per-class conf). When given it
    overrides the global conf (each candidate kept iff conf >= thr[its class])."""
    model = YOLO(ckpt)
    cm, dg, dr, names15, cls2obj, cls2mat = get_decode_cfg(model)

    thr = None
    if thresholds is not None:
        import json
        thr = np.array(json.load(open(thresholds))["thresholds"], dtype=float)
        conf = float(min(thr.min(), conf))  # infer floor = lowest per-class thr
        print(f"using per-class thresholds from {thresholds} (floor={conf:.3f})")
    obj_labels, mat_labels = dg["object"], dg["material"]
    n_obj, n_mat = len(obj_labels), len(mat_labels)
    mat_pos = {l: i for i, l in enumerate(mat_labels)}  # label -> 0..6; none->n_mat

    out_dir = Path(out_dir or f"runs/detect/sweep_eval_{Path(ckpt).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ckpt={ckpt}  strategy={strategy}  conf>={conf}  out={out_dir}")

    C17 = np.zeros((18, 18))                    # 17 + background
    Cobj = np.zeros((n_obj + 1, n_obj + 1))     # 8 + background
    Cmat = np.zeros((n_mat + 2, n_mat + 2))     # 7 materials + none + background
    BG17, BGobj, BGmat = 17, n_obj, n_mat + 1
    NONE_MAT = n_mat

    def omat(cls17):
        o = cls2obj[cls17]
        m = cls2mat[cls17]
        return o, (mat_pos[m] if m >= 0 else NONE_MAT)

    val_imgs = sorted(Path(VAL_IMAGES).glob("*.jpg"))
    if n_val:
        val_imgs = val_imgs[:n_val]
    for k, ip in enumerate(val_imgs):
        img0 = cv2.imread(str(ip)); h, w = img0.shape[:2]
        lp = str(ip).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        gt_boxes, gt_cls = load_gt(lp, w, h)
        c = infer_candidates(model, ip, cm, dg, dr, strategy=strategy, conf_floor=conf)
        if thr is not None and len(c["cls"]):
            keep = c["conf"] >= thr[c["cls"]]
            c = {k: v[keep] for k, v in c.items()}
        m, used_g = match(c["boxes"], gt_boxes)
        for ci, gi in enumerate(m):
            p = c["cls"][ci]
            if gi >= 0:                          # matched: true=gt, pred=p
                g = gt_cls[gi]
                C17[g, p] += 1
                og, mg = omat(g); op, mp = omat(p)
                Cobj[og, op] += 1; Cmat[mg, mp] += 1
            else:                                # FP: true=background, pred=p
                C17[BG17, p] += 1
                op, mp = omat(p)
                Cobj[BGobj, op] += 1; Cmat[BGmat, mp] += 1
        for gi in range(len(gt_cls)):            # missed GT: true=g, pred=background
            if gi not in used_g:
                g = gt_cls[gi]
                C17[g, BG17] += 1
                og, mg = omat(g)
                Cobj[og, BGobj] += 1; Cmat[mg, BGmat] += 1
        if (k + 1) % 200 == 0:
            print(f"  val {k+1}/{len(val_imgs)}")

    # NoLitter false positives (separate, like eval_fp)
    neg_imgs = sorted(Path(NEG_IMAGES).glob("*.jpg"))
    neg_fp_cls = np.zeros(17, dtype=int)
    neg_imgs_with_fp = 0
    for ip in neg_imgs:
        c = infer_candidates(model, ip, cm, dg, dr, strategy=strategy, conf_floor=conf)
        if thr is not None and len(c["cls"]):
            keep = c["conf"] >= thr[c["cls"]]
            c = {k: v[keep] for k, v in c.items()}
        if len(c["cls"]):
            neg_imgs_with_fp += 1
            for p in c["cls"]:
                neg_fp_cls[p] += 1

    # --- Plots ---
    labels17 = NAMES_17 + ["background"]
    labels_obj = [names15[l] for l in obj_labels] + ["background"]
    labels_mat = [names15[l] for l in mat_labels] + ["none", "background"]
    _plot_cm(C17, labels17, labels17, f"17-class CM ({Path(ckpt).stem}, {strategy})", out_dir / "cm_17class.png")
    _plot_cm(Cobj, labels_obj, labels_obj, f"Object CM ({Path(ckpt).stem}, {strategy})", out_dir / "cm_object.png")
    _plot_cm(Cmat, labels_mat, labels_mat, f"Material CM ({Path(ckpt).stem}, {strategy})", out_dir / "cm_material.png")

    # --- Tables ---
    _prf_table(C17, labels17, "=== Per 17-class ===", skip=(BG17,))
    _prf_table(Cobj, labels_obj, "=== Per object ===", skip=(BGobj,))
    _prf_table(Cmat, labels_mat, "=== Per material ===", skip=(NONE_MAT, BGmat))

    print(f"\n=== NoLitter FP (conf>={conf}) ===")
    print(f"Images: {len(neg_imgs)}  with FP: {neg_imgs_with_fp}  "
          f"FP-rate: {100*neg_imgs_with_fp/max(len(neg_imgs),1):.1f}%  total FP: {int(neg_fp_cls.sum())}")
    for c in np.argsort(neg_fp_cls)[::-1]:
        if neg_fp_cls[c]:
            print(f"  {NAMES_17[c]:<28s}{int(neg_fp_cls[c]):>5d}")
    return out_dir


# ----------------------------------------------------------------------------
# Predict: annotated inference on a few images (visual check, uses thresholds)
# ----------------------------------------------------------------------------
def predict(ckpt, sources, thresholds=None, strategy="mean", conf=0.25, n=5, out_dir=None):
    """Run multi-hot inference on a few images and save annotated copies.

    sources: list of folders and/or image files. For a folder, the first `n` images
    are taken. thresholds: path to thresholds_*.json (per-class); else global `conf`.
    """
    import json
    model = YOLO(ckpt)
    cm, dg, dr, names15, *_ = get_decode_cfg(model)

    thr = None
    floor = conf
    if thresholds:
        thr = np.array(json.load(open(thresholds))["thresholds"], dtype=float)
        floor = float(min(thr.min(), conf))
    out = Path(out_dir or "runs/detect/predict_check")
    out.mkdir(parents=True, exist_ok=True)

    imgs = []
    for s in sources:
        p = Path(s)
        if p.is_dir():
            imgs += (sorted(p.glob("*.jpg")) + sorted(p.glob("*.png")))[:n]
        elif p.exists():
            imgs.append(p)
    print(f"ckpt={Path(ckpt).name}  strategy={strategy}  "
          f"{'thresholds='+Path(thresholds).name if thr is not None else 'conf='+str(conf)}  "
          f"images={len(imgs)}")

    for ip in imgs:
        c = infer_candidates(model, ip, cm, dg, dr, strategy=strategy, conf_floor=floor)
        if thr is not None and len(c["cls"]):
            keep = c["conf"] >= thr[c["cls"]]
            c = {k: v[keep] for k, v in c.items()}
        img0 = cv2.imread(str(ip))
        print(f"  {ip.name}  ({len(c['cls'])} dets)")
        for i in range(len(c["cls"])):
            x1, y1, x2, y2 = c["boxes"][i].astype(int)
            onm, osc = names15[int(c["obj"][i])], c["obj_conf"][i]
            mnm, msc = names15[int(c["mat"][i])], c["mat_conf"][i]
            cls17, cf = NAMES_17[c["cls"][i]], c["conf"][i]
            l1 = f"{onm}:{osc:.2f} {mnm}:{msc:.2f}"   # object + material (15D heads)
            l2 = f"{cls17} {cf:.2f}"                   # decoded 17-class
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img0, l1, (x1, max(y1 - 20, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            cv2.putText(img0, l2, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            print(f"      {onm}:{osc:.2f} + {mnm}:{msc:.2f}  ->  {cls17} {cf:.2f}")
        cv2.imwrite(str(out / ip.name), img0)
    print(f"\nsaved annotated images to {out}/")
    return out


# ----------------------------------------------------------------------------
# Smoke test: foundation plumbing on the current checkpoint
# ----------------------------------------------------------------------------
def smoke(ckpt, n_val=8, strategy="mean"):
    model = YOLO(ckpt)
    cm, dg, dr, names15, cls2obj, cls2mat = get_decode_cfg(model)
    print(f"ckpt={ckpt}  labels={cm.shape[1]}  classes={cm.shape[0]}  strategy={strategy}")
    print(f"cls2obj={cls2obj.tolist()}")
    print(f"cls2mat={cls2mat.tolist()}")

    val_imgs = sorted(Path(VAL_IMAGES).glob("*.jpg"))[:n_val]
    tot_tp = tot_fp = tot_gt = 0
    for ip in val_imgs:
        img0 = cv2.imread(str(ip)); h, w = img0.shape[:2]
        lp = str(ip).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        gt_boxes, gt_cls = load_gt(lp, w, h)
        c = infer_candidates(model, ip, cm, dg, dr, strategy=strategy)
        m, used_g = match(c["boxes"], gt_boxes)
        tp = sum(1 for ci, gi in enumerate(m) if gi >= 0 and c["cls"][ci] == gt_cls[gi])
        fp = len(c["cls"]) - sum(1 for gi in m if gi >= 0)
        tot_tp += tp; tot_fp += fp; tot_gt += len(gt_cls)
        print(f"  {ip.name:40s} dets={len(c['cls']):3d}  gt={len(gt_cls):3d}  "
              f"clsTP={tp:3d}  FP={fp:3d}")
    print(f"\nVAL[{n_val}]  class-correct TP={tot_tp}  FP={tot_fp}  GT={tot_gt}")

    neg_imgs = sorted(Path(NEG_IMAGES).glob("*.jpg"))[:n_val]
    neg_fp = sum(len(infer_candidates(model, ip, cm, dg, dr, strategy=strategy)["cls"]) for ip in neg_imgs)
    print(f"NEG[{n_val}]  total FP detections={neg_fp}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="yolo30 multi-hot sweep / thresholds / metrics")
    ap.add_argument("--mode", default="eval", choices=["smoke", "sweep", "thresholds", "eval", "predict"])
    ap.add_argument("--ckpt", default=f"{RUN_DIR}/weights/best.pt", help="checkpoint (thresholds/eval/smoke)")
    ap.add_argument("--run", default=RUN_DIR, help="run dir (sweep)")
    ap.add_argument("--strategy", default="mean", choices=["leader", "max", "mean"])
    ap.add_argument("--conf", type=float, default=0.25, help="global conf (sweep ranking, eval w/o thresholds)")
    ap.add_argument("--thresholds", default=None, help="thresholds_*.json (eval)")
    ap.add_argument("--beta", type=float, default=0.5, help="F-beta (<1 favors precision)")
    ap.add_argument("--w-neg", type=float, default=1.0, dest="w_neg", help="NoLitter FP weight in precision")
    ap.add_argument("--epochs", default=None, help="sweep epochs, e.g. '85-99' or '0,50,99'")
    ap.add_argument("--n", type=int, default=8, help="smoke #imgs; predict #imgs per source")
    ap.add_argument("--images", nargs="+", default=None, help="predict: folders and/or image files")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    def parse_epochs(s):
        if not s:
            return None
        if "-" in s:
            lo, hi = map(int, s.split("-")); return list(range(lo, hi + 1))
        return [int(x) for x in s.split(",")]

    if a.mode == "smoke":
        smoke(a.ckpt, a.n, a.strategy)
    elif a.mode == "sweep":
        sweep(a.run, parse_epochs(a.epochs), a.strategy, a.conf, a.out)
    elif a.mode == "thresholds":
        optimize_thresholds(a.ckpt, a.strategy, a.beta, a.w_neg, out=a.out)
    elif a.mode == "predict":
        srcs = a.images or [NEG_IMAGES, VAL_IMAGES]
        predict(a.ckpt, srcs, thresholds=a.thresholds, strategy=a.strategy, conf=a.conf, n=a.n, out_dir=a.out)
    else:
        evaluate(a.ckpt, a.strategy, a.conf, out_dir=a.out, thresholds=a.thresholds)
