"""False positive evaluation for multi-hot model with Strategy C (per-label thresholds)."""

import sys
sys.path.insert(0, "/home/charles/Programs/datasetManipulation")

import csv
import json
import cv2
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.multihot import decode_multihot_groups
from ultralytics.utils.metrics import box_iou
from ultralytics.data.augment import LetterBox
from torchvision.ops import nms


# --- Defaults ---
DEFAULT_MODEL = "runs/segment/train65/weights/best.pt"
DEFAULT_THRESHOLDS = "runs/segment/train65/strategy_c_thresholds.json"
DEFAULT_IMAGES = "/home/charles/Programs/datasetManipulation/NoLitter-3/train/split/images"

NAMES_17 = {
    0: "alimentaire-papier", 1: "alimentaire-plastique", 2: "autre-bois",
    3: "autre-carton", 4: "autre-metal", 5: "autre-papier-carton",
    6: "autre-plastique-fragments", 7: "autre-polystyrene",
    8: "bouteille-en-plastique", 9: "bouteille-en-verre", 10: "canette",
    11: "encombrant", 12: "megot", 13: "paquet_cigarette",
    14: "sac-ordures-menageres", 15: "textile", 16: "verre",
}


def load_strategy_c_config(path):
    """Load Strategy C thresholds and decode config from JSON."""
    with open(path) as f:
        cfg = json.load(f)

    # Reconstruct label_thresholds as ordered tensor
    label_names = list(cfg["label_thresholds"].keys())
    label_thresh = torch.tensor([cfg["label_thresholds"][n] for n in label_names])

    class_map = torch.tensor(cfg["class_map"], dtype=torch.float)
    decode_groups = cfg["decode_groups"]
    decode_rules = cfg["decode_rules"]
    # JSON turns dict keys to strings; fix material_priority keys back to int
    if "material_priority" in decode_rules:
        decode_rules["material_priority"] = {
            int(k): v for k, v in decode_rules["material_priority"].items()
        }

    return label_thresh, class_map, decode_groups, decode_rules


def predict_multihot(model, img_path, label_thresh, class_map, decode_groups, decode_rules):
    """Run inference with Strategy C pipeline.

    Returns list of dicts with keys: class_id, class_name, confidence, bbox (xyxy).
    """
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        return []
    letterbox = LetterBox(new_shape=640, auto=True, stride=32)
    img = letterbox(image=img0)
    img = img.transpose(2, 0, 1)[::-1]  # HWC->CHW, BGR->RGB
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).unsqueeze(0).float().to(model.device) / 255.0

    # Raw inference
    inner = model.model
    inner.eval()
    with torch.no_grad():
        preds = inner(img_t)

    raw_one2one = preds[1]["one2one"]
    scores_raw = raw_one2one["scores"]  # (1, 14, anchors) pre-sigmoid
    head = inner.model[-1]
    dbox = head._get_decode_boxes(raw_one2one)  # (1, 4, anchors)

    boxes_all = dbox[0].T  # (anchors, 4)
    scores_14 = scores_raw[0].T.sigmoid()  # (anchors, 14)

    cm = class_map.to(model.device)
    lt = label_thresh.to(model.device)

    # Pre-filter top-1000
    max_scores = scores_14.max(dim=1).values
    k = min(1000, boxes_all.shape[0])
    topk_idx = max_scores.topk(k).indices
    boxes_k = boxes_all[topk_idx]
    scores_k = scores_14[topk_idx]
    max_k = max_scores[topk_idx]

    # NMS-merge
    leaders = nms(boxes_k, max_k, 0.6)
    leader_boxes = boxes_k[leaders]
    leader_scores = scores_k[leaders]

    if leaders.shape[0] > 0 and leaders.shape[0] < k:
        iou = box_iou(leader_boxes, boxes_k)
        overlap_mask = iou > 0.6
        scores_exp = scores_k.unsqueeze(0).expand(leaders.shape[0], -1, -1)
        leader_scores = (scores_exp * overlap_mask.unsqueeze(-1)).max(dim=1).values

    # Group-argmax decode with per-label thresholds (Strategy C)
    pred_cls = decode_multihot_groups(
        leader_scores, cm, decode_groups, decode_rules,
        label_thresholds=lt,
    )

    # Keep everything that decoded to a valid class
    keep = pred_cls >= 0
    boxes_f = leader_boxes[keep]
    cls_f = pred_cls[keep]
    scores_f = leader_scores[keep]

    if boxes_f.shape[0] == 0:
        return []

    # Confidence for NMS ordering: min(sigmoid) of active labels
    active_mask = cm[cls_f]
    conf_f = (scores_f * active_mask + (1 - active_mask) * 999.0).min(dim=1).values

    # Class-aware NMS
    boxes_offset = boxes_f + cls_f.float().unsqueeze(1) * 4096
    nms_keep = nms(boxes_offset, conf_f, 0.5)[:300]
    boxes_f = boxes_f[nms_keep]
    cls_f = cls_f[nms_keep]
    conf_f = conf_f[nms_keep]

    # Scale boxes back to original image size
    h0, w0 = img0.shape[:2]
    h1, w1 = img_t.shape[2:]
    gain = min(h1 / h0, w1 / w0)
    pad_x = (w1 - w0 * gain) / 2
    pad_y = (h1 - h0 * gain) / 2
    boxes_f[:, [0, 2]] = ((boxes_f[:, [0, 2]] - pad_x) / gain).clamp(0, w0)
    boxes_f[:, [1, 3]] = ((boxes_f[:, [1, 3]] - pad_y) / gain).clamp(0, h0)

    detections = []
    for i in range(boxes_f.shape[0]):
        cid = cls_f[i].item()
        detections.append({
            "class_id": int(cid),
            "class_name": NAMES_17.get(int(cid), f"class_{int(cid)}"),
            "confidence": conf_f[i].item(),
            "bbox": boxes_f[i].int().tolist(),
        })
    return detections


def eval_fp(model_path, thresholds_path, images_folder):
    """Evaluate false positives using Strategy C pipeline."""
    label_thresh, class_map, decode_groups, decode_rules = load_strategy_c_config(thresholds_path)
    print(f"Loaded Strategy C thresholds from {thresholds_path}")

    model = YOLO(model_path)

    images = sorted(Path(images_folder).glob("*.jpg")) + sorted(Path(images_folder).glob("*.png"))
    print(f"Running Strategy C inference on {len(images)} images...")

    total_fps = 0
    images_with_fps = 0
    fps_per_class = defaultdict(int)
    fps_per_image = []

    save_dir = Path("runs/detect/eval_fp_multihot")
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(images):
        dets = predict_multihot(model, img_path, label_thresh, class_map, decode_groups, decode_rules)
        num_dets = len(dets)

        image_fps = {
            "path": str(img_path),
            "count": num_dets,
            "classes": defaultdict(int),
            "detections": dets,
        }

        if num_dets > 0:
            images_with_fps += 1
            total_fps += num_dets
            for d in dets:
                fps_per_class[d["class_name"]] += 1
                image_fps["classes"][d["class_name"]] += 1

            # Save annotated image
            img0 = cv2.imread(str(img_path))
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{d['class_name']} {d['confidence']:.2f}"
                cv2.putText(img0, label, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(save_dir / img_path.name), img0)

        fps_per_image.append(image_fps)

        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            print(f"  [{i+1}/{len(images)}] FPs so far: {total_fps} in {images_with_fps} images")

    # Print summary
    print(f"\nTotal images: {len(images)}")
    print(f"Images with FPs: {images_with_fps}")
    print(f"Total FPs: {total_fps}")
    print(f"FP rate: {images_with_fps/len(images)*100:.2f}%")
    print(f"\nFPs per class:")
    for class_name, count in sorted(fps_per_class.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")

    # Save CSV
    csv_path = save_dir / "fps_per_image.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Total FPs"] + sorted(fps_per_class.keys()))
        for img_data in fps_per_image:
            row = [img_data["path"], img_data["count"]]
            for cn in sorted(fps_per_class.keys()):
                row.append(img_data["classes"].get(cn, 0))
            writer.writerow(row)

    print(f"\nAnnotated images saved to {save_dir}/")
    print(f"CSV saved to {csv_path}")
    return fps_per_image


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate false positives with Strategy C multi-hot pipeline")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model weights")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Path to strategy_c_thresholds.json")
    parser.add_argument("--images", default=DEFAULT_IMAGES, help="Folder of no-litter images")
    args = parser.parse_args()
    eval_fp(args.model, args.thresholds, args.images)
