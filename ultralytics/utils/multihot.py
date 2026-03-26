# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Multi-hot label decoding utilities for group-argmax class assignment."""

import torch


def decode_multihot_groups(scores, class_map, decode_groups, decode_rules=None,
                           obj_threshold=0.3, label_thresholds=None):
    """Decode multi-hot sigmoid scores to single class IDs using group argmax with rules.

    Strategies:
      - No label_thresholds (Strategy A): raw argmax, flat obj_threshold for presence.
      - With label_thresholds (Strategy C): raw argmax, per-label threshold for both
        material and object presence. No material + non-standalone object → discard.

    Args:
        scores: (N, n_labels) sigmoid scores
        class_map: (n_classes, n_labels) tensor, binary multi-hot matrix
        decode_groups: dict with 'material' and 'object' keys, each a list of label indices
        decode_rules: dict with optional keys:
            - object_priority: list of object label indices that always map to their default class
            - discard_on_invalid: list of object label indices to discard when material is invalid
            - material_priority: dict {mat_label: [allowed_obj_labels]} — materials that override
              object when combo is invalid and object is not priority/standalone
        obj_threshold: minimum max-score for object group to be considered present
            (used only when label_thresholds is None)
        label_thresholds: (n_labels,) tensor of per-label thresholds. When provided,
            presence is checked per-label: material/object must exceed its own threshold.

    Returns:
        class_ids: (N,) tensor of class IDs (0 to n_classes-1), -1 for discarded detections
    """
    if decode_rules is None:
        decode_rules = {}

    device = scores.device
    n_det = scores.shape[0]
    n_classes = class_map.shape[0]

    if n_det == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    mat_labels = decode_groups["material"]  # list of label indices
    obj_labels = decode_groups["object"]  # list of label indices

    # Extract group scores
    mat_idx_t = torch.tensor(mat_labels, device=device)
    obj_idx_t = torch.tensor(obj_labels, device=device)
    mat_scores = scores[:, mat_idx_t]  # (N, n_mat)
    obj_scores = scores[:, obj_idx_t]  # (N, n_obj)

    # Raw argmax for both groups (all strategies)
    mat_best_pos = mat_scores.argmax(dim=-1)  # (N,)
    mat_best_label = mat_idx_t[mat_best_pos]  # (N,) actual label index
    obj_best_pos = obj_scores.argmax(dim=-1)  # (N,)
    obj_best_label = obj_idx_t[obj_best_pos]  # (N,) actual label index

    if label_thresholds is not None:
        # Strategy C: per-label presence for both groups
        lt = label_thresholds.to(device)
        mat_thresh = lt[mat_idx_t]  # (n_mat,)
        obj_thresh = lt[obj_idx_t]  # (n_obj,)

        mat_winner_score = mat_scores.gather(1, mat_best_pos.unsqueeze(1)).squeeze(1)
        mat_winner_thresh = mat_thresh[mat_best_pos]
        mat_present = mat_winner_score >= mat_winner_thresh  # (N,)

        obj_winner_score = obj_scores.gather(1, obj_best_pos.unsqueeze(1)).squeeze(1)
        obj_winner_thresh = obj_thresh[obj_best_pos]
        obj_present = obj_winner_score >= obj_winner_thresh  # (N,)
    else:
        # Strategy A: material always present, flat obj threshold
        mat_present = torch.ones(n_det, dtype=torch.bool, device=device)
        obj_max_score = obj_scores.max(dim=-1).values  # (N,)
        obj_present = obj_max_score > obj_threshold

    # --- Build lookup tables from class_map ---
    combo_to_class = {}  # (mat_label, obj_label) → class_id
    mat_only_class = {}  # mat_label → class_id (for material-only / "autre-*" classes)
    obj_only_class = {}  # obj_label → class_id (for standalone objects)
    valid_mats_for_obj = {}  # obj_label → list of valid mat_labels

    mat_set = set(mat_labels)
    obj_set = set(obj_labels)

    for cls_id in range(n_classes):
        row = class_map[cls_id]
        active_mats = [la for la in mat_labels if row[la] > 0]
        active_objs = [la for la in obj_labels if row[la] > 0]

        if len(active_objs) == 1 and len(active_mats) >= 1:
            for m in active_mats:
                combo_to_class[(m, active_objs[0])] = cls_id
            valid_mats_for_obj.setdefault(active_objs[0], []).extend(active_mats)
        elif len(active_objs) == 1 and len(active_mats) == 0:
            obj_only_class[active_objs[0]] = cls_id
            valid_mats_for_obj.setdefault(active_objs[0], [])
        elif len(active_objs) == 0 and len(active_mats) == 1:
            # Material-only class — first one wins if duplicate (e.g., plastique has class 6 and 7)
            if active_mats[0] not in mat_only_class:
                mat_only_class[active_mats[0]] = cls_id

    # Standalone objects: those with no valid materials in class_map
    standalone_objs = {ol for ol, mats in valid_mats_for_obj.items() if len(mats) == 0}

    # Parse rules
    obj_priority = set(decode_rules.get("object_priority", []))
    discard_on_invalid = set(decode_rules.get("discard_on_invalid", []))
    mat_priority = {}
    for k, v in decode_rules.get("material_priority", {}).items():
        mat_priority[int(k)] = set(v)

    # --- Decode each detection ---
    result = torch.full((n_det,), -1, dtype=torch.long, device=device)

    for i in range(n_det):
        ml = mat_best_label[i].item()
        ol = obj_best_label[i].item()
        has_mat = mat_present[i].item()
        has_obj = obj_present[i].item()

        # Nothing above threshold → discard
        if not has_mat and not has_obj:
            continue

        # No object → material-only class (if material present)
        if not has_obj:
            result[i] = mat_only_class.get(ml, -1) if has_mat else -1
            continue

        # Standalone object (encombrant, megot) → doesn't need material
        if ol in standalone_objs:
            result[i] = obj_only_class.get(ol, -1)
            continue

        # Object present but no material → discard (can't form a valid class)
        if not has_mat:
            # Priority objects can still map to their default
            if ol in obj_priority:
                vm = valid_mats_for_obj.get(ol, [])
                result[i] = combo_to_class.get((vm[0], ol), -1) if vm else obj_only_class.get(ol, -1)
            # Otherwise discard
            continue

        # Both present — try combo
        if (ml, ol) in combo_to_class:
            result[i] = combo_to_class[(ml, ol)]
            continue

        # Priority object → map to default class regardless of material
        if ol in obj_priority:
            vm = valid_mats_for_obj.get(ol, [])
            if vm:
                result[i] = combo_to_class.get((vm[0], ol), -1)
            else:
                result[i] = obj_only_class.get(ol, -1)
            continue

        # Material priority override
        if ml in mat_priority and ol not in mat_priority[ml]:
            result[i] = mat_only_class.get(ml, -1)
            continue

        # Discard on invalid material
        if ol in discard_on_invalid:
            result[i] = -1
            continue

        # Fallback — pick best valid material for this object
        vm = valid_mats_for_obj.get(ol, [])
        if vm:
            vm_scores = scores[i, vm]
            best_vm = vm[vm_scores.argmax().item()]
            result[i] = combo_to_class.get((best_vm, ol), -1)
        else:
            result[i] = -1

    return result


def postprocess_group_nms(
    boxes, scores_raw, class_map, decode_groups, decode_rules=None,
    max_pre=2000, merge_iou=0.6, nms_iou=0.5, conf_thresh=0.01,
    max_det=300, obj_threshold=0.3,
):
    """Smart NMS in group-argmax space with score merging across overlapping anchors.

    Pipeline:
      1. Pre-filter to top max_pre anchors by max sigmoid score
      2. Greedy merge: cluster overlapping boxes (IoU > merge_iou), take element-wise
         max of 14D scores within each cluster → merged scores capture complementary
         signals from nearby anchors
      3. Group-argmax decode on merged scores → class + confidence
      4. Class-aware NMS in 17-class space (IoU > nms_iou)

    Args:
        boxes: (N, 4) decoded xyxy boxes for ALL anchors
        scores_raw: (N, 14) raw scores (pre-sigmoid) for all anchors
        class_map: (n_classes, n_labels) binary multi-hot matrix
        decode_groups: dict with 'material' and 'object' keys
        decode_rules: dict with rules for group-argmax
        max_pre: keep top-K anchors by max sigmoid before merging
        merge_iou: IoU threshold for merging overlapping proposals
        nms_iou: IoU threshold for final class-aware NMS
        conf_thresh: minimum confidence for output detections
        max_det: maximum number of output detections
        obj_threshold: threshold for object group in group-argmax

    Returns:
        dict with keys:
            boxes: (M, 4) xyxy boxes
            scores_14: (M, 14) merged sigmoid scores
            cls: (M,) class IDs (0 to n_classes-1, -1 filtered out)
            conf: (M,) confidence = min(sigmoid[active labels]) for predicted class
    """
    from ultralytics.utils.metrics import box_iou

    scores_14 = scores_raw.sigmoid()  # (N, 14)
    n_anchors = boxes.shape[0]

    # --- Step 1: Pre-filter to top-K by max sigmoid ---
    max_scores = scores_14.max(dim=1).values  # (N,)
    k = min(max_pre, n_anchors)
    topk_idx = max_scores.topk(k).indices
    boxes_k = boxes[topk_idx]  # (K, 4)
    scores_k = scores_14[topk_idx]  # (K, 14)
    max_scores_k = max_scores[topk_idx]  # (K,)

    # --- Step 2: Greedy merge of overlapping proposals ---
    order = max_scores_k.argsort(descending=True)
    boxes_k = boxes_k[order]
    scores_k = scores_k[order]

    merged_boxes = []
    merged_scores = []
    used = torch.zeros(k, dtype=torch.bool, device=boxes.device)

    # Compute pairwise IoU once
    iou_matrix = box_iou(boxes_k, boxes_k)  # (K, K)

    for i in range(k):
        if used[i]:
            continue
        # Find overlapping boxes not yet used
        overlap = (iou_matrix[i] > merge_iou) & ~used
        overlap[i] = True  # include self

        # Merge: element-wise max of 14D scores in cluster
        cluster_scores = scores_k[overlap]  # (C, 14)
        merged_score = cluster_scores.max(dim=0).values  # (14,)

        merged_boxes.append(boxes_k[i])
        merged_scores.append(merged_score)
        used |= overlap

    if not merged_boxes:
        empty = torch.zeros(0, device=boxes.device)
        return {"boxes": empty.reshape(0, 4), "scores_14": empty.reshape(0, 14),
                "cls": empty.long(), "conf": empty}

    merged_boxes = torch.stack(merged_boxes)  # (M, 4)
    merged_scores = torch.stack(merged_scores)  # (M, 14)

    # --- Step 3: Group-argmax decode ---
    pred_cls = decode_multihot_groups(
        merged_scores, class_map, decode_groups, decode_rules, obj_threshold=obj_threshold,
    )

    # Compute confidence: min sigmoid of active labels for predicted class
    safe_cls = pred_cls.clamp(min=0)
    active_mask = class_map[safe_cls]  # (M, 14)
    masked = merged_scores * active_mask + (1 - active_mask) * 2.0
    conf = masked.min(dim=1).values
    conf[pred_cls < 0] = 0

    # Filter by confidence and discard
    keep = (conf >= conf_thresh) & (pred_cls >= 0)
    merged_boxes = merged_boxes[keep]
    merged_scores = merged_scores[keep]
    pred_cls = pred_cls[keep]
    conf = conf[keep]

    if merged_boxes.shape[0] == 0:
        empty = torch.zeros(0, device=boxes.device)
        return {"boxes": empty.reshape(0, 4), "scores_14": empty.reshape(0, 14),
                "cls": empty.long(), "conf": empty}

    # --- Step 4: Class-aware NMS ---
    # Offset boxes by class to make class-aware NMS from class-agnostic
    nc = class_map.shape[0]
    class_offset = pred_cls.float() * 4096  # large offset per class
    boxes_offset = merged_boxes + class_offset.unsqueeze(1)

    from torchvision.ops import nms
    nms_keep = nms(boxes_offset, conf, nms_iou)
    nms_keep = nms_keep[:max_det]

    return {
        "boxes": merged_boxes[nms_keep],
        "scores_14": merged_scores[nms_keep],
        "cls": pred_cls[nms_keep],
        "conf": conf[nms_keep],
    }
