# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, mask_iou


class SegmentationValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on a segmentation model.

    This validator handles the evaluation of segmentation models, processing both bounding box and mask predictions to
    compute metrics such as mAP for both detection and segmentation tasks.

    Attributes:
        plot_masks (list): List to store masks for plotting.
        process (callable): Function to process masks based on save_json and save_txt flags.
        args (SimpleNamespace): Arguments for the validator.
        metrics (SegmentMetrics): Metrics calculator for segmentation tasks.
        stats (dict): Dictionary to store statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationValidator
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml")
        >>> validator = SegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch of images for YOLO segmentation validation.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch.
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics and select mask processing function based on save_json flag.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        if self.args.save_json:
            check_requirements("faster-coco-eval>=1.6.7")
        # More accurate vs faster
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        # Multi-hot class mapping: decode 14D predictions back to 17-class space
        # yaml may live on model directly or on model.model (when wrapped in AutoBackend)
        yaml_cfg = getattr(model, "yaml", None) or getattr(getattr(model, "model", None), "yaml", None) or {}
        class_primary = yaml_cfg.get("class_primary")
        class_map = yaml_cfg.get("class_map")
        if class_primary is not None and class_map is not None:
            self.class_primary = torch.tensor(class_primary, dtype=torch.long, device=self.device)
            self.class_map_t = torch.tensor(class_map, dtype=torch.float, device=self.device)  # (17, 14)
            # Keep ref to the actual model (unwrap AutoBackend if needed) for _get_decode_boxes
            self.model = getattr(model, "model", model)
            # Predictions are decoded to n_old_classes — set nc/names to match
            nc_orig = self.class_map_t.shape[0]
            if self.nc != nc_orig:
                self.nc = nc_orig
                # If trainer already set 17-class names, keep them; otherwise generate defaults
                if len(self.names) != nc_orig:
                    self.names = {i: f"class_{i}" for i in range(nc_orig)}
                self.metrics.names = self.names
                from ultralytics.utils.metrics import ConfusionMatrix
                self.confusion_matrix = ConfusionMatrix(
                    names=self.names, save_matches=self.args.plots and self.args.visualize
                )
            # Multi-label accumulators: list of (pred_scores_14, gt_multihot_14) per detection
            self.ml_pred_scores = []  # each: (n_det, n_labels) sigmoid scores
            self.ml_gt_matched = []  # each: (n_det, n_labels) multi-hot GT for matched det (zeros if unmatched)
            self.ml_gt_unmatched = []  # each: (n_unmatched, n_labels) multi-hot GT for unmatched GTs
            self.ml_label_names = yaml_cfg.get("names", {i: f"label_{i}" for i in range(self.class_map_t.shape[1])})
            # Group-argmax decode config
            self.decode_groups = yaml_cfg.get("decode_groups")
            self.decode_rules = yaml_cfg.get("decode_rules", {})
            # Also accumulate GT class IDs (original 0-16 space) for group-decode confusion matrix
            self.ml_gt_cls_matched = []  # (n_det,) original class ID per matched pred (-1 if unmatched)
            self.ml_gt_cls_unmatched = []  # (n_unmatched,) original class ID for unmatched GTs
            self.ml_det_conf = []  # (n_det,) detection confidence from head
            # Store original 17-class names for the group-decode confusion matrix
            self._orig_names = dict(self.names)  # copy before any override
            LOGGER.info(
                f"Validator: multi-hot decode {self.class_map_t.shape[1]}D -> {nc_orig} classes"
            )
        else:
            self.class_primary = None
            self.class_map_t = None

    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Post-process YOLO predictions and return output detections with proto.

        Args:
            preds (list[torch.Tensor]): Raw predictions from the model.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed detection predictions with masks.
        """
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        # Decode multi-hot 14D scores to 17-class scores for metrics in original class space
        if getattr(self, "class_map_t", None) is not None and isinstance(preds[1], dict) and "one2one" in preds[1]:
            preds = self._postprocess_multihot(preds)
        else:
            preds = super().postprocess(preds[0])
        imgsz = [4 * x for x in proto.shape[2:]]  # get image size from proto
        for i, pred in enumerate(preds):
            coefficient = pred.pop("extra")
            pred["masks"] = (
                self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
        return preds

    def _postprocess_multihot(self, preds):
        """Decode 14D multi-hot predictions using merge+group-argmax NMS pipeline.

        Pipeline per image:
          1. Decode ALL anchor boxes and 14D sigmoid scores
          2. Pre-filter to top max_pre anchors by max sigmoid
          3. Greedy NMS-merge: use low-IoU NMS to find cluster leaders, then scatter-max
             14D scores from all overlapping anchors onto each leader
          4. Group-argmax decode on merged scores → class + confidence
          5. Class-aware NMS in 17-class space
        """
        from torchvision.ops import nms

        from ultralytics.utils.metrics import box_iou
        from ultralytics.utils.multihot import decode_multihot_groups

        raw_one2one = preds[1]["one2one"]
        scores_raw = raw_one2one["scores"]  # (bs, 14, anchors) pre-sigmoid

        # Decode boxes for ALL anchors
        head = None
        if hasattr(self, 'model') and self.model is not None:
            head = self.model.model[-1] if hasattr(self.model, 'model') else None
        if head is None or not hasattr(head, '_get_decode_boxes'):
            raise RuntimeError("Cannot decode boxes: head._get_decode_boxes not available")

        dbox = head._get_decode_boxes(raw_one2one)  # (bs, 4, anchors)
        mc_all = raw_one2one.get("mask_coefficient")  # (bs, nm, anchors) or None

        bs = scores_raw.shape[0]
        max_pre = 1000
        merge_iou = 0.6
        nms_iou = 0.5
        max_det = self.args.max_det
        cm = self.class_map_t

        results = []
        for xi in range(bs):
            boxes_i = dbox[xi].T  # (anchors, 4)
            scores_14 = scores_raw[xi].T.sigmoid()  # (anchors, 14)
            mc_i = mc_all[xi].T if mc_all is not None else None  # (anchors, nm)
            n_anchors = boxes_i.shape[0]

            # Step 1: Pre-filter top-K by max sigmoid
            max_scores = scores_14.max(dim=1).values
            k = min(max_pre, n_anchors)
            topk_idx = max_scores.topk(k).indices
            boxes_k = boxes_i[topk_idx]
            scores_k = scores_14[topk_idx]
            max_k = max_scores[topk_idx]
            mc_k = mc_i[topk_idx] if mc_i is not None else None

            # Step 2: NMS to find cluster leaders, then merge scores from overlapping anchors
            leaders = nms(boxes_k, max_k, merge_iou)  # indices of leaders in boxes_k
            leader_boxes = boxes_k[leaders]  # (L, 4)
            leader_scores = scores_k[leaders]  # (L, 14)
            leader_mc = mc_k[leaders] if mc_k is not None else None

            # Merge: for each leader, max-pool 14D scores from all overlapping anchors
            if leaders.shape[0] > 0 and leaders.shape[0] < k:
                iou = box_iou(leader_boxes, boxes_k)  # (L, K)
                overlap_mask = iou > merge_iou  # (L, K) bool
                # Scatter-max: for each leader, take element-wise max of overlapping scores
                # Use broadcasting: (L, K, 1) * (1, K, 14) with masked fill
                scores_expanded = scores_k.unsqueeze(0).expand(leaders.shape[0], -1, -1)  # (L, K, 14)
                scores_expanded = scores_expanded * overlap_mask.unsqueeze(-1)  # zero non-overlapping
                leader_scores = scores_expanded.max(dim=1).values  # (L, 14)
                # Ensure leader's own scores are included (already are via overlap_mask diagonal)

            m_boxes = leader_boxes
            m_scores = leader_scores
            m_mc = leader_mc

            if m_boxes.shape[0] == 0:
                device = boxes_i.device
                nl = scores_14.shape[1]
                nm = mc_i.shape[1] if mc_i is not None else 32
                results.append(self._empty_result(device, nl, nm))
                continue

            # Step 3: Group-argmax decode
            pred_cls = decode_multihot_groups(
                m_scores, cm, self.decode_groups, self.decode_rules, obj_threshold=0.3,
            )

            # Confidence: min sigmoid of active labels for predicted class
            safe_cls = pred_cls.clamp(min=0)
            active_mask = cm[safe_cls]
            conf = (m_scores * active_mask + (1 - active_mask) * 2.0).min(dim=1).values
            conf[pred_cls < 0] = 0

            # Filter
            keep = (conf >= self.args.conf) & (pred_cls >= 0)
            m_boxes, m_scores, pred_cls, conf = m_boxes[keep], m_scores[keep], pred_cls[keep], conf[keep]
            if m_mc is not None:
                m_mc = m_mc[keep]

            if m_boxes.shape[0] == 0:
                device = boxes_i.device
                nl = scores_14.shape[1]
                nm = mc_i.shape[1] if mc_i is not None else 32
                results.append(self._empty_result(device, nl, nm))
                continue

            # Step 4: Class-aware NMS
            boxes_offset = m_boxes + pred_cls.float().unsqueeze(1) * 4096
            nms_keep = nms(boxes_offset, conf, nms_iou)[:max_det]

            results.append({
                "bboxes": m_boxes[nms_keep],
                "conf": conf[nms_keep],
                "cls": pred_cls[nms_keep].float(),
                "extra": m_mc[nms_keep] if m_mc is not None else torch.zeros(nms_keep.shape[0], 32, device=boxes_i.device),
                "scores_multilabel": m_scores[nms_keep],
            })
        return results

    @staticmethod
    def _empty_result(device, n_labels, n_mask_coeffs):
        """Return an empty result dict for images with no detections."""
        return {
            "bboxes": torch.zeros(0, 4, device=device),
            "conf": torch.zeros(0, device=device),
            "cls": torch.zeros(0, device=device),
            "extra": torch.zeros(0, n_mask_coeffs, device=device),
            "scores_multilabel": torch.zeros(0, n_labels, device=device),
        }

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for validation by processing images and targets.

        Args:
            si (int): Sample index within the batch.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        prepared_batch = super()._prepare_batch(si, batch)
        nl = prepared_batch["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in prepared_batch["imgsz"]]
            if masks.shape[1:] != mask_size:
                masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0]
                masks = masks.gt_(0.5)
        prepared_batch["masks"] = masks
        return prepared_batch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing predictions with keys like 'cls' and 'masks'.
            batch (dict[str, Any]): Dictionary containing batch data with keys like 'cls' and 'masks'.

        Returns:
            (dict[str, np.ndarray]): A dictionary containing correct prediction matrices including 'tp_m' for mask IoU.

        Examples:
            >>> preds = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> batch = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> correct_preds = validator._process_batch(preds, batch)

        Notes:
            - This method computes IoU between predicted and ground truth masks.
            - Overlapping masks are handled based on the overlap_mask argument setting.
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_m = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            iou = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1).float())  # float, uint8
            tp_m = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_m": tp_m})  # update tp with mask IoU
        return tp

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update metrics, including multi-label stats when class_map is active."""
        if getattr(self, "class_map_t", None) is not None and not self.training:
            # Only accumulate multi-label stats for standalone val, not during training
            for si, pred in enumerate(preds):
                self._accumulate_multilabel_stats(si, pred, batch)
        super().update_metrics(preds, batch)

    def _accumulate_multilabel_stats(self, si, pred, batch):
        """Accumulate per-label stats for one image via class-agnostic IoU matching."""
        from ultralytics.utils.metrics import box_iou

        scores_ml = pred.get("scores_multilabel")  # (n_det, 14) or None
        if scores_ml is None:
            return

        n_labels = self.class_map_t.shape[1]

        # GT for this image
        idx = batch["batch_idx"] == si
        gt_cls_old = batch["cls"][idx].squeeze(-1).long()
        gt_bboxes = batch["bboxes"][idx]
        n_gt = gt_cls_old.shape[0]
        n_det = scores_ml.shape[0]

        if n_gt == 0 and n_det == 0:
            return

        gt_multihot = self.class_map_t[gt_cls_old] if n_gt > 0 else torch.zeros(0, n_labels)

        if n_det == 0:
            self.ml_gt_unmatched.append(gt_multihot.cpu())
            self.ml_gt_cls_unmatched.append(gt_cls_old.cpu())
            return

        if n_gt == 0:
            self.ml_pred_scores.append(scores_ml.cpu())
            self.ml_det_conf.append(pred["conf"].cpu())
            self.ml_gt_matched.append(torch.zeros(n_det, n_labels))
            self.ml_gt_cls_matched.append(torch.full((n_det,), -1, dtype=torch.long))
            return

        # Scale GT bboxes to image coords (same as _prepare_batch)
        imgsz = batch["img"].shape[2:]
        gt_bboxes_xyxy = ops.xywh2xyxy(gt_bboxes) * torch.tensor(imgsz, device=gt_bboxes.device)[[1, 0, 1, 0]]

        # Class-agnostic box IoU matching at IoU >= 0.5
        iou = box_iou(gt_bboxes_xyxy, pred["bboxes"])  # (n_gt, n_det)
        iou_np = iou.cpu().numpy()
        matches = np.nonzero(iou_np >= 0.5)
        matches = np.array(matches).T
        matched_gt = set()
        matched_det = set()
        gt_for_det = {}
        if matches.shape[0]:
            order = iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]
            matches = matches[order]
            for gi, di in matches:
                if gi not in matched_gt and di not in matched_det:
                    matched_gt.add(gi)
                    matched_det.add(di)
                    gt_for_det[di] = gi

        gt_matched = torch.zeros(n_det, n_labels)
        for di, gi in gt_for_det.items():
            gt_matched[di] = gt_multihot[gi]

        self.ml_pred_scores.append(scores_ml.cpu())
        self.ml_det_conf.append(pred["conf"].cpu())
        self.ml_gt_matched.append(gt_matched)

        # Store original class IDs for group-decode confusion matrix
        gt_cls_for_det = torch.full((n_det,), -1, dtype=torch.long)
        for di, gi in gt_for_det.items():
            gt_cls_for_det[di] = gt_cls_old[gi]
        self.ml_gt_cls_matched.append(gt_cls_for_det)

        unmatched_gt_idx = [i for i in range(n_gt) if i not in matched_gt]
        if unmatched_gt_idx:
            self.ml_gt_unmatched.append(gt_multihot[unmatched_gt_idx].cpu())
            self.ml_gt_cls_unmatched.append(gt_cls_old[unmatched_gt_idx].cpu())

    def finalize_metrics(self) -> None:
        """Finalize metrics, adding multi-label per-label report when class_map is active."""
        super().finalize_metrics()
        if getattr(self, "class_map_t", None) is not None and not self.training and self.ml_pred_scores:
            self._print_multilabel_metrics()

    def _print_multilabel_metrics(self):
        """Compute integrated F1 via group-argmax decode + confidence threshold sweep.

        Production-equivalent metric: group-argmax assigns one of 17 classes to each
        detection, confidence = min(active label sigmoids), then sweep confidence threshold
        to find F1-optimal operating point. Class-aware matching (pred class must match GT).
        """
        from ultralytics.utils.multihot import decode_multihot_groups

        all_scores = torch.cat(self.ml_pred_scores, dim=0)  # (N, n_labels)
        all_det_conf = torch.cat(self.ml_det_conf, dim=0)  # (N,) detection confidence from head
        all_gt_cls = torch.cat(self.ml_gt_cls_matched, dim=0)  # (N,) -1 for unmatched preds
        all_gt_cls_unmatched = (
            torch.cat(self.ml_gt_cls_unmatched, dim=0) if self.ml_gt_cls_unmatched else torch.zeros(0, dtype=torch.long)
        )

        nc = self.class_map_t.shape[0]  # 17
        eps = 1e-9

        # --- Group-argmax decode ---
        pred_cls = decode_multihot_groups(
            all_scores, self.class_map_t.cpu(), self.decode_groups, self.decode_rules,
        )

        # --- Confidence: min sigmoid of active labels for predicted class ---
        cm_cpu = self.class_map_t.cpu()
        safe_cls = pred_cls.clamp(min=0)
        active_mask = cm_cpu[safe_cls]  # (N, n_labels)
        masked_scores = all_scores * active_mask + (1 - active_mask) * 2.0
        conf = masked_scores.min(dim=1).values  # (N,)
        conf[pred_cls < 0] = 0  # discarded predictions

        # --- Class-aware TP: pred class must match GT class ---
        is_tp = (pred_cls == all_gt_cls) & (pred_cls >= 0)  # (N,)

        # GT count per class
        gt_count = torch.zeros(nc, dtype=torch.long)
        for c in range(nc):
            gt_count[c] = (all_gt_cls == c).sum() + (all_gt_cls_unmatched == c).sum()
        total_gt = gt_count.sum().float()

        n_total_pred = all_scores.shape[0]
        n_discarded = int((pred_cls < 0).sum().item())
        data_names = getattr(self, "_orig_names", {i: f"class_{i}" for i in range(nc)})

        LOGGER.info("")
        LOGGER.info("=" * 80)
        LOGGER.info("Integrated F1 (group-argmax decode + confidence threshold)")
        LOGGER.info("=" * 80)
        LOGGER.info(f"  Total predictions: {n_total_pred}  |  Discarded by rules: {n_discarded}")
        LOGGER.info(f"  Total GT objects:  {int(total_gt.item())}")
        LOGGER.info(f"  Confidence = min(sigmoid[active labels]) for predicted class")

        # --- Per-class F1 with per-class optimal threshold (like standard YOLO) ---
        LOGGER.info("")
        LOGGER.info("Per-class F1 (per-class optimal threshold):")
        LOGGER.info(f"{'Class':<28s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'P':>8s} {'R':>8s} {'F1':>8s} {'Thresh':>8s} {'GT':>6s}")
        LOGGER.info("-" * 92)

        all_class_f1 = []
        all_class_thresh = []
        for c in range(nc):
            pred_c_mask = pred_cls == c  # all predictions decoded as class c
            gt_c_count = gt_count[c].item()

            if gt_c_count == 0 or pred_c_mask.sum() == 0:
                name = data_names.get(c, f"class_{c}")
                LOGGER.info(f"{name:<28s} {0:>6d} {0:>6d} {gt_c_count:>6d} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {gt_c_count:>6d}")
                all_class_f1.append(0)
                all_class_thresh.append(0)
                continue

            # Sort class c predictions by confidence descending
            c_indices = torch.where(pred_c_mask)[0]
            c_conf = conf[c_indices]
            c_is_tp = is_tp[c_indices]
            order_c = c_conf.argsort(descending=True)
            c_tp_sorted = c_is_tp[order_c].float()
            c_conf_sorted = c_conf[order_c]

            tp_cum = c_tp_sorted.cumsum(dim=0)
            total_cum = torch.arange(1, len(c_tp_sorted) + 1, dtype=torch.float)
            fp_cum = total_cum - tp_cum

            p_curve = tp_cum / (total_cum + eps)
            r_curve = tp_cum / (gt_c_count + eps)
            f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

            best_idx = f1_curve.argmax().item()
            best_f1_c = f1_curve[best_idx].item()
            best_p_c = p_curve[best_idx].item()
            best_r_c = r_curve[best_idx].item()
            best_thresh_c = c_conf_sorted[best_idx].item()
            best_tp_c = int(tp_cum[best_idx].item())
            best_fp_c = int(fp_cum[best_idx].item())
            best_fn_c = gt_c_count - best_tp_c

            all_class_f1.append(best_f1_c)
            all_class_thresh.append(best_thresh_c)

            name = data_names.get(c, f"class_{c}")
            LOGGER.info(
                f"{name:<28s} {best_tp_c:>6d} {best_fp_c:>6d} {best_fn_c:>6d} "
                f"{best_p_c:>8.3f} {best_r_c:>8.3f} {best_f1_c:>8.3f} {best_thresh_c:>8.4f} {gt_c_count:>6d}"
            )

        macro_f1 = sum(all_class_f1) / len(all_class_f1) if all_class_f1 else 0
        LOGGER.info("-" * 92)
        LOGGER.info(f"{'Macro F1':<28s} {'':>6s} {'':>6s} {'':>6s} {'':>8s} {'':>8s} {macro_f1:>8.3f}")

        # --- Global threshold: micro F1 ---
        LOGGER.info("")
        is_valid = pred_cls >= 0
        order = conf.argsort(descending=True)
        is_tp_sorted = is_tp[order].float()
        is_valid_sorted = is_valid[order].float()
        conf_sorted = conf[order]

        tp_cum = is_tp_sorted.cumsum(dim=0)
        valid_cum = is_valid_sorted.cumsum(dim=0)
        fp_cum = valid_cum - tp_cum
        precision = tp_cum / (valid_cum + eps)
        recall = tp_cum / (total_gt + eps)
        f1_global = 2 * precision * recall / (precision + recall + eps)

        best_idx = f1_global.argmax().item()
        LOGGER.info(
            f"Global optimal threshold: {conf_sorted[best_idx].item():.4f}  |  "
            f"Micro F1: {f1_global[best_idx].item():.4f}  "
            f"(P={precision[best_idx].item():.4f} R={recall[best_idx].item():.4f})"
        )

        # Strategy A: 17-class confusion matrix (raw group-argmax, thresholds in 17-class space)
        if self.decode_groups is not None:
            self._print_group_decode_confusion_matrix(
                all_scores, "A_raw", "Strategy A: group-argmax (raw sigmoids)",
            )

        # Strategy B: 14-label threshold sweep → threshold-aware group-argmax → 17-class eval
        self._run_strategy_b(all_scores, all_gt_cls, all_gt_cls_unmatched, gt_count)

    def _print_group_decode_confusion_matrix(self, all_scores, prefix="", title="",
                                              label_thresholds=None):
        """Compute and print a confusion matrix using group-argmax decode rules.

        Args:
            all_scores: (N, n_labels) sigmoid scores for all predictions
            prefix: filename prefix for saved plots
            title: section title for log output
            label_thresholds: optional (n_labels,) tensor of per-label thresholds for decode
        """
        from ultralytics.utils.multihot import decode_multihot_groups

        # Gather GT class IDs
        all_gt_cls = torch.cat(self.ml_gt_cls_matched, dim=0)  # (N_det,) -1 for unmatched preds
        all_gt_cls_unmatched = (
            torch.cat(self.ml_gt_cls_unmatched, dim=0) if self.ml_gt_cls_unmatched else torch.zeros(0, dtype=torch.long)
        )

        # Decode predictions via group-argmax
        pred_cls = decode_multihot_groups(
            all_scores,
            self.class_map_t.cpu(),
            self.decode_groups,
            self.decode_rules,
            label_thresholds=label_thresholds,
        )

        nc_orig = self.class_map_t.shape[0]  # 17
        # Build confusion matrix: (nc+1) x (nc+1) — last row/col for background/discarded
        # Rows = GT, Cols = Pred
        cm = torch.zeros(nc_orig + 1, nc_orig + 1, dtype=torch.long)

        for i in range(pred_cls.shape[0]):
            gt = all_gt_cls[i].item()
            pr = pred_cls[i].item()
            gt_idx = gt if gt >= 0 else nc_orig  # -1 (unmatched pred) → background row
            pr_idx = pr if pr >= 0 else nc_orig  # -1 (discarded) → background col
            cm[gt_idx, pr_idx] += 1

        # Unmatched GTs → missed (background prediction)
        for gt in all_gt_cls_unmatched:
            cm[gt.item(), nc_orig] += 1

        # Build name list for display
        data_names = getattr(self, "_orig_names", None)
        if data_names is None or len(data_names) != nc_orig:
            data_names = {i: f"class_{i}" for i in range(nc_orig)}
        name_list = [data_names.get(i, f"class_{i}") for i in range(nc_orig)] + ["background"]

        # Print
        LOGGER.info("")
        LOGGER.info(f"{title or 'Group-argmax decode'} confusion matrix (rows=GT, cols=Pred):")
        header = f"{'':>22s}" + "".join(f"{n[:8]:>9s}" for n in name_list)
        LOGGER.info(header)
        for ri in range(nc_orig + 1):
            row_name = name_list[ri]
            row_vals = "".join(f"{cm[ri, ci].item():>9d}" for ci in range(nc_orig + 1))
            LOGGER.info(f"{row_name:>22s}{row_vals}")

        # Per-class accuracy summary
        LOGGER.info("")
        LOGGER.info(f"{'Class':<22s} {'Correct':>8s} {'Total':>8s} {'Acc':>8s}")
        LOGGER.info("-" * 50)
        total_correct = 0
        total_gt = 0
        for ci in range(nc_orig):
            correct = cm[ci, ci].item()
            total = cm[ci, :].sum().item()
            acc = correct / total if total > 0 else 0
            total_correct += correct
            total_gt += total
            LOGGER.info(f"{name_list[ci]:<22s} {correct:>8d} {total:>8d} {acc:>8.3f}")
        overall_acc = total_correct / total_gt if total_gt > 0 else 0
        LOGGER.info("-" * 50)
        LOGGER.info(f"{'OVERALL':<22s} {total_correct:>8d} {total_gt:>8d} {overall_acc:>8.3f}")

        # Save confusion matrix plots if plots enabled
        if self.args.plots and self.save_dir:
            import matplotlib.pyplot as plt

            cm_np = cm[:nc_orig, :nc_orig].numpy().astype(float)
            short_names = [name_list[i][:12] for i in range(nc_orig)]

            def _plot_cm(cm_norm, title, filename):
                fig, ax = plt.subplots(1, 1, figsize=(14, 12))
                im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks(range(nc_orig))
                ax.set_yticks(range(nc_orig))
                ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(short_names, fontsize=8)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Ground Truth")
                ax.set_title(title)
                for ri in range(nc_orig):
                    for ci in range(nc_orig):
                        val = cm_norm[ri, ci]
                        count = int(cm_np[ri, ci])
                        if count > 0:
                            color = "white" if val > 0.5 else "black"
                            ax.text(ci, ri, f"{val:.2f}\n({count})", ha="center", va="center", fontsize=6, color=color)
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(self.save_dir / filename, dpi=150)
                plt.close(fig)

            pfx = f"{prefix}_" if prefix else ""
            # Row-normalized (recall: of all GT=X, fraction predicted as each class)
            row_sums = cm_np.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            _plot_cm(cm_np / row_sums, f"{title} (recall, row-norm)", f"{pfx}cm_recall.png")

            # Column-normalized (precision: of all pred=X, fraction that are each GT class)
            col_sums = cm_np.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            _plot_cm(cm_np / col_sums, f"{title} (precision, col-norm)", f"{pfx}cm_precision.png")

            LOGGER.info(f"Confusion matrices ({prefix}) saved to {self.save_dir}")

    def _run_strategy_b(self, all_scores, all_gt_cls, all_gt_cls_unmatched, gt_count):
        """Strategy B: find optimal per-label thresholds on 14D, then use them in group-argmax.

        1. Per-label F1 threshold sweep (matched detections only)
        2. Feed those thresholds into threshold-aware group-argmax decode
        3. Evaluate resulting 17-class assignments
        """
        from ultralytics.utils.multihot import decode_multihot_groups

        n_labels = self.class_map_t.shape[1]  # 14
        nc = self.class_map_t.shape[0]  # 17
        cm_cpu = self.class_map_t.cpu()
        label_names = self.ml_label_names
        eps = 1e-9

        # --- Step 1: Per-label threshold sweep on matched detections ---
        matched = all_gt_cls >= 0
        if matched.sum() == 0:
            LOGGER.info("No matched detections for Strategy B.")
            return

        scores_matched = all_scores[matched]
        gt_multihot = cm_cpu[all_gt_cls[matched].long()]

        LOGGER.info("")
        LOGGER.info("=" * 80)
        LOGGER.info("Strategy B: per-label threshold sweep (14 labels)")
        LOGGER.info("=" * 80)
        LOGGER.info(f"  Matched detections: {int(matched.sum().item())}")
        LOGGER.info(f"{'Label':<22s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'P':>8s} {'R':>8s} {'F1':>8s} {'Thresh':>8s} {'GT+':>6s}")
        LOGGER.info("-" * 86)

        label_thresh = torch.full((n_labels,), 0.5)  # fallback
        all_label_f1 = []

        for li in range(n_labels):
            gt_pos = gt_multihot[:, li]  # (M,) binary
            n_pos = int(gt_pos.sum().item())
            n_neg = int((gt_pos == 0).sum().item())

            if n_pos == 0:
                name = label_names.get(li, f"label_{li}")
                LOGGER.info(f"{name:<22s} {0:>6d} {0:>6d} {0:>6d} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'0.500':>8s} {0:>6d}")
                all_label_f1.append(0)
                continue

            # Sort by score descending, sweep threshold
            s = scores_matched[:, li]
            order = s.argsort(descending=True)
            gt_sorted = gt_pos[order].float()
            s_sorted = s[order]

            tp_cum = gt_sorted.cumsum(dim=0)
            total_cum = torch.arange(1, len(gt_sorted) + 1, dtype=torch.float)
            fp_cum = total_cum - tp_cum

            p_curve = tp_cum / (total_cum + eps)
            r_curve = tp_cum / (n_pos + eps)
            f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

            best_idx = f1_curve.argmax().item()
            best_f1 = f1_curve[best_idx].item()
            best_p = p_curve[best_idx].item()
            best_r = r_curve[best_idx].item()
            best_thresh = s_sorted[best_idx].item()
            best_tp = int(tp_cum[best_idx].item())
            best_fp = int(fp_cum[best_idx].item())
            best_fn = n_pos - best_tp

            label_thresh[li] = best_thresh
            all_label_f1.append(best_f1)

            name = label_names.get(li, f"label_{li}")
            LOGGER.info(
                f"{name:<22s} {best_tp:>6d} {best_fp:>6d} {best_fn:>6d} "
                f"{best_p:>8.3f} {best_r:>8.3f} {best_f1:>8.3f} {best_thresh:>8.4f} {n_pos:>6d}"
            )

        macro_label_f1 = sum(all_label_f1) / len(all_label_f1) if all_label_f1 else 0
        LOGGER.info("-" * 86)
        LOGGER.info(f"{'Macro label F1':<22s} {'':>6s} {'':>6s} {'':>6s} {'':>8s} {'':>8s} {macro_label_f1:>8.3f}")
        LOGGER.info(f"\nOptimal label thresholds: {[f'{label_names.get(i, i)}={label_thresh[i]:.3f}' for i in range(n_labels)]}")

        # --- Step 2: Threshold-aware group-argmax decode → 17 classes ---
        pred_cls_b = decode_multihot_groups(
            all_scores, cm_cpu, self.decode_groups, self.decode_rules,
            label_thresholds=label_thresh,
        )

        # Confidence: min(sigmoid / threshold) for active labels of predicted class
        safe_cls = pred_cls_b.clamp(min=0)
        active_mask = cm_cpu[safe_cls]
        # Use sigmoid/threshold ratio as confidence — > 1 means above threshold
        label_thresh_safe = label_thresh.clamp(min=eps)
        ratio = all_scores / label_thresh_safe  # (N, n_labels)
        masked_ratio = ratio * active_mask + (1 - active_mask) * 999.0
        conf_b = masked_ratio.min(dim=1).values
        conf_b[pred_cls_b < 0] = 0

        # --- Step 3: Evaluate in 17-class space ---
        is_tp_b = (pred_cls_b == all_gt_cls) & (pred_cls_b >= 0)
        total_gt = gt_count.sum().float()
        data_names = getattr(self, "_orig_names", {i: f"class_{i}" for i in range(nc)})
        n_discarded = int((pred_cls_b < 0).sum().item())

        LOGGER.info("")
        LOGGER.info(f"Strategy B: 17-class F1 (using per-label thresholds in decode)")
        LOGGER.info(f"  Discarded by rules: {n_discarded}")
        LOGGER.info(f"{'Class':<28s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'P':>8s} {'R':>8s} {'F1':>8s} {'GT':>6s}")
        LOGGER.info("-" * 82)

        all_class_f1_b = []
        for c in range(nc):
            pred_c = pred_cls_b == c
            gt_c = gt_count[c].item()
            tp_c = int((pred_c & is_tp_b).sum().item())
            fp_c = int(pred_c.sum().item()) - tp_c
            fn_c = gt_c - tp_c
            p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
            r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0
            all_class_f1_b.append(f1_c)
            name = data_names.get(c, f"class_{c}")
            LOGGER.info(f"{name:<28s} {tp_c:>6d} {fp_c:>6d} {fn_c:>6d} {p_c:>8.3f} {r_c:>8.3f} {f1_c:>8.3f} {gt_c:>6d}")

        macro_f1_b = sum(all_class_f1_b) / len(all_class_f1_b) if all_class_f1_b else 0
        LOGGER.info("-" * 82)
        LOGGER.info(f"{'Macro F1 (B)':<28s} {'':>6s} {'':>6s} {'':>6s} {'':>8s} {'':>8s} {macro_f1_b:>8.3f}")

        # Micro F1 with threshold on confidence ratio
        is_valid = pred_cls_b >= 0
        order = conf_b.argsort(descending=True)
        tp_sorted = is_tp_b[order].float()
        valid_sorted = is_valid[order].float()
        conf_sorted = conf_b[order]

        tp_cum = tp_sorted.cumsum(dim=0)
        valid_cum = valid_sorted.cumsum(dim=0)
        precision = tp_cum / (valid_cum + eps)
        recall = tp_cum / (total_gt + eps)
        f1_g = 2 * precision * recall / (precision + recall + eps)
        best_g = f1_g.argmax().item()
        LOGGER.info(
            f"\nGlobal confidence ratio threshold: {conf_sorted[best_g].item():.4f}  |  "
            f"Micro F1: {f1_g[best_g].item():.4f}  "
            f"(P={precision[best_g].item():.4f} R={recall[best_g].item():.4f})"
        )

        # Strategy B confusion matrix
        self._print_group_decode_confusion_matrix(
            all_scores, "B_thresh", "Strategy B: threshold-aware group-argmax",
            label_thresholds=label_thresh,
        )

        # 14-label co-occurrence plot using optimal thresholds
        if self.args.plots and self.save_dir:
            self._plot_14label_cooccurrence(scores_matched, gt_multihot, label_thresh)

    def _plot_14label_cooccurrence(self, scores_matched, gt_multihot, label_thresh):
        """Plot 14x14 co-occurrence matrix using per-label optimal thresholds."""
        import matplotlib.pyplot as plt

        n_labels = scores_matched.shape[1]
        label_names = self.ml_label_names
        pred_binary = (scores_matched >= label_thresh).float()

        cooccur = torch.zeros(n_labels, n_labels)
        for li in range(n_labels):
            mask_i = gt_multihot[:, li] == 1
            if mask_i.sum() == 0:
                continue
            for lj in range(n_labels):
                cooccur[li, lj] = pred_binary[mask_i, lj].mean()

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        cm_np = cooccur.numpy()
        im = ax.imshow(cm_np, cmap="Blues", vmin=0, vmax=1)
        short_names = [label_names.get(i, f"l{i}")[:10] for i in range(n_labels)]
        ax.set_xticks(range(n_labels))
        ax.set_yticks(range(n_labels))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("Predicted label (above optimal threshold)")
        ax.set_ylabel("GT label active")
        ax.set_title("14-label co-occurrence (per-label optimal thresholds)")
        for ri in range(n_labels):
            for ci in range(n_labels):
                val = cm_np[ri, ci]
                if val > 0.01:
                    color = "white" if val > 0.5 else "black"
                    ax.text(ci, ri, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(self.save_dir / "B_label14_cooccurrence.png", dpi=150)
        plt.close(fig)
        LOGGER.info(f"14-label co-occurrence matrix saved to {self.save_dir / 'B_label14_cooccurrence.png'}")

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """Plot batch predictions with masks and bounding boxes.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
        """
        for p in preds:
            masks = p["masks"]
            if masks.shape[0] > self.args.max_det:
                LOGGER.warning(f"Limiting validation plots to 'max_det={self.args.max_det}' items.")
            p["masks"] = torch.as_tensor(masks[: self.args.max_det], dtype=torch.uint8).cpu()
        super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)  # plot bboxes

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', 'cls', and 'masks' keys.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image.
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Save one JSON result for COCO evaluation.

        Args:
            predn (dict[str, torch.Tensor]): Predictions containing bboxes, masks, confidence scores, and classes.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
        """

        def to_string(counts: list[int]) -> str:
            """Converts the RLE object into a compact string representation. Each count is delta-encoded and
            variable-length encoded as a string.

            Args:
                counts (list[int]): List of RLE counts.
            """
            result = []

            for i in range(len(counts)):
                x = int(counts[i])

                # Apply delta encoding for all counts after the second entry
                if i > 2:
                    x -= int(counts[i - 2])

                # Variable-length encode the value
                while True:
                    c = x & 0x1F  # Take 5 bits
                    x >>= 5

                    # If the sign bit (0x10) is set, continue if x != -1;
                    # otherwise, continue if x != 0
                    more = (x != -1) if (c & 0x10) else (x != 0)
                    if more:
                        c |= 0x20  # Set continuation bit
                    c += 48  # Shift to ASCII
                    result.append(chr(c))
                    if not more:
                        break

            return "".join(result)

        def multi_encode(pixels: torch.Tensor) -> list[int]:
            """Convert multiple binary masks using Run-Length Encoding (RLE).

            Args:
                pixels (torch.Tensor): A 2D tensor where each row represents a flattened binary mask with shape [N,
                    H*W].

            Returns:
                (list[list[int]]): A list of RLE counts for each mask.
            """
            transitions = pixels[:, 1:] != pixels[:, :-1]
            row_idx, col_idx = torch.where(transitions)
            col_idx = col_idx + 1

            # Compute run lengths
            counts = []
            for i in range(pixels.shape[0]):
                positions = col_idx[row_idx == i]
                if len(positions):
                    count = torch.diff(positions).tolist()
                    count.insert(0, positions[0].item())
                    count.append(len(pixels[i]) - positions[-1].item())
                else:
                    count = [len(pixels[i])]

                # Ensure starting with background (0) count
                if pixels[i][0].item() == 1:
                    count = [0, *count]
                counts.append(count)

            return counts

        pred_masks = predn["masks"].transpose(2, 1).contiguous().view(len(predn["masks"]), -1)  # N, H*W
        h, w = predn["masks"].shape[1:3]
        counts = multi_encode(pred_masks)
        rles = []
        for c in counts:
            rles.append({"size": [h, w], "counts": to_string(c)})
        super().pred_to_json(predn, pbatch)
        for i, r in enumerate(rles):
            self.jdict[-len(rles) + i]["segmentation"] = r  # segmentation

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **super().scale_preds(predn, pbatch),
            "masks": ops.scale_masks(predn["masks"][None], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])[
                0
            ].byte(),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Return COCO-style instance segmentation evaluation metrics."""
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm"], suffix=["Box", "Mask"])
