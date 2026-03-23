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
        yaml_cfg = getattr(model, "yaml", None) or {}
        class_primary = yaml_cfg.get("class_primary")
        class_map = yaml_cfg.get("class_map")
        if class_primary is not None and class_map is not None:
            self.class_primary = torch.tensor(class_primary, dtype=torch.long, device=self.device)
            self.class_map_t = torch.tensor(class_map, dtype=torch.float, device=self.device)  # (17, 14)
            self.model = model  # keep ref for _get_decode_boxes in postprocess
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
        """Decode 14D multi-hot predictions to 17-class space using class_map.

        For each of the 17 original classes, the score is the minimum sigmoid score
        among its active labels. This is the most conservative aggregation — a class
        scores high only when ALL its constituent attributes are detected.
        """
        y_postprocessed = preds[0][0]  # (bs, 300, 6+nm) — head's 14-label postprocess
        raw_one2one = preds[1]["one2one"]

        # Get decoded boxes from the head's postprocessed output (already xyxy)
        # and the 14D sigmoid scores from raw one2one
        scores_14 = raw_one2one["scores"].sigmoid()  # (bs, 14, anchors)
        scores_14 = scores_14.permute(0, 2, 1)  # (bs, anchors, 14)

        # Decode 14D -> 17D: score_c = min(active label scores) for each old class
        cm = self.class_map_t  # (17, 14)
        # Where label is active use score, where inactive use large value (won't affect min)
        masked = scores_14.unsqueeze(-2) * cm + (1 - cm) * 2.0  # (bs, anchors, 17, 14)
        scores_17 = masked.min(dim=-1).values  # (bs, anchors, 17)

        # Use the boxes from the already-postprocessed y (head already decoded DFL + anchors)
        boxes = y_postprocessed[:, :, :4]  # (bs, 300, 4)
        mask_coeffs = y_postprocessed[:, :, 6:]  # (bs, 300, nm)

        # Gather 17D scores for the 300 anchors the head selected
        # The head selected top-300 by 14D max — we reuse those same anchors
        # but reclassify them in 17D space
        # We need anchor indices, but they're not in y. Instead, match via boxes.
        # Simpler: just re-run topk on 17D scores ourselves.
        bs = scores_17.shape[0]
        nc_orig = scores_17.shape[-1]  # 17
        max_det = y_postprocessed.shape[1]  # 300

        # Top-k selection in 17D space (same logic as Detect.get_topk_index)
        k = min(max_det, scores_17.shape[1])
        ori_index = scores_17.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)  # (bs, k, 1)
        sel_scores = scores_17.gather(dim=1, index=ori_index.expand(-1, -1, nc_orig))  # (bs, k, 17)
        flat_scores, flat_idx = sel_scores.flatten(1).topk(k)
        anchor_idx = ori_index[torch.arange(bs)[..., None], flat_idx // nc_orig]  # (bs, k, 1)
        class_idx = (flat_idx % nc_orig).unsqueeze(-1).float()  # (bs, k, 1)
        conf = flat_scores.unsqueeze(-1)  # (bs, k, 1)

        # Gather boxes from raw decoded boxes (reuse head's _inference output)
        # The raw inference output is (bs, 4+14, anchors) but boxes are at [:4].
        # We can get decoded boxes from preds[0][0] only for the top-300.
        # Instead, re-decode from raw one2one using the head module.
        head = None
        if hasattr(self, 'model') and self.model is not None:
            head = self.model.model[-1] if hasattr(self.model, 'model') else None

        if head is not None and hasattr(head, '_get_decode_boxes'):
            dbox = head._get_decode_boxes(raw_one2one)  # (bs, 4, anchors)
            dbox = dbox.permute(0, 2, 1)  # (bs, anchors, 4)
            sel_boxes = dbox.gather(dim=1, index=anchor_idx.expand(-1, -1, 4))
        else:
            # Fallback: reuse head's top-300 boxes (approximate — anchor sets may differ)
            sel_boxes = boxes

        # Gather mask coefficients and 14D scores for selected anchors
        if "mask_coefficient" in raw_one2one:
            mc = raw_one2one["mask_coefficient"].permute(0, 2, 1)  # (bs, anchors, nm)
            nm = mc.shape[-1]
            sel_mc = mc.gather(dim=1, index=anchor_idx.expand(-1, -1, nm))
        else:
            sel_mc = mask_coeffs  # fallback

        nc_multi = scores_14.shape[-1]  # 14
        sel_scores_14 = scores_14.gather(dim=1, index=anchor_idx.expand(-1, -1, nc_multi))  # (bs, k, 14)

        # NMS-style output: filter by confidence
        results = []
        for xi in range(bs):
            filt = conf[xi, :, 0] > self.args.conf
            det = torch.cat([
                sel_boxes[xi][filt],
                conf[xi][filt],
                class_idx[xi][filt],
                sel_mc[xi][filt],
            ], dim=-1)
            results.append({
                "bboxes": det[:, :4],
                "conf": det[:, 4],
                "cls": det[:, 5],
                "extra": det[:, 6:],
                "scores_multilabel": sel_scores_14[xi][filt],  # (n_det, 14) sigmoid scores
            })
        return results

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
            return

        if n_gt == 0:
            self.ml_pred_scores.append(scores_ml.cpu())
            self.ml_gt_matched.append(torch.zeros(n_det, n_labels))
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
        self.ml_gt_matched.append(gt_matched)

        unmatched_gt_idx = [i for i in range(n_gt) if i not in matched_gt]
        if unmatched_gt_idx:
            self.ml_gt_unmatched.append(gt_multihot[unmatched_gt_idx].cpu())

    def finalize_metrics(self) -> None:
        """Finalize metrics, adding multi-label per-label report when class_map is active."""
        super().finalize_metrics()
        if getattr(self, "class_map_t", None) is not None and not self.training and self.ml_pred_scores:
            self._print_multilabel_metrics()

    def _print_multilabel_metrics(self):
        """Compute and print per-label precision/recall/F1 with optimal thresholds."""
        all_scores = torch.cat(self.ml_pred_scores, dim=0)  # (N_total_det, n_labels)
        all_gt_matched = torch.cat(self.ml_gt_matched, dim=0)  # (N_total_det, n_labels)
        all_gt_unmatched = torch.cat(self.ml_gt_unmatched, dim=0) if self.ml_gt_unmatched else torch.zeros(0, all_scores.shape[1])

        n_labels = all_scores.shape[1]
        n_thresh = 1000
        thresholds = torch.linspace(0, 1, n_thresh)  # (T,)
        eps = 1e-9

        LOGGER.info("")
        LOGGER.info("Multi-label per-label metrics (IoU>=0.5, class-agnostic matching):")
        LOGGER.info(f"{'Label':<20s} {'P':>8s} {'R':>8s} {'F1':>8s} {'Thresh':>8s} {'Support':>8s}")
        LOGGER.info("-" * 72)

        all_f1 = []
        for li in range(n_labels):
            scores_l = all_scores[:, li]  # (N,)
            gt_l = all_gt_matched[:, li]  # (N,)
            n_fn_unmatched = all_gt_unmatched[:, li].sum().item() if all_gt_unmatched.shape[0] > 0 else 0
            support = int(gt_l.sum().item() + n_fn_unmatched)

            # Vectorized: (N,) >= (T, 1) -> (T, N) boolean
            pred_pos = scores_l.unsqueeze(0) >= thresholds.unsqueeze(1)  # (T, N)
            gt_pos = gt_l.bool().unsqueeze(0)  # (1, N)

            tp = (pred_pos & gt_pos).sum(dim=1).float()  # (T,)
            fp = (pred_pos & ~gt_pos).sum(dim=1).float()  # (T,)
            fn = (~pred_pos & gt_pos).sum(dim=1).float() + n_fn_unmatched  # (T,)

            p = tp / (tp + fp + eps)
            r = tp / (tp + fn + eps)
            f1 = 2 * p * r / (p + r + eps)

            best_idx = f1.argmax().item()
            best_f1 = f1[best_idx].item()
            best_p = p[best_idx].item()
            best_r = r[best_idx].item()
            best_t = thresholds[best_idx].item()
            all_f1.append(best_f1)

            name = self.ml_label_names.get(li, f"label_{li}")
            LOGGER.info(f"{name:<20s} {best_p:>8.3f} {best_r:>8.3f} {best_f1:>8.3f} {best_t:>8.3f} {support:>8d}")

        mean_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
        LOGGER.info("-" * 72)
        LOGGER.info(f"{'MEAN':<20s} {'':>8s} {'':>8s} {mean_f1:>8.3f}")

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
