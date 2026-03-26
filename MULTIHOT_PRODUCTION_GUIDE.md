# Multi-Hot Model: Raw Outputs to Production Predictions

## Overview

The model outputs **14 sigmoid scores** per detection, representing attribute labels rather than flat classes. These 14 labels decompose the original 17 waste classes into two exclusive groups:

| Index | Label             | Group    |
|-------|-------------------|----------|
| 0     | alimentaire       | object   |
| 1     | plastique         | material |
| 2     | papier            | material |
| 3     | carton            | material |
| 4     | bois              | material |
| 5     | metal             | material |
| 6     | verre             | material |
| 7     | bouteille         | object   |
| 8     | canette           | object   |
| 9     | encombrant        | object   |
| 10    | megot             | object   |
| 11    | paquet_cigarette  | object   |
| 12    | sac               | object   |
| 13    | textile           | material |

The 17 original classes are combinations of these labels (e.g. class "bouteille-en-plastique" = bouteille + plastique).

---

## Production Pipeline: Raw Outputs → Final Predictions

### Step 1: Score Merging (NMS-merge)

The model head outputs ~8400 anchor proposals, each with a 4D box and 14D raw score vector.
Nearby anchors often capture complementary information (one sees the material, another sees the object).
We merge overlapping anchors before classification.

```
Input:  boxes (N, 4),  raw_scores (N, 14)
```

1. **Sigmoid** the raw scores → 14D sigmoid per anchor
2. **Pre-filter** top 1000 anchors by max sigmoid score
3. **NMS-merge** at IoU > 0.6: use standard NMS to find cluster leaders, then for each leader, take the **element-wise max** of 14D sigmoid scores across all overlapping anchors in its cluster. This captures the best signal for each label from nearby proposals.
4. Leader's box geometry is kept; mask coefficients come from the leader anchor.

**Result:** ~200-500 merged proposals, each with a high-quality 14D score vector.

### Step 2: Group-Argmax Decode (14D → 17 classes)

For each merged proposal:

1. **Material selection:** `argmax(sigmoid[material_group])` → best material (always picks one)
2. **Object selection:** `argmax(sigmoid[object_group])` → best object candidate
3. **Object presence check:** Is `sigmoid[best_object] > obj_threshold`?
   - **No object** → material-only class (e.g. "autre-plastique")
   - **Valid combo** (material, object) exists in class_map → use it (e.g. "bouteille-en-plastique")
   - **Standalone object** (encombrant, megot — no valid materials) → object-only class
   - **Priority object** (canette, paquet_cigarette) → always maps to its default class regardless of material
   - **Material priority** (metal allows only canette/encombrant as objects) → material-only if object not allowed
   - **Discard on invalid** (sac without plastique) → discard detection
   - **Fallback:** pick best valid material for this object from sigmoid scores

### Step 3: Confidence Score

After decode, confidence = `min(sigmoid[active_labels])` for the predicted class.

Example: prediction is "bouteille-en-plastique" (active labels: bouteille, plastique).
If sigmoid[bouteille]=0.7, sigmoid[plastique]=0.4, then confidence = 0.4 (the bottleneck label).

### Step 4: Class-Aware NMS

Standard class-aware NMS at IoU > 0.5 on the 17-class predictions, using the confidence from Step 3. Keep top 300 detections.

### Step 5: Confidence Threshold

Filter detections below a confidence threshold.

---

## Recommended Thresholds (Strategy A)

Strategy A uses raw sigmoid argmax + per-class confidence threshold sweep, and is the recommended approach.

**Global threshold** (single value, simpler): **0.214** → Micro F1 = 0.500 (P=0.571, R=0.445)

**Per-class optimal thresholds** (better F1, more complex):

| Class                       | Threshold | F1    | Precision | Recall | GT count |
|-----------------------------|-----------|-------|-----------|--------|----------|
| alimentaire-papier          | 0.307     | 0.338 | 0.417     | 0.285  | 158      |
| alimentaire-plastique       | 0.404     | 0.405 | 0.538     | 0.325  | 237      |
| autre-bois                  | 0.174     | 0.394 | 0.500     | 0.325  | 40       |
| autre-carton                | 0.231     | 0.488 | 0.609     | 0.406  | 96       |
| autre-metal                 | 0.333     | 0.305 | 0.370     | 0.259  | 116      |
| autre-papier-carton         | 0.194     | 0.523 | 0.541     | 0.505  | 1307     |
| autre-plastique-fragments   | 0.217     | 0.307 | 0.365     | 0.266  | 813      |
| autre-polystyrene           | —         | 0.000 | —         | —      | 44       |
| bouteille-en-plastique      | 0.466     | 0.642 | 0.729     | 0.573  | 89       |
| bouteille-en-verre          | 0.306     | 0.646 | 0.913     | 0.500  | 42       |
| canette                     | 0.762     | 0.699 | 0.891     | 0.576  | 99       |
| encombrant                  | 0.856     | 0.458 | 0.917     | 0.306  | 36       |
| megot                       | 0.300     | 0.596 | 0.707     | 0.515  | 2246     |
| paquet_cigarette            | 0.582     | 0.441 | 0.625     | 0.341  | 44       |
| sac-ordures-menageres       | 0.224     | 0.654 | 0.737     | 0.587  | 143      |
| textile                     | 0.960     | 0.154 | 1.000     | 0.083  | 24       |
| verre                       | 0.348     | 0.077 | 0.094     | 0.065  | 46       |

**Macro F1 = 0.419** (average of per-class F1 at per-class optimal thresholds)

### Strategy A summary
- obj_threshold for presence check: **0.3** (flat across all object labels)
- Confidence = `min(sigmoid[active labels])` for the predicted class
- Threshold sweep in 17-class space on this confidence

---

## Alternative: Strategy B (Per-Label Thresholds)

Strategy B finds optimal per-label thresholds on the 14 sigmoid outputs, then uses them in the presence check. Macro F1 = 0.413, very close to A.

**Optimal 14-label thresholds** (useful diagnostics even if not used in decode):

| Label             | Threshold | Label F1 |
|-------------------|-----------|----------|
| alimentaire       | 0.135     | 0.420    |
| plastique         | 0.147     | 0.451    |
| papier            | 0.079     | 0.509    |
| carton            | 0.067     | 0.487    |
| bois              | 0.174     | 0.270    |
| metal             | 0.383     | 0.535    |
| verre             | 0.186     | 0.413    |
| bouteille         | 0.309     | 0.657    |
| canette           | 0.762     | 0.697    |
| encombrant        | 0.472     | 0.650    |
| megot             | 0.054     | 0.657    |
| paquet_cigarette  | 0.362     | 0.438    |
| sac               | 0.020     | 0.694    |
| textile           | 0.135     | 0.160    |

**14-label Macro F1 = 0.623**

### Strategy B summary
- Raw argmax for within-group ranking (same as A)
- Object presence: winning object's score must exceed **its own per-label threshold** (not a flat value)
- Confidence = `min(sigmoid / label_threshold)` for active labels (ratio > 1 means all above threshold)
- Per-class threshold sweep on this ratio in 17-class space

### Key insight: why B doesn't beat A
Per-label thresholds are calibrated on matched detections ("given this is a real object, is label X correct?"). The presence check asks a different question ("is any object present at all?"). The flat 0.3 threshold in Strategy A happens to be a reasonable middle ground. The per-label thresholds are individually better calibrated but don't compose into a significantly better presence decision.


## Strategy C (to implement and test)

### Decode strategy
We have 14-label threshold, see below how to compute them.

1. Output shape is transmuted: [1, 8400, 18] (18 = 14 labels + 4 coordinates) already sigmoid
2. **Pre-filter** loop over 8400 and take only two max per group, keep if at least one is above its class threshold
3. **NMS-merge** at IoU > 0.6: use class agnostic NMS to find cluster leaders, then for each leader, take the **element-wise max** of 14D scores across all overlapping anchors in its cluster. This captures the best signal for each label from nearby proposals. Leader's box geometry is kept.
4. **Material selection:** `argmax([material_group]) > class_threshold` → best material candidate
5. **Object selection:** `argmax([object_group]) > class_threshold` → best object candidate
6. **Litter specific rules:** Is `[best_object] > class_threshold`?
   - **No object** → material-only class (e.g. "autre-plastique") if above threshold
   - **Valid combo** (material, object) exists in class_map → use it (e.g. "bouteille-en-plastique")
   - **Standalone object** (encombrant, megot — no valid materials) → object-only class
   - **Priority object** (canette, paquet_cigarette) → always maps to its default class regardless of material
   - **Material priority** (metal allows only canette/encombrant as objects) → material-only if object not allowed
   - **Discard on invalid** (sac without plastique) → discard detection
   - **Fallback: No material, other object** → discard detection


### Threshold sweep
How to find the 14-label threshold, such that the F1 score is best in 17-label preds/gts.

---

## Decode Rules Reference

```yaml
decode_groups:
  material: [1, 2, 3, 4, 5, 6, 13]  # plastique, papier, carton, bois, metal, verre, textile
  object: [0, 7, 8, 9, 10, 11, 12]   # alimentaire, bouteille, canette, encombrant, megot, paquet_cigarette, sac

decode_rules:
  # Always map to their default class regardless of material
  object_priority: [8, 11]  # canette, paquet_cigarette
  # Discard when paired with invalid material
  discard_on_invalid: [12]  # sac (only valid with plastique)
  # Material overrides object when combo is invalid
  material_priority:
    5: [8, 9]  # metal: only allow canette and encombrant as objects
```

## Class Map (17 classes → 14 labels)

```
 0: alimentaire-papier        → alimentaire + papier
 1: alimentaire-plastique     → alimentaire + plastique
 2: autre-bois                → bois
 3: autre-carton              → carton
 4: autre-metal               → metal
 5: autre-papier-carton       → papier
 6: autre-plastique-fragments → plastique
 7: autre-polystyrene         → plastique  (same as 6, unresolvable)
 8: bouteille-en-plastique    → bouteille + plastique
 9: bouteille-en-verre        → bouteille + verre
10: canette                   → canette + metal
11: encombrant                → encombrant
12: megot                     → megot
13: paquet_cigarette          → paquet_cigarette + papier
14: sac-ordures-menageres     → sac + plastique
15: textile                   → textile
16: verre                     → verre
```

## Known Limitations

- **autre-polystyrene** (class 7) shares the exact same multi-hot encoding as autre-plastique-fragments (class 6) — both are just [plastique]. The model cannot distinguish them. F1 = 0.
- **textile** and **verre** (as standalone material) have very few training examples and poor F1 (0.154 and 0.077).
- **autre-metal** is often confused with other material-only classes (F1 = 0.305).
- The merge step adds ~25ms per image (acceptable for production, validated at 26ms total postprocess).

## Implementation Checklist

1. Get raw model outputs: boxes (N, 4) and scores (N, 14) pre-sigmoid
2. Apply sigmoid to scores
3. Run NMS-merge pipeline (top-1000 → cluster at IoU>0.6 → element-wise max of 14D)
4. Group-argmax decode with rules → 17-class assignment
5. Compute confidence = min(sigmoid[active labels])
6. Class-aware NMS at IoU > 0.5
7. Filter by confidence threshold (0.214 global or per-class)
8. Output: boxes, class IDs (0-16), confidence scores

Key files:
- `ultralytics/utils/multihot.py` — `decode_multihot_groups()` and `postprocess_group_nms()`
- `ultralytics/cfg/models/26/yolo28-seg.yaml` — class_map, decode_groups, decode_rules
- `ultralytics/models/yolo/segment/val.py` — validation integration with both strategies


## Next steps and observations

### Improve training

- During training, the TaskAligner uses "class_primary", which is arbitrary and might create problems (does it)?
- class_primary seems off isn't it? [0, 0, 4, 3, 5, 2, 1, 1, 7, 7, 8, 9, 10, 11, 12, 13, 6] 
- We use a special rule by multiplying material and object scores, which might be too detrimental (well it worked quite well at first glance though). Can we have something more principled and linked to our inference strategy?
- The selection of best model is made on validation, using an inference rule that is NOT the one we built for inference. This might lead to 1. misleading val stats and 2. wrong selection of model.
- We use class weights as well, maybe they could be improved.

### Validation
- polystyrene is not with the classes, we lose a bit of F1 score
- verify that everything is made box-wise, and not mask-wise in IoU computations.

### Visualisation
- make labels a bit larger in fp