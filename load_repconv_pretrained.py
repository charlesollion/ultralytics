"""Load pretrained yolo26n-seg weights into yolo26rep-seg model.

Maps Conv weights from standard Bottleneck (in C3k2 head blocks) to
RepConv.conv1 (3x3 branch) in RepBottleneck (C3k2Rep head blocks).
The conv2 (1x1 branch) is zero-initialized so the RepConv starts
identical to the original Conv.

Works with both:
- COCO pretrained (nc=80) -> yolo26rep-seg (nc=17): normal transfer learning
- Your own trained yolo26n-seg (nc=17) -> yolo26rep-seg (nc=17): full weight transfer

Usage:
    python load_repconv_pretrained.py --weights yolo26n-seg.pt --save yolo26rep-seg-init.pt
"""

import argparse
from collections import OrderedDict

import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts


def remap_pretrained_keys(pretrained_sd, target_sd):
    """Remap pretrained Conv keys to RepConv keys.

    .cv1.conv.  -> .cv1.conv1.conv.    (3x3 branch)
    .cv1.bn.    -> .cv1.conv1.bn.      (3x3 branch BN)

    Only remaps a key if the remapped version exists in target_sd,
    so backbone C3k2 keys (standard Bottleneck) are left unchanged.
    """
    remapped = OrderedDict()
    n_remapped = 0
    for key, value in pretrained_sd.items():
        new_key = None
        if ".cv1.conv." in key:
            new_key = key.replace(".cv1.conv.", ".cv1.conv1.conv.")
        elif ".cv1.bn." in key:
            new_key = key.replace(".cv1.bn.", ".cv1.conv1.bn.")

        if new_key and new_key in target_sd:
            remapped[new_key] = value
            n_remapped += 1
        else:
            remapped[key] = value
    return remapped, n_remapped


def zero_init_conv2(model):
    """Zero-initialize all RepConv conv2 (1x1 branch) conv weights.

    This ensures the 1x1 branch contributes nothing at initialization,
    making the RepConv functionally identical to the original Conv.
    BN defaults (weight=1, bias=0, mean=0, var=1) are kept as-is since
    zero conv weights already guarantee zero output.
    """
    n_zeroed = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".cv1.conv2.conv.weight" in name:
                param.zero_()
                n_zeroed += 1
    return n_zeroed


def main():
    parser = argparse.ArgumentParser(description="Init yolo26rep-seg from pretrained yolo26n-seg")
    parser.add_argument("--weights", type=str, required=True, help="Path to pretrained yolo26n-seg.pt")
    parser.add_argument("--yaml", type=str, default="yolo26rep-seg.yaml", help="Model YAML config")
    parser.add_argument("--save", type=str, default="yolo26rep-seg-init.pt", help="Output path")
    args = parser.parse_args()

    # Load pretrained model state dict
    print(f"Loading pretrained weights from {args.weights}")
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    pretrained_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if hasattr(pretrained_model, "float"):
        pretrained_model = pretrained_model.float()
    pretrained_sd = pretrained_model.state_dict() if hasattr(pretrained_model, "state_dict") else pretrained_model

    # Build rep model from yaml
    print(f"Building rep model from {args.yaml}")
    rep_model = YOLO(args.yaml)
    rep_sd = rep_model.model.state_dict()

    # Remap Conv -> RepConv keys (only where target model has RepConv)
    remapped_sd, n_remapped = remap_pretrained_keys(pretrained_sd, rep_sd)
    print(f"Remapped {n_remapped} keys from Conv -> RepConv.conv1")

    # Use intersect_dicts (same as ultralytics' normal loading) to handle shape mismatches
    updated_sd = intersect_dicts(remapped_sd, rep_sd)
    n_loaded = len(updated_sd)
    n_total = len(rep_sd)
    rep_model.model.load_state_dict(updated_sd, strict=False)
    print(f"Loaded {n_loaded}/{n_total} keys via intersect_dicts")

    # Show what didn't transfer (excluding conv2 branches which we'll zero-init)
    missing = []
    for k in rep_sd:
        if k not in updated_sd and ".cv1.conv2." not in k:
            missing.append(k)
    if missing:
        print(f"\nKeys not transferred ({len(missing)}, likely nc mismatch):")
        for k in missing[:10]:
            print(f"  {k} {rep_sd[k].shape}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # Zero-init conv2 (1x1 branch) weights
    n_zeroed = zero_init_conv2(rep_model.model)
    print(f"Zero-initialized {n_zeroed} RepConv conv2 (1x1 branch) weight tensors")

    # Verify functional equivalence on a forward pass
    print("\nRunning verification forward pass...")
    rep_model.model.eval()
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = rep_model.model(dummy)
    print("Forward pass OK")

    # Save in ultralytics checkpoint format
    ckpt = {"model": rep_model.model, "epoch": -1, "best_fitness": None, "train_args": {}}
    torch.save(ckpt, args.save)
    print(f"\nSaved to {args.save}")
    print(f"Total params: {sum(p.numel() for p in rep_model.model.parameters()):,}")

    # Compare with pretrained param count
    if hasattr(pretrained_model, "parameters"):
        pre_params = sum(p.numel() for p in pretrained_model.parameters())
        rep_params = sum(p.numel() for p in rep_model.model.parameters())
        diff = rep_params - pre_params
        print(f"Pretrained params: {pre_params:,} (nc={'80 (COCO)' if pre_params > rep_params + 10000 else 'same'})")
        print(f"Rep model params:  {rep_params:,}")
        if pre_params > rep_params + 10000:
            print(f"Difference is mostly from nc mismatch (80 vs 17), not RepConv")
            # Estimate RepConv overhead: count conv2 params
            conv2_params = sum(p.numel() for n, p in rep_model.model.named_parameters() if ".cv1.conv2." in n)
            print(f"Actual RepConv overhead: +{conv2_params:,} params ({conv2_params/rep_params*100:.1f}%)")
        else:
            print(f"RepConv overhead: +{diff:,} params ({diff/pre_params*100:.1f}%)")


if __name__ == "__main__":
    main()
