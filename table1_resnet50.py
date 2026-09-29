"""
Replication of Table 1 from:
  "M4: A Unified XAI Benchmark for Faithfulness Evaluation of Feature
   Attribution Methods across Metrics, Modalities and Models"
  NeurIPS 2023 D&B Track.

Target: ResNet-50, ImageNet validation, 5 methods × 5 metrics.

Metrics implemented here:
  - MoRF  : Most Relevant First AUC (Eq. 1)
  - LeRF  : Least Relevant First AUC (Eq. 2)
  - ABPC  : MoRF - LeRF (Eq. 3)

Attribution methods:
  - constant  : constant-zero saliency (baseline)
  - random    : random N(0,1) saliency
  - gradcam   : Grad-CAM (layer4 of ResNet-50)
  - ig        : Integrated Gradients (captum)
  - sg        : SmoothGrad (captum)

Dataset: ImageNet validation images with labels.
Paper uses 5 images per class × 1000 classes = 5000 images.
Here we work with whatever images are available in --data_dir.

Usage:
    python table1_resnet50.py \\
        --data_dir /path/to/imagenet/val \\
        --label_file /path/to/ILSVRC2012_validation_ground_truth.txt \\
        --num_samples 500 \\
        --patch_size 16 \\
        --n_steps 50 \\
        --output_csv results_table1.csv
"""

import argparse
import csv
import os
import random
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from captum.attr import IntegratedGradients, NoiseTunnel

# ── Optional: pytorch-grad-cam ──────────────────────────────────────────────
try:
    from pytorch_grad_cam import GradCAM as PytorchGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    print("[warn] pytorch-grad-cam not installed; GradCAM will fall back to "
          "gradient × input. Install with: pip install grad-cam", flush=True)

# ── Preprocessing & model ────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_resnet50(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    return model


# ── Masking baseline ─────────────────────────────────────────────────────────

def get_mean_baseline(device):
    """
    The paper uses the channel-wise mean as the masking constant.
    mean pixel values (after normalisation) = (0-mean)/std = -mean/std.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)
    # normalized value that corresponds to the original mean pixel
    # i.e. (mean_pixel - mean) / std = 0 for each channel
    return torch.zeros(1, 3, 1, 1, device=device)


# ── Attribution methods ───────────────────────────────────────────────────────

def attr_constant(input_tensor, pred_class, model, device, **kw):
    """All-zero attribution map — constant baseline."""
    return torch.zeros(1, 224, 224, device=device)


def attr_random(input_tensor, pred_class, model, device, **kw):
    """Random N(0,1) attribution — random baseline."""
    return torch.randn(1, 224, 224, device=device)


def _aggregate_channels(attr):
    """Sum absolute values over colour channels → (1, H, W) cpu tensor."""
    return attr.abs().sum(dim=1)   # (1, H, W)


def attr_ig(input_tensor, pred_class, model, device, n_steps=50, **kw):
    """Integrated Gradients with zero baseline."""
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor)
    attrs = ig.attribute(
        input_tensor, baselines=baseline,
        target=pred_class, n_steps=n_steps,
        internal_batch_size=4,
    )
    return _aggregate_channels(attrs.cpu())


def attr_sg(input_tensor, pred_class, model, device, n_steps=50,
            n_samples=50, stdev_spread=0.15, **kw):
    """SmoothGrad (Noise Tunnel over Gradient × input)."""
    from captum.attr import Saliency
    sal = Saliency(model)
    nt  = NoiseTunnel(sal)
    # stdev = spread × (max - min) of input  ≈ paper default
    stdev = stdev_spread * (input_tensor.max() - input_tensor.min()).item()
    attrs = nt.attribute(
        input_tensor,
        n_samples=n_samples,
        stdevs=stdev,
        nt_type='smoothgrad_sq',
        target=pred_class,
    )
    return _aggregate_channels(attrs.cpu())


def attr_gradcam(input_tensor, pred_class, model, device, **kw):
    """
    Grad-CAM on ResNet-50 layer4.
    Uses pytorch-grad-cam if installed; otherwise falls back to
    gradient × input aggregated spatially.
    """
    if HAS_GRADCAM:
        target_layer = [model.layer4[-1]]
        cam = PytorchGradCAM(model=model, target_layers=target_layer)
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # shape (1, H, W); already [0,1]
        return torch.from_numpy(grayscale_cam)

    # Fallback: gradient magnitude
    inp = input_tensor.clone().requires_grad_(True)
    out = model(inp)
    score = out[0, pred_class]
    score.backward()
    grad = inp.grad.abs()
    return _aggregate_channels(grad.cpu())


ATTRIBUTORS = {
    "constant": attr_constant,
    "random":   attr_random,
    "ig":       attr_ig,
    "sg":       attr_sg,
    "gradcam":  attr_gradcam,
}


# ── Patch masking ─────────────────────────────────────────────────────────────

def build_patch_ranking(attr_map: torch.Tensor, patch_size: int):
    """
    attr_map : (1, 224, 224) float tensor — higher = more important.
    Returns list of (row, col) patch indices sorted descending by importance.
    """
    H = W = 224
    gh = H // patch_size
    gw = W // patch_size
    patches = []
    arr = attr_map.squeeze(0).cpu().numpy()   # (224, 224)
    for pr in range(gh):
        for pc in range(gw):
            r0, r1 = pr * patch_size, (pr + 1) * patch_size
            c0, c1 = pc * patch_size, (pc + 1) * patch_size
            val = arr[r0:r1, c0:c1].mean()
            patches.append((val, pr, pc))
    patches.sort(key=lambda x: x[0], reverse=True)  # descending
    return patches  # MoRF order


def apply_mask(input_tensor, patches_to_mask, patch_size, mask_value=0.0):
    """Return a copy of input_tensor with specified patches replaced."""
    out = input_tensor.clone()
    for _, pr, pc in patches_to_mask:
        r0, r1 = pr * patch_size, (pr + 1) * patch_size
        c0, c1 = pc * patch_size, (pc + 1) * patch_size
        out[0, :, r0:r1, c0:c1] = mask_value
    return out


# ── MoRF / LeRF curves ───────────────────────────────────────────────────────

PERCENTAGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

@torch.no_grad()
def evaluate_curve(
    model, input_tensor, pred_class, ranked_patches,
    patch_size, device, reverse=False
):
    """
    Compute perturbation curve.
    reverse=False → MoRF  (mask top-ranked first)
    reverse=True  → LeRF  (mask bottom-ranked first)
    Returns list of probabilities at each percentage step.
    """
    total = len(ranked_patches)
    # Original unmasked probability
    orig_logits = model(input_tensor)
    orig_prob = F.softmax(orig_logits, dim=1)[0, pred_class].item()

    curve = [orig_prob]
    order = list(reversed(ranked_patches)) if reverse else ranked_patches

    for p in PERCENTAGES[1:]:
        n = int(p * total)
        masked = apply_mask(input_tensor, order[:n], patch_size)
        masked = masked.to(device)
        logits = model(masked)
        prob = F.softmax(logits, dim=1)[0, pred_class].item()
        curve.append(prob)

    return curve


def auc_curve(curve):
    """Trapezoid AUC of the perturbation curve over [0, 1]."""
    return float(np.trapezoid(curve, PERCENTAGES))


# ── Main evaluation loop ──────────────────────────────────────────────────────

def load_images_with_labels(data_dir, label_file, num_samples, preprocess, device):
    """
    Loads images from data_dir.
    If label_file is provided, reads ground-truth labels (1-indexed).
    If label_file is None (test set), no ground-truth labels — uses top-1 pred.

    Returns list of (input_tensor, true_label_or_None, image_path).
    """
    import glob
    paths = sorted(glob.glob(os.path.join(data_dir, "*.JPEG")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(data_dir, "**/*.JPEG"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No JPEG images in {data_dir}")

    labels_map = {}
    if label_file and os.path.isfile(label_file):
        with open(label_file) as f:
            for i, line in enumerate(f, start=1):
                labels_map[i] = int(line.strip()) - 1  # 0-indexed

    # Limit sample count
    if num_samples and num_samples < len(paths):
        rng = random.Random(42)
        paths = rng.sample(paths, num_samples)
        paths.sort()

    records = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"[skip] {p}: {e}")
            continue
        # Derive 1-based index from filename if possible
        basename = os.path.basename(p)
        idx = None
        try:
            # e.g. ILSVRC2012_val_00000001.JPEG or ILSVRC2012_test_00000001.JPEG
            idx = int(basename.split("_")[-1].split(".")[0])
        except Exception:
            pass
        gt = labels_map.get(idx, None)
        records.append((tensor, gt, p))

    return records


def run_evaluation(
    model, records, device, patch_size, n_steps, n_sg_samples,
    methods=None, verbose=True,
):
    if methods is None:
        methods = list(ATTRIBUTORS.keys())

    # results[method] = {"morf_auc": [], "lerf_auc": []}
    results = {m: {"morf_auc": [], "lerf_auc": []} for m in methods}

    for idx, (input_tensor, gt_label, path) in enumerate(records):
        # Determine prediction class
        with torch.no_grad():
            logits = model(input_tensor)
            pred_class = int(torch.argmax(logits, dim=1).item())

        if verbose and (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(records)}] pred={pred_class}", flush=True)

        for method in methods:
            attr_fn = ATTRIBUTORS[method]
            try:
                attr_map = attr_fn(
                    input_tensor=input_tensor,
                    pred_class=pred_class,
                    model=model,
                    device=device,
                    n_steps=n_steps,
                    n_samples=n_sg_samples,
                )
            except Exception as e:
                print(f"[warn] {method} attribution failed on {os.path.basename(path)}: {e}")
                continue

            torch.cuda.empty_cache()

            ranked = build_patch_ranking(attr_map, patch_size)

            morf_curve = evaluate_curve(
                model, input_tensor, pred_class, ranked,
                patch_size, device, reverse=False
            )
            lerf_curve = evaluate_curve(
                model, input_tensor, pred_class, ranked,
                patch_size, device, reverse=True
            )

            results[method]["morf_auc"].append(auc_curve(morf_curve))
            results[method]["lerf_auc"].append(auc_curve(lerf_curve))

        # Clear gradients
        torch.cuda.empty_cache()

    return results


def summarise(results):
    summary = {}
    for method, data in results.items():
        morf_vals = data["morf_auc"]
        lerf_vals = data["lerf_auc"]
        if not morf_vals:
            continue
        morf = float(np.mean(morf_vals))
        lerf = float(np.mean(lerf_vals))
        abpc = morf - lerf
        summary[method] = {"MoRF": morf, "LeRF": lerf, "ABPC": abpc}
    return summary


def print_table(summary):
    header = f"{'Method':<12}  {'MoRF':>8}  {'LeRF':>8}  {'ABPC':>8}"
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for method, vals in summary.items():
        print(f"{method:<12}  {vals['MoRF']:>8.4f}  {vals['LeRF']:>8.4f}  {vals['ABPC']:>8.4f}")
    print("=" * len(header))
    print()
    # Paper Table 1 reference values (ResNet-50, from paper):
    print("Paper Table 1 reference (ResNet-50):")
    paper = {
        "constant": (None, None, None),
        "random":   (0.597, None, 0.007),
        "gradcam":  (0.628, None, 0.371),
        "ig":       (0.623, None, 0.418),
        "sg":       (None,  None, None),
    }
    for m, (morf, lerf, abpc) in paper.items():
        vals = []
        for v in [morf, lerf, abpc]:
            vals.append(f"{v:>8.3f}" if v is not None else f"{'N/A':>8}")
        print(f"  {m:<12} MoRF={vals[0]}  LeRF={vals[1]}  ABPC={vals[2]}")
    print()


def save_csv(summary, output_csv, num_samples, patch_size, n_steps):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "MoRF", "LeRF", "ABPC",
                         "num_samples", "patch_size", "n_steps"])
        for method, vals in summary.items():
            writer.writerow([
                method,
                f"{vals['MoRF']:.6f}",
                f"{vals['LeRF']:.6f}",
                f"{vals['ABPC']:.6f}",
                num_samples, patch_size, n_steps,
            ])
    print(f"Saved → {output_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Table 1 replication: ResNet-50, MoRF + LeRF + ABPC"
    )
    p.add_argument("--data_dir",   default="data",
                   help="Directory containing ImageNet JPEG images")
    p.add_argument("--label_file", default=None,
                   help="Path to ILSVRC2012_validation_ground_truth.txt "
                        "(1 label per line, 1-indexed). Omit for test set.")
    p.add_argument("--num_samples", type=int, default=100,
                   help="Number of images to evaluate (default: 100)")
    p.add_argument("--patch_size",  type=int, default=16,
                   help="Patch size for masking (default: 16 → 14×14 grid)")
    p.add_argument("--n_steps",     type=int, default=50,
                   help="IG integration steps (default: 50)")
    p.add_argument("--n_sg_samples",type=int, default=50,
                   help="SmoothGrad noise samples (default: 50)")
    p.add_argument("--methods",     nargs="+",
                   default=["constant", "random", "gradcam", "ig", "sg"],
                   choices=list(ATTRIBUTORS.keys()),
                   help="Attribution methods to evaluate")
    p.add_argument("--output_csv",  default="results_table1.csv",
                   help="Output CSV path")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Methods: {args.methods}")
    print(f"Patch size: {args.patch_size}  n_steps: {args.n_steps}  "
          f"n_sg_samples: {args.n_sg_samples}")

    print("Loading ResNet-50 (ImageNet-1K weights)...")
    model = load_resnet50(device)

    preprocess = build_preprocess()

    print(f"Loading images from: {args.data_dir}")
    records = load_images_with_labels(
        args.data_dir, args.label_file,
        args.num_samples, preprocess, device
    )
    print(f"Loaded {len(records)} images.")

    if not records:
        print("No images loaded. Exiting.")
        sys.exit(1)

    print(f"\nRunning evaluation...")
    results = run_evaluation(
        model, records, device,
        patch_size=args.patch_size,
        n_steps=args.n_steps,
        n_sg_samples=args.n_sg_samples,
        methods=args.methods,
    )

    summary = summarise(results)
    print_table(summary)
    save_csv(summary, args.output_csv,
             num_samples=len(records),
             patch_size=args.patch_size,
             n_steps=args.n_steps)


if __name__ == "__main__":
    main()
