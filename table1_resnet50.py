"""
Table 1 Replication — M4 XAI Benchmark (NeurIPS 2023)
=======================================================
Target: ResNet-50, ImageNet-val 5000-image subset

EXACT metric formulas (from medical_image_example.ipynb in the repo):

  MoRF  = mean_images( mean_k( probas[0] - MoRF_probas[k] ) for k=0..K )
         = mean_images( probas[0] - mean(MoRF_probas) )
         Note: probas[0] = original unmasked image probability
         ↑ higher = more faithful (important pixels removed → big prob drop)

  ABPC  = mean_images( mean_k( LeRF_probas[k] - MoRF_probas[k] ) )
         ↑ higher = more faithful (area between perturbation curves)

  LeRF (not reported in paper) analogous to MoRF with least-relevant-first order.

Perturbation implementation (from InterpretDL perturbation.py):
  • Images stored as uint8 [N,H,W,C] in [0,255] RGB
  • Masking baseline: mx = (127, 127, 127) in [0,255] space
  • Then normalised: /255, -mean, /std
  • Percentile schedule (n=20 steps, cumulative):
        q  = 100/20 = 5
        qs = [95, 90, 85, …, 5, 0]      (19 down to 0, i.e. q*(i-1) descending)
  • MoRF: iterate qs in order, mask pixels > p  (cumulative)
  • LeRF: iterate qs REVERSED [0, 5, …, 95], mask pixels < p (cumulative)
  • class-of-interest = argmax(model(original_image))

Attribution params matched to paper (from run_expl.sh):
  SmoothGrad: noise_amount=0.1, n_samples=100
  IntGrad:    num_random_trials=10, baselines=random, steps=50
  GradCAM:    target_layer_name=layer4.2.relu  (= ResNet-50 layer4[-1])

Paper Table-1 reference (ResNet-50):
  Method       MoRF↑   ABPC↑
  Constant     0.000   0.000
  Random-16    0.596   0.007
  Random       0.599   0.008
  GradCAM      0.628   0.424
  IG           0.709   0.377
  SG           0.701   0.369
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

try:
    from captum.attr import IntegratedGradients, NoiseTunnel, Saliency
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("[warn] captum not found — ig/sg will be skipped")

try:
    from pytorch_grad_cam import GradCAM as _GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    print("[warn] pytorch-grad-cam not found — gradcam will use grad×input fallback")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MX_UINT8      = 127.0    # grey masking baseline in [0,255] space
N_STEPS       = 20


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O — exactly matching InterpretDL's pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _resize_short(img_hwc: np.ndarray, target: int) -> np.ndarray:
    """Resize so the shorter side = target, keeping aspect ratio."""
    h, w = img_hwc.shape[:2]
    if h < w:
        new_h, new_w = target, max(1, int(w * target / h))
    else:
        new_h, new_w = max(1, int(h * target / w)), target
    return np.array(Image.fromarray(img_hwc).resize((new_w, new_h), Image.BILINEAR))


def _center_crop(img_hwc: np.ndarray, size: int) -> np.ndarray:
    h, w = img_hwc.shape[:2]
    top  = (h - size) // 2
    left = (w - size) // 2
    return img_hwc[top:top+size, left:left+size]


def read_image_uint8(img_path: str, resize_to: int = 224, crop_to: int = 224) -> np.ndarray:
    """Returns (H, W, 3) uint8 numpy RGB — matches InterpretDL's read_image()."""
    with open(img_path, "rb") as f:
        img = Image.open(f).convert("RGB")
    img = np.array(img, dtype=np.uint8)
    img = _resize_short(img, resize_to)
    img = _center_crop(img, crop_to)
    return img  # (224, 224, 3) uint8


def preprocess_image(imgs_nhwc: np.ndarray) -> np.ndarray:
    """
    Exactly InterpretDL's preprocess_image():
      input : (N, H, W, 3)  uint8 or float  — RGB
      output: (N, 3, H, W)  float32 normalised
    """
    imgs = imgs_nhwc.astype(np.float32) / 255.0
    imgs = imgs.transpose(0, 3, 1, 2)           # (N, 3, H, W)
    imgs -= IMAGENET_MEAN.reshape(3, 1, 1)
    imgs /= IMAGENET_STD.reshape(3, 1, 1)
    return imgs


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation sample generation — exact port of InterpretDL's
# generate_samples_array() in perturbation.py
# ─────────────────────────────────────────────────────────────────────────────

def generate_morf_lerf_images(
    img_nhwc: np.ndarray,     # (1, H, W, 3) uint8  — as returned by read_image
    attr_map: np.ndarray,     # (H, W) float32 — higher = more important
    n_steps: int = N_STEPS,
) -> tuple:
    """
    Returns:
        morf_nhwc : (n_steps+1, H, W, 3) uint8  — step 0 = original
        lerf_nhwc : (n_steps+1, H, W, 3) uint8

    Exactly mirrors InterpretDL generate_samples_array():
        q  = 100 / n_steps
        qs = [q*(n-1), q*(n-2), ..., q*0]  →  [95, 90, ..., 5, 0]
        percentiles = np.percentile(attr_map, qs)

        MoRF: for p in percentiles       → cumulative mask pixels > p
        LeRF: for p in percentiles[::-1] → cumulative mask pixels < p
    """
    q    = 100.0 / n_steps
    qs   = [q * (n_steps - 1 - i) for i in range(n_steps)]  # [95, 90, ..., 0]
    thrs = np.percentile(attr_map, qs)                        # shape (n_steps,)

    img_chw = img_nhwc[0]   # (H, W, 3) uint8 — InterpretDL stores as (H,W,C)

    # InterpretDL assigns mx to [channel] indices, img stored as (H,W,C) here
    def _mask_hwc(base_hwc: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
        out = base_hwc.copy()
        out[mask_hw] = MX_UINT8   # broadcast over channels for matching rows
        return out

    # ── MoRF (cumulative, most-relevant-first) ────────────────────────────
    morf_list = [img_chw.copy()]
    fudged = img_chw.copy()
    for p in thrs:            # descending: 95 → 0
        fudged = fudged.copy()
        mask = attr_map > p   # (H, W) bool
        fudged[mask] = MX_UINT8
        morf_list.append(fudged)

    # ── LeRF (cumulative, least-relevant-first) ───────────────────────────
    lerf_list = [img_chw.copy()]
    fudged = img_chw.copy()
    for p in thrs[::-1]:      # ascending: 0 → 95
        fudged = fudged.copy()
        mask = attr_map < p   # (H, W) bool
        fudged[mask] = MX_UINT8
        lerf_list.append(fudged)

    morf_nhwc = np.stack(morf_list, axis=0)   # (K+1, H, W, 3)
    lerf_nhwc = np.stack(lerf_list, axis=0)
    return morf_nhwc, lerf_nhwc


@torch.no_grad()
def run_model_on_batch(model, imgs_nhwc: np.ndarray, device, bs: int = 8) -> np.ndarray:
    """
    Runs model on (N, H, W, 3) uint8 numpy batch.
    Returns (N, n_classes) softmax probabilities.
    """
    data = preprocess_image(imgs_nhwc)   # (N, 3, H, W) float32
    all_probs = []
    for start in range(0, len(data), bs):
        chunk = torch.tensor(data[start:start+bs], dtype=torch.float32, device=device)
        p = F.softmax(model(chunk), dim=1).cpu().numpy()
        all_probs.append(p)
        del chunk
    return np.concatenate(all_probs)   # (N, n_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Attribution methods  →  (H, W) float32 saliency map
# ─────────────────────────────────────────────────────────────────────────────

def _agg_to_2d(attr_tensor: torch.Tensor, hw: tuple) -> np.ndarray:
    """
    attr_tensor: (1, C, H', W') — aggregate |attr| over channels → resize to hw.
    Matches InterpretDL: explanation = np.abs(explanation).sum(0) then cv2.resize.
    """
    a = attr_tensor.detach().cpu().numpy().squeeze(0)  # (C, H', W')
    a = np.abs(a).sum(axis=0)                           # (H', W')
    if a.shape != hw:
        a = cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return a.astype(np.float32)


def attr_constant(img_hwc, inp_norm, pred_cls, model, device, **kw) -> np.ndarray:
    """All-ones saliency → ties everywhere → no masking order → MoRF = 0."""
    h, w = img_hwc.shape[:2]
    return np.ones((h, w), dtype=np.float32)


def attr_random(img_hwc, inp_norm, pred_cls, model, device, rng=None, **kw) -> np.ndarray:
    """Pixel-level uniform random saliency."""
    h, w = img_hwc.shape[:2]
    rng = rng if rng is not None else np.random.default_rng()
    return rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)


def attr_random_16(img_hwc, inp_norm, pred_cls, model, device, rng=None, **kw) -> np.ndarray:
    """Patch-level (16×16) random saliency — 'Random-16' in Table 1."""
    h, w = img_hwc.shape[:2]
    P = 16
    rng = rng if rng is not None else np.random.default_rng()
    ph, pw = h // P, w // P
    patch_vals = rng.uniform(0.0, 1.0, size=(ph, pw)).astype(np.float32)
    return np.kron(patch_vals, np.ones((P, P), dtype=np.float32))


def attr_gradcam(img_hwc, inp_norm, pred_cls, model, device, **kw) -> np.ndarray:
    """Grad-CAM on layer4[-1] (= layer4.2.relu in ResNet-50)."""
    if HAS_GRADCAM:
        cam = _GradCAM(model=model, target_layers=[model.layer4[-1]])
        gc  = cam(input_tensor=inp_norm, targets=[ClassifierOutputTarget(pred_cls)])
        out = gc.squeeze(0).astype(np.float32)   # (224, 224)
        h, w = img_hwc.shape[:2]
        if out.shape != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return out
    # Fallback: gradient × input
    x = inp_norm.clone().requires_grad_(True)
    model(x)[0, pred_cls].backward()
    return _agg_to_2d(x.grad, img_hwc.shape[:2])


def attr_ig(img_hwc, inp_norm, pred_cls, model, device,
            ig_steps=50, ig_trials=10, **kw) -> np.ndarray:
    """
    Integrated Gradients with random baselines averaged over ig_trials.
    Matches InterpretDL: baselines='random', num_random_trials=10, steps=50.
    """
    if not HAS_CAPTUM:
        raise RuntimeError("captum required for ig")
    ig_attr  = IntegratedGradients(model)
    hw = img_hwc.shape[:2]
    attrs = []
    for _ in range(ig_trials):
        baseline = torch.randn_like(inp_norm) * 0.001  # small random noise baseline
        attr = ig_attr.attribute(inp_norm, baselines=baseline,
                                  target=pred_cls, n_steps=ig_steps,
                                  internal_batch_size=1)
        attrs.append(_agg_to_2d(attr, hw))
    return np.mean(np.stack(attrs, axis=0), axis=0).astype(np.float32)


def attr_sg(img_hwc, inp_norm, pred_cls, model, device,
            sg_samples=100, sg_stdev=0.1, **kw) -> np.ndarray:
    """
    SmoothGrad (NoiseTunnel over Saliency).
    Matches InterpretDL: noise_amount=0.1, n_samples=100.
    sg_stdev=0.1 is the noise_amount fraction applied to normalised input range.
    """
    if not HAS_CAPTUM:
        raise RuntimeError("captum required for sg")
    stdev = sg_stdev * (inp_norm.max() - inp_norm.min()).item()
    sal   = Saliency(model)
    nt    = NoiseTunnel(sal)
    attr  = nt.attribute(inp_norm, nt_samples=sg_samples,
                          nt_samples_batch_size=2,
                          stdevs=stdev, nt_type="smoothgrad_sq",
                          target=pred_cls)
    return _agg_to_2d(attr, img_hwc.shape[:2])


ATTRIBUTORS = {
    "constant":  attr_constant,
    "random":    attr_random,
    "random_16": attr_random_16,
    "gradcam":   attr_gradcam,
    "ig":        attr_ig,
    "sg":        attr_sg,
}


# ─────────────────────────────────────────────────────────────────────────────
# Model & dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_resnet50(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    return model


def load_records(data_dir: str):
    ds = ImageFolder(data_dir)
    return [(p, c) for p, c in ds.samples]


def make_norm_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model, records, device, args):
    methods  = args.methods
    n_steps  = args.n_steps
    results  = {m: {"morf": [], "lerf": [], "abpc": []} for m in methods}
    rng      = np.random.default_rng(args.seed)
    total    = len(records)
    t0       = time.time()
    norm_tfm = make_norm_transform()

    for img_idx, (img_path, _) in enumerate(records):

        # ── 1. Load uint8 image (matches InterpretDL read_image) ─────────────
        try:
            img_hwc = read_image_uint8(img_path, resize_to=args.resize_to, crop_to=args.crop_to)
        except Exception as e:
            print(f"[skip] {img_path}: {e}")
            continue

        img_nhwc = img_hwc[np.newaxis]  # (1, H, W, 3)

        # ── 2. Normalised tensor for gradient-based methods ──────────────────
        inp_norm = norm_tfm(Image.fromarray(img_hwc)).unsqueeze(0).to(device)  # (1,3,H,W)

        # ── 3. Predicted class (from original normalised image) ──────────────
        with torch.no_grad():
            pred_cls = int(torch.argmax(model(inp_norm), dim=1).item())

        # ── 4. Per-method ────────────────────────────────────────────────────
        for method in methods:
            attr_fn = ATTRIBUTORS[method]
            try:
                attr_map = attr_fn(
                    img_hwc=img_hwc, inp_norm=inp_norm,
                    pred_cls=pred_cls, model=model, device=device,
                    rng=rng,
                    ig_steps=args.ig_steps, ig_trials=args.ig_trials,
                    sg_samples=args.sg_samples, sg_stdev=args.sg_stdev,
                )
            except Exception as e:
                print(f"  [warn] {method} attr failed on {img_path}: {e}")
                continue

            # Resize attr_map to match image spatial dims if needed
            h, w = img_hwc.shape[:2]
            if attr_map.shape != (h, w):
                attr_map = cv2.resize(attr_map, (w, h), interpolation=cv2.INTER_LINEAR)

            torch.cuda.empty_cache()

            # ── 5. Generate MoRF/LeRF perturbed image stacks ─────────────────
            morf_nhwc, lerf_nhwc = generate_morf_lerf_images(img_nhwc, attr_map, n_steps)

            # ── 6. Run model on all stacks ────────────────────────────────────
            morf_probs_nc = run_model_on_batch(model, morf_nhwc, device)  # (K+1, C)
            lerf_probs_nc = run_model_on_batch(model, lerf_nhwc, device)

            morf_probas = morf_probs_nc[:, pred_cls]  # (K+1,)
            lerf_probas = lerf_probs_nc[:, pred_cls]

            # ── 7. Scores (from medical_image_example.ipynb — exact formula) ──
            # MoRF = mean drop: mean(probas[0] - MoRF_probas)
            # Note: probas[0] = original (unmasked) probability
            morf_score = float((morf_probas[0] - morf_probas).mean())
            lerf_score = float((lerf_probas[0] - lerf_probas).mean())
            # ABPC = mean(LeRF_probas - MoRF_probas)
            abpc_score = float((lerf_probas - morf_probas).mean())

            results[method]["morf"].append(morf_score)
            results[method]["lerf"].append(lerf_score)
            results[method]["abpc"].append(abpc_score)

            torch.cuda.empty_cache()

        # ── Progress ─────────────────────────────────────────────────────────
        if (img_idx + 1) % 100 == 0 or (img_idx + 1) == total:
            elapsed = time.time() - t0
            rate    = (img_idx + 1) / elapsed
            eta     = (total - img_idx - 1) / rate if rate > 0 else 0
            print(f"  [{img_idx+1}/{total}]  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m", flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

PAPER_TABLE1 = {
    # NeurIPS 2023 Table 1, ResNet-50 — (MoRF↑, ABPC↑, PScore↑, INFD↓, SynScore↑)
    "Constant":  (0.000, 0.000, None,  3.015, 0.072),
    "Random-16": (0.596, 0.007, None,  3.039, 0.077),
    "Random":    (0.599, 0.008, None,  3.015, 0.078),
    "GradCAM":   (0.628, 0.424, 0.835, 2.496, 1.000),
    "IG":        (0.709, 0.377, 0.812, 2.373, 0.999),
    "SG":        (0.701, 0.369, 0.820, 2.323, 0.998),
}


def summarise(results):
    out = {}
    for method, data in results.items():
        n = len(data["morf"])
        if n == 0:
            continue
        out[method] = {
            "MoRF":     float(np.mean(data["morf"])),
            "LeRF":     float(np.mean(data["lerf"])),
            "ABPC":     float(np.mean(data["abpc"])),
            "PScore":   float(np.mean(data["pscore"]))   if data.get("pscore")   else None,
            "INFD":     float(np.mean(data["infd"]))     if data.get("infd")     else None,
            "SynScore": float(np.mean(data["synscore"])) if data.get("synscore") else None,
            "n":        n,
        }
    return out


def print_table(summary):
    mc, ac, pc, ic, sc = 9, 9, 10, 9, 12

    def _fmt(v, w, d=3):
        return f"{v:>{w}.{d}f}" if v is not None else f"{'N/A':>{w}}"

    hdr = (f"{'Method':<16}  {'MoRF(^)':>{mc}}  {'ABPC(^)':>{ac}}"
           f"  {'PScore(^)':>{pc}}  {'INFD(v)':>{ic}}  {'SynScore(^)':>{sc}}  {'N':>6}")
    sep = "=" * len(hdr)

    print()
    print(sep)
    print("  REPRODUCED (ResNet-50)")
    print(sep)
    print(hdr)
    print("-" * len(hdr))
    for method, v in summary.items():
        print(f"{method:<16}  {_fmt(v['MoRF'], mc)}  {_fmt(v['ABPC'], ac)}"
              f"  {_fmt(v['PScore'], pc)}  {_fmt(v['INFD'], ic)}  {_fmt(v['SynScore'], sc)}"
              f"  {v['n']:>6}")
    print(sep)

    ph = (f"{'AttributionMethods':<20}  {'MoRF(^)':>{mc}}  {'ABPC(^)':>{ac}}"
          f"  {'PScore(^)':>{pc}}  {'INFD(v)':>{ic}}  {'SynScore(^)':>{sc}}")
    print()
    print(sep)
    print("  PAPER TABLE 1 (ResNet-50, NeurIPS 2023)")
    print(sep)
    print(ph)
    print("-" * len(ph))
    for m, (morf, abpc, pscore, infd, synscore) in PAPER_TABLE1.items():
        print(f"{m:<20}  {_fmt(morf, mc)}  {_fmt(abpc, ac)}"
              f"  {_fmt(pscore, pc)}  {_fmt(infd, ic)}  {_fmt(synscore, sc)}")
    print(sep)
    print()


def save_results(summary, output_dir: str, args):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(output_dir, "table1_resnet50.csv")

    def _csv_val(v):
        return f"{v:.3f}" if v is not None else ""

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "MoRF", "LeRF", "ABPC", "PScore", "INFD", "SynScore",
                    "N", "n_steps", "ig_steps", "ig_trials", "sg_samples", "sg_stdev", "seed"])
        for method, v in summary.items():
            w.writerow([
                method,
                _csv_val(v["MoRF"]),
                _csv_val(v["LeRF"]),
                _csv_val(v["ABPC"]),
                _csv_val(v["PScore"]),
                _csv_val(v["INFD"]),
                _csv_val(v["SynScore"]),
                v["n"],
                args.n_steps, args.ig_steps, args.ig_trials,
                args.sg_samples, args.sg_stdev, args.seed,
            ])
    print(f"Results saved → {csv_path}")



# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="M4 Table 1 replication — ResNet-50 (exact InterpretDL formula)"
    )
    p.add_argument("--data_dir",    default="data/M4_5000",
                   help="ImageFolder-style directory (synset subdirs)")
    p.add_argument("--methods",     nargs="+",
                   default=["constant", "random", "gradcam", "ig", "sg"],
                   choices=list(ATTRIBUTORS.keys()))
    p.add_argument("--n_steps",     type=int, default=N_STEPS,
                   help="Perturbation steps (paper: 20)")
    p.add_argument("--resize_to",   type=int, default=224)
    p.add_argument("--crop_to",     type=int, default=224)
    p.add_argument("--ig_steps",    type=int, default=50,
                   help="IG integration steps (paper: 50)")
    p.add_argument("--ig_trials",   type=int, default=10,
                   help="IG random baseline trials (paper: 10)")
    p.add_argument("--sg_samples",  type=int, default=100,
                   help="SmoothGrad noise samples (paper: 100)")
    p.add_argument("--sg_stdev",    type=float, default=0.1,
                   help="SmoothGrad noise_amount fraction (paper: 0.1)")
    p.add_argument("--output_dir",  default="results/table1_faithful")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--limit",       type=int, default=None,
                   help="Limit number of images (for quick testing)")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Methods    : {args.methods}")
    print(f"n_steps    : {args.n_steps}")
    print(f"ig_steps   : {args.ig_steps}  ig_trials: {args.ig_trials}")
    print(f"sg_samples : {args.sg_samples}  sg_stdev: {args.sg_stdev}")

    model = load_resnet50(device)
    print("ResNet-50 (IMAGENET1K_V1) loaded.")

    records = load_records(args.data_dir)
    print(f"Dataset    : {len(records)} images from {args.data_dir}")

    if args.limit:
        rng_s = random.Random(args.seed)
        rng_s.shuffle(records)
        records = records[:args.limit]
        print(f"           (limited to {len(records)} images)")

    print("\nEvaluating …")
    results = run_evaluation(model, records, device, args)

    summary = summarise(results)
    print_table(summary)
    save_results(summary, args.output_dir, args)


if __name__ == "__main__":
    main()
