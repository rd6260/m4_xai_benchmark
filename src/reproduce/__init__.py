"""
Mini-reproduction of Table 1 of "M4: A Unified XAI Benchmark for Faithfulness
Evaluation of Feature Attribution Methods" (NeurIPS 2023) on ResNet-50.

Implements: Constant / Random-16 / Random baselines, GradCAM, IG, SmoothGrad
Metrics   : MoRF, ABPC (= LeRF-curve minus MoRF-curve), INFD, PScore
Not here  : SynScore (needs re-training on a synthetic-patch ImageNet subset)

NOTE: written for PyTorch + Captum (the paper uses PaddlePaddle + InterpretDL,
so absolute numbers will differ; compare the *ranking/pattern*).

    pip install torch torchvision captum pillow numpy
    python m4_resnet50_repro.py --val_dir /path/to/imagenet/val --n_images 200

val_dir layout: one sub-folder per class (torchvision ImageFolder style).
"""
import argparse, random
import numpy as np, torch, torch.nn.functional as F
from torchvision import models, transforms, datasets
from captum.attr import IntegratedGradients, NoiseTunnel, LayerGradCam
from captum.metrics import infidelity

dev = "cuda" if torch.cuda.is_available() else "cpu"
P = 16                       # patch size used for perturbation
STEPS = 14                   # number of perturbation levels (of 196 patches)


# ---------------------------------------------------------------- data
def get_loader(val_dir, n_images, seed=0):
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = datasets.ImageFolder(val_dir, tf)
    random.seed(seed)
    idx = random.sample(range(len(ds)), n_images)   # paper: 5 imgs/class, 5000 total
    return [ds[i][0] for i in idx]


# ---------------------------------------------------------------- explainers
def to_patch(attr):                       # (1,C or 1,H,W) -> (14,14) patch scores
    a = attr.abs().sum(1, keepdim=True) if attr.shape[1] > 1 else attr
    return F.avg_pool2d(a, P).squeeze()


def make_explainers(model):
    ig = IntegratedGradients(model)
    sg = NoiseTunnel(IntegratedGradients(model))     # replaced below by plain SG
    from captum.attr import Saliency
    sal = NoiseTunnel(Saliency(model))
    gc = LayerGradCam(model, model.layer4)

    def e_const(x, t):  return torch.ones(1, 1, 224, 224, device=dev)
    def e_rand16(x, t): return F.interpolate(torch.rand(1, 1, 14, 14, device=dev), scale_factor=P)
    def e_rand(x, t):   return torch.rand(1, 1, 224, 224, device=dev)
    def e_gc(x, t):
        a = gc.attribute(x, target=t)
        return F.interpolate(a, size=224, mode="bilinear")
    def e_ig(x, t):     return ig.attribute(x, target=t, n_steps=20, internal_batch_size=20)
    def e_sg(x, t):     # SmoothGrad: mean of gradients under Gaussian noise
        return sal.attribute(x, nt_type="smoothgrad", nt_samples=20, stdevs=0.15,
                             target=t, abs=False)
    return {"Constant": e_const, "Random-16": e_rand16, "Random": e_rand,
            "GradCAM": e_gc, "IG": e_ig, "SG": e_sg}


# ---------------------------------------------------------------- metrics
@torch.no_grad()
def prob(model, x, t):
    return F.softmax(model(x), 1)[:, t]


@torch.no_grad()
def perturb_curve(model, x, t, order, descending):
    """Mask (set to 0 == dataset mean after normalisation) the top-k / bottom-k patches."""
    order = order if descending else order[::-1]
    n = len(order)
    ks = np.linspace(0, n, STEPS + 1).astype(int)
    xs = []
    for k in ks:
        xp = x.clone()
        for j in order[:k]:
            r, c = divmod(int(j), 14)
            xp[..., r*P:(r+1)*P, c*P:(c+1)*P] = 0
        xs.append(xp)
    return F.softmax(model(torch.cat(xs)), 1)[:, t]      # (STEPS+1,)


def morf_abpc(model, x, t, expl):
    ps = to_patch(expl).flatten().cpu().numpy()
    order = list(np.argsort(-ps, kind="stable"))          # most relevant first
    morf_c = perturb_curve(model, x, t, order, True)
    lerf_c = perturb_curve(model, x, t, order, False)
    morf = (morf_c[0] - morf_c).mean().item()             # Eq.(1)
    abpc = (lerf_c - morf_c).mean().item()                # Eq.(3)
    return morf, abpc


def infd(model, x, t, expl, sigma=0.1):
    def perturb_fn(inp):
        noise = torch.randn_like(inp) * sigma
        return noise, inp - noise
    if expl.shape[1] == 1:                                # broadcast spatial maps to 3 channels
        expl = expl.expand_as(x)
    return infidelity(model, perturb_fn, x, expl, target=t, n_perturb_samples=10).item()


def pscore(expl_test, expl_others):
    """cosine( mean of normalised attributions from other models, test attribution )"""
    def norm(a):
        a = to_patch(a).flatten()
        return (a - a.mean()) / (a.std() + 1e-8)
    gt = torch.stack([norm(e) for e in expl_others]).mean(0)
    return F.cosine_similarity(gt, norm(expl_test), dim=0).item()


# ---------------------------------------------------------------- main
def main(a):
    imgs = get_loader(a.val_dir, a.n_images)
    tgt = models.resnet50(weights="IMAGENET1K_V2" if False else "IMAGENET1K_V1").eval().to(dev)
    # consensus models for PScore (paper uses 9 image models; we use a lighter set)
    others = [models.resnet101(weights="IMAGENET1K_V1"), models.resnet152(weights="IMAGENET1K_V1"),
              models.vgg16(weights="IMAGENET1K_V1"), models.mobilenet_v3_large(weights="IMAGENET1K_V1"),
              models.densenet121(weights="IMAGENET1K_V1")]
    others = [m.eval().to(dev) for m in others]

    ex_t = make_explainers(tgt)
    ig_others = [IntegratedGradients(m) for m in others]
    res = {k: {"MoRF": [], "ABPC": [], "INFD": [], "PScore": []} for k in ex_t}

    for i, x in enumerate(imgs):
        x = x.unsqueeze(0).to(dev)
        with torch.no_grad():
            t = tgt(x).argmax(1).item()
        # consensus for PScore, computed with IG on the other models (same target class)
        cons = [g.attribute(x, target=t, n_steps=20, internal_batch_size=20).detach()
                for g in ig_others]
        for name, fn in ex_t.items():
            e = fn(x, t).detach()
            m, ab = morf_abpc(tgt, x, t, e)
            res[name]["MoRF"].append(m); res[name]["ABPC"].append(ab)
            res[name]["INFD"].append(infd(tgt, x, t, e))
            if name in ("GradCAM", "IG", "SG"):
                res[name]["PScore"].append(pscore(e, cons))
        if (i + 1) % 20 == 0: print(f"{i+1}/{len(imgs)} done")

    print(f"\n{'Method':10s} {'MoRF↑':>8s} {'ABPC↑':>8s} {'PScore↑':>8s} {'INFD↓':>8s}")
    for k, v in res.items():
        ps = f"{np.mean(v['PScore']):8.3f}" if v["PScore"] else "     N/A"
        print(f"{k:10s} {np.mean(v['MoRF']):8.3f} {np.mean(v['ABPC']):8.3f} {ps} {np.mean(v['INFD']):8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--n_images", type=int, default=200)
    main(ap.parse_args())
