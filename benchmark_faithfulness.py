import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel

def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_image_paths(data_dir):
    pattern = os.path.join(data_dir, "ILSVRC2012_test_*.JPEG")
    return sorted(glob.glob(pattern))

def compute_attributions(model, input_tensor, pred_class, method_name, device):
    """
    Computes patch-level attributions (14x14 grid) for a given attribution method.
    """
    patch_size = 16
    grid_h, grid_w = 14, 14

    if method_name == 'constant':
        return np.ones((grid_h, grid_w), dtype=np.float32)
    elif method_name == 'random':
        return np.random.rand(grid_h, grid_w).astype(np.float32)

    baseline = torch.zeros_like(input_tensor).to(device)

    if method_name == 'ig': # Integrated Gradients
        ig = IntegratedGradients(model)
        attr = ig.attribute(input_tensor, baselines=baseline, target=pred_class, n_steps=30, internal_batch_size=2)
    elif method_name == 'saliency': # Vanilla Gradient
        sal = Saliency(model)
        attr = sal.attribute(input_tensor, target=pred_class)
    elif method_name == 'smoothgrad': # SmoothGrad
        sal = Saliency(model)
        nt = NoiseTunnel(sal)
        attr = nt.attribute(input_tensor, target=pred_class, nt_type='smoothgrad', nt_samples=10, stdevs=0.1)
    else:
        raise ValueError(f"Unknown method {method_name}")

    torch.cuda.empty_cache()

    # Aggregate pixel-level attributions to 14x14 grid
    attr_map = attr.abs().sum(dim=1).squeeze(0).cpu().detach().numpy()
    patch_map = np.zeros((grid_h, grid_w), dtype=np.float32)

    for ph in range(grid_h):
        for pw in range(grid_w):
            h_s, h_e = ph * patch_size, (ph + 1) * patch_size
            w_s, w_e = pw * patch_size, (pw + 1) * patch_size
            patch_map[ph, pw] = attr_map[h_s:h_e, w_s:w_e].mean()

    return patch_map

def evaluate_perturbation(model, input_tensor, pred_class, patch_map, mode='morf'):
    """
    Evaluates perturbation curve for MoRF or LeRF.
    Returns:
      morf_morf_curve: list of predicted probabilities at step k=0..10
      scalar_score: mean probability drop over all steps k=1..10 (Eq. 1 in paper)
    """
    patch_size = 16
    grid_h, grid_w = 14, 14
    total_patches = grid_h * grid_w

    patch_scores = []
    for ph in range(grid_h):
        for pw in range(grid_w):
            patch_scores.append((patch_map[ph, pw], ph, pw))

    # Sort: MoRF (descending, highest first), LeRF (ascending, lowest first)
    reverse_sort = (mode == 'morf')
    patch_scores.sort(key=lambda x: x[0], reverse=reverse_sort)

    percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    with torch.no_grad():
        orig_logits = model(input_tensor)
        orig_prob = torch.softmax(orig_logits, dim=1)[0, pred_class].item()

    curve = [orig_prob]
    drops = []

    for p in percentages[1:]:
        num_to_mask = int(p * total_patches)
        patches_to_mask = patch_scores[:num_to_mask]

        perturbed = input_tensor.clone()
        for _, ph, pw in patches_to_mask:
            h_s, h_e = ph * patch_size, (ph + 1) * patch_size
            w_s, w_e = pw * patch_size, (pw + 1) * patch_size
            perturbed[0, :, h_s:h_e, w_s:w_e] = 0.0

        with torch.no_grad():
            pert_logits = model(perturbed)
            pert_prob = torch.softmax(pert_logits, dim=1)[0, pred_class].item()

        curve.append(pert_prob)
        drops.append(orig_prob - pert_prob)

    # Scalar score (Eq 1): Average probability drop over steps
    scalar_score = np.mean(drops)
    return curve, scalar_score

def run_benchmark(model_name, num_samples=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==================================================")
    print(f"Running Faithfulness Benchmark for Model: {model_name} on {device}")
    print(f"==================================================")

    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    elif model_name == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
    else:
        raise ValueError(f"Unknown model {model_name}")

    model.eval().to(device)
    preprocess = get_transforms()
    image_paths = get_image_paths('data')[:num_samples]

    methods = ['constant', 'random', 'saliency', 'ig', 'smoothgrad']
    
    results = {m: {'morf_scalar': [], 'lerf_scalar': [], 'abpc_scalar': []} for m in methods}
    curves = {m: {'morf': [], 'lerf': []} for m in methods}

    for idx, img_path in enumerate(image_paths):
        raw_img = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(raw_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()

        for method in methods:
            patch_map = compute_attributions(model, input_tensor, pred_class, method, device)
            
            morf_curve, morf_val = evaluate_perturbation(model, input_tensor, pred_class, patch_map, mode='morf')
            lerf_curve, lerf_val = evaluate_perturbation(model, input_tensor, pred_class, patch_map, mode='lerf')
            
            # ABPC (Area Between Perturbation Curves) = MoRF_drop - LeRF_drop
            # For a faithful method, MoRF drop is large and LeRF drop is small, resulting in a high positive ABPC score.
            abpc_val = morf_val - lerf_val

            results[method]['morf_scalar'].append(morf_val)
            results[method]['lerf_scalar'].append(lerf_val)
            results[method]['abpc_scalar'].append(abpc_val)

            curves[method]['morf'].append(morf_curve)
            curves[method]['lerf'].append(lerf_curve)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
            print(f"Processed {idx + 1}/{len(image_paths)} images.")

    # Summary Table
    print(f"\n--- Benchmark Results Table ({model_name.upper()}) ---")
    print(f"{'Method':<15} | {'MoRF Drop (↑)':<15} | {'LeRF Drop (↓)':<15} | {'ABPC Score (↑)':<15}")
    print("-" * 65)

    summary_table = {}
    for method in methods:
        m_morf = np.mean(results[method]['morf_scalar'])
        m_lerf = np.mean(results[method]['lerf_scalar'])
        m_abpc = np.mean(results[method]['abpc_scalar'])
        summary_table[method] = {'MoRF': m_morf, 'LeRF': m_lerf, 'ABPC': m_abpc}
        print(f"{method:<15} | {m_morf:<15.4f} | {m_lerf:<15.4f} | {m_abpc:<15.4f}")

    # Plot Comparison Curves
    percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.figure(figsize=(10, 6))
    for method in methods:
        mean_morf = np.mean(curves[method]['morf'], axis=0)
        plt.plot(percentages, mean_morf, label=f"{method.upper()} (MoRF)")
    
    plt.xlabel("Fraction of Patches Masked")
    plt.ylabel("Mean Predicted Probability")
    plt.title(f"M4 Benchmark MoRF Comparison ({model_name.upper()})")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"morf_comparison_{model_name}.png")
    print(f"\nPlot saved to morf_comparison_{model_name}.png")

    return summary_table

if __name__ == '__main__':
    run_benchmark('resnet50', num_samples=30)
    run_benchmark('vgg16', num_samples=30)
