import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from captum.attr import IntegratedGradients

def load_vgg16_model(device):
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    model.eval()
    model.to(device)
    return model, weights

def get_transforms():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess

def get_image_paths(data_dir):
    pattern = os.path.join(data_dir, "ILSVRC2012_test_*.JPEG")
    image_paths = sorted(glob.glob(pattern))
    return image_paths

def evaluate_morf_vgg16(model, image_paths, preprocess, device, num_samples=30, patch_size=16):
    model.eval()
    ig = IntegratedGradients(model)
    
    all_morf_curves = []
    num_eval_images = min(num_samples, len(image_paths))
    print(f"Evaluating VGG16 MoRF on {num_eval_images} images...")

    grid_h = 224 // patch_size
    grid_w = 224 // patch_size
    total_patches = grid_h * grid_w

    for idx, img_path in enumerate(image_paths[:num_eval_images]):
        try:
            raw_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        input_tensor = preprocess(raw_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            orig_prob = torch.softmax(output, dim=1)[0, pred_class].item()

        # Baseline: zero tensor
        baseline = torch.zeros_like(input_tensor).to(device)

        # Compute Integrated Gradients
        attributions = ig.attribute(
            input_tensor,
            baselines=baseline,
            target=pred_class,
            n_steps=30,
            internal_batch_size=2
        )
        torch.cuda.empty_cache()

        # Channel-wise magnitude of attributions: shape (224, 224)
        attr_map = attributions.abs().sum(dim=1).squeeze(0).cpu().detach().numpy()

        # Calculate average attribution for each patch
        patch_attributions = []
        for ph in range(grid_h):
            for pw in range(grid_w):
                h_start, h_end = ph * patch_size, (ph + 1) * patch_size
                w_start, w_end = pw * patch_size, (pw + 1) * patch_size
                patch_attr = attr_map[h_start:h_end, w_start:w_end].mean()
                patch_attributions.append((patch_attr, ph, pw))

        # Sort patches by attribution descending (Most Relevant First)
        patch_attributions.sort(key=lambda x: x[0], reverse=True)

        percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        morf_curve = [orig_prob]

        # Perturbation evaluation
        for p in percentages[1:]:
            num_to_mask = int(p * total_patches)
            patches_to_mask = patch_attributions[:num_to_mask]

            perturbed_tensor = input_tensor.clone()
            for _, ph, pw in patches_to_mask:
                h_start, h_end = ph * patch_size, (ph + 1) * patch_size
                w_start, w_end = pw * patch_size, (pw + 1) * patch_size
                # Mask with zero (black background in normalized space)
                perturbed_tensor[0, :, h_start:h_end, w_start:w_end] = 0.0

            with torch.no_grad():
                pert_output = model(perturbed_tensor)
                pert_prob = torch.softmax(pert_output, dim=1)[0, pred_class].item()

            morf_curve.append(pert_prob)

        all_morf_curves.append(morf_curve)

        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{num_eval_images} images.")

    mean_morf = np.mean(all_morf_curves, axis=0)
    return mean_morf

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = 'data'
    image_paths = get_image_paths(data_dir)
    print(f"Found {len(image_paths)} ImageNet test images.")

    if len(image_paths) == 0:
        print(f"No test images found in {data_dir}!")
        return

    print("Loading pre-trained VGG16 model...")
    model, weights = load_vgg16_model(device)
    preprocess = get_transforms()

    mean_morf = evaluate_morf_vgg16(model, image_paths, preprocess, device, num_samples=30)

    print("\nVGG16 Mean MoRF Curve (Probability of top predicted class as features are masked):")
    percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for p, prob in zip(percentages, mean_morf):
        print(f"Masked {p*100:.0f}%: Prob = {prob:.4f}")

    # Save Plot
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, mean_morf, marker='s', color='red', label='VGG16 MoRF (Integrated Gradients)')
    plt.xlabel("Fraction of Patches Masked")
    plt.ylabel("Predicted Probability")
    plt.title("Faithfulness Evaluation: VGG16 MoRF")
    plt.grid(True)
    plt.legend()
    plt.savefig("morf_curve_vgg16.png")
    print("\nPlot saved to morf_curve_vgg16.png")

if __name__ == '__main__':
    main()
