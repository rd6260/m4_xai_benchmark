import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

def load_test_data(data_dir):
    docs_dir = os.path.join(data_dir, 'docs')
    samples = []
    labels = []
    with open(os.path.join(data_dir, 'test.jsonl'), 'r') as f:
        for line in f:
            item = json.loads(line)
            annotation_id = item['annotation_id']
            label = 1 if item['classification'] == 'POS' else 0
            with open(os.path.join(docs_dir, annotation_id), 'r') as doc_f:
                text = doc_f.read().strip()
            samples.append(text)
            labels.append(label)
    return samples, labels

def custom_forward(inputs, attention_mask, model):
    outputs = model(inputs_embeds=inputs, attention_mask=attention_mask)
    return outputs.logits

def evaluate_morf(model, tokenizer, texts, labels, device, max_len=256):
    model.eval()
    
    # We will use the model's embeddings layer for Captum
    embeddings = model.bert.embeddings.word_embeddings
    
    ig = IntegratedGradients(custom_forward)
    
    all_morf_curves = []
    
    print(f"Evaluating {len(texts)} samples...")
    for idx, (text, label) in enumerate(zip(texts, labels)):
        if idx >= 50: # Just evaluate on 50 samples for speed during replication
            break
            
        encoded = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            pred_class = torch.argmax(outputs.logits, dim=1).item()
            orig_prob = torch.softmax(outputs.logits, dim=1)[0, pred_class].item()
            
        input_embeds = embeddings(input_ids)
        
        # Compute attributions
        # Baseline is pad token embedding
        pad_ids = torch.full_like(input_ids, fill_value=tokenizer.pad_token_id)
        pad_embeds = embeddings(pad_ids)
        
        attributions, delta = ig.attribute(
            inputs=input_embeds, 
            baselines=pad_embeds, 
            additional_forward_args=(attention_mask, model),
            target=pred_class,
            n_steps=50,
            internal_batch_size=2,
            return_convergence_delta=True
        )
        torch.cuda.empty_cache()
        
        # Sum across embedding dimensions to get token-level attribution
        attributions_sum = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        
        # Only consider actual tokens (ignore padding)
        valid_length = attention_mask.sum().item()
        
        # Ignore [CLS] and [SEP]
        valid_attributions = attributions_sum[1:valid_length-1]
        
        # Sort indices by attribution (Most Relevant First)
        sorted_indices = np.argsort(-np.abs(valid_attributions)) + 1 # +1 to account for [CLS]
        
        # Compute MoRF curve
        percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        morf_curve = [orig_prob]
        
        for p in percentages:
            num_to_mask = int(p * len(valid_attributions))
            mask_indices = sorted_indices[:num_to_mask]
            
            perturbed_ids = input_ids.clone()
            for mask_idx in mask_indices:
                perturbed_ids[0, mask_idx] = tokenizer.pad_token_id # Mask with PAD
                
            with torch.no_grad():
                pert_outputs = model(perturbed_ids, attention_mask=attention_mask)
                pert_prob = torch.softmax(pert_outputs.logits, dim=1)[0, pred_class].item()
                
            morf_curve.append(pert_prob)
            
        all_morf_curves.append(morf_curve)
        
        if (idx+1) % 10 == 0:
            print(f"Processed {idx+1} samples.")
            
    mean_morf = np.mean(all_morf_curves, axis=0)
    return mean_morf

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading test data...")
    texts, labels = load_test_data('data/movies')
    
    model_path = './models/bert-base-movies'
    if not os.path.exists(model_path):
        print("Model not found! Waiting for training to finish...")
        return
        
    print("Loading trained model...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    mean_morf = evaluate_morf(model, tokenizer, texts, labels, device)
    
    print("Mean MoRF Curve (Probability of predicted class as top features are masked):")
    x_axis = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for x, y in zip(x_axis, mean_morf):
        print(f"Masked {x*100}%: Prob = {y:.4f}")
        
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, mean_morf, marker='o', label='MoRF (Integrated Gradients)')
    plt.xlabel("Fraction of Features Masked")
    plt.ylabel("Predicted Probability")
    plt.title("Faithfulness Evaluation: MoRF")
    plt.grid(True)
    plt.legend()
    plt.savefig("morf_curve.png")
    print("Plot saved to morf_curve.png")

if __name__ == '__main__':
    main()
