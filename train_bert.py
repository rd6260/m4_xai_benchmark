import os
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def load_data(data_dir):
    docs_dir = os.path.join(data_dir, 'docs')
    
    def load_split(split_name):
        samples = []
        labels = []
        with open(os.path.join(data_dir, f'{split_name}.jsonl'), 'r') as f:
            for line in f:
                item = json.loads(line)
                annotation_id = item['annotation_id']
                label = 1 if item['classification'] == 'POS' else 0
                
                with open(os.path.join(docs_dir, annotation_id), 'r') as doc_f:
                    text = doc_f.read().strip()
                
                samples.append(text)
                labels.append(label)
        return Dataset.from_dict({'text': samples, 'label': labels})

    train_dataset = load_split('train')
    val_dataset = load_split('val')
    test_dataset = load_split('test')

    return DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    })

def main():
    print("Loading dataset...")
    dataset = load_data('data/movies')
    
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=50,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating...")
    trainer.evaluate()
    
    print("Saving model...")
    trainer.save_model('./models/bert-base-movies')
    tokenizer.save_pretrained('./models/bert-base-movies')

if __name__ == '__main__':
    main()
