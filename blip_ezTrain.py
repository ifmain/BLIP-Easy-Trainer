import os
import argparse
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, load_metric
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
import numpy as np
import gc

class CustomDataset(TorchDataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return preprocess_data(self.dataset[idx], self.processor)

def load_data(data_dir):
    img_paths, text_data = [], []
    total_files = sum(len(files) for _, _, files in os.walk(data_dir))

    with tqdm(total=total_files, desc="Loading data") as pbar:
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.jpeg'):
                    img_paths.append(file_path)
                elif file.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_data.append(f.read())
                pbar.update(1)
    return img_paths, text_data

def preprocess_data(example, processor):
    try:
        image = Image.open(example['image']).convert("RGB")
        inputs = processor(images=image, text=example['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs.update({'labels': inputs.input_ids.clone()})
        return inputs
    except Exception as e:
        print(f"Problem with image file {example['image']}: {e}")
        return None

def collate_fn(batch):
    batch = [b for b in batch if b]  # Убираем пустые элементы
    if not batch:
        return {}
    
    input_ids = torch.cat([b['input_ids'] for b in batch])
    attention_mask = torch.cat([b['attention_mask'] for b in batch])
    pixel_values = torch.cat([b['pixel_values'] for b in batch])
    labels = torch.cat([b['labels'] for b in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels
    }

def main(path_to_model, subgroups_count, data_dir, training_args):
    img_paths, text_data = load_data(data_dir)

    if len(img_paths) != len(text_data):
        raise ValueError("The number of images and texts does not match.")

    dataset = Dataset.from_dict({'image': img_paths, 'text': text_data})
    del img_paths, text_data  # Free your RAM
    gc.collect()

    processor = BlipProcessor.from_pretrained(path_to_model)
    train_subgroups = np.array_split(dataset, subgroups_count)  # Optimization for low RAM GPU
    model = BlipForConditionalGeneration.from_pretrained(path_to_model).to("cuda")

    for i, subgroup in enumerate(train_subgroups):
        print(f"Training on subgroup {i + 1}/{len(train_subgroups)}")

        dataset = CustomDataset(subgroup, processor)
        train_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collate_fn
        )

        trainer.train()
        torch.cuda.empty_cache()

    model.save_pretrained("blip-image2promt-stable-diffusion-v0.2")
    processor.save_pretrained("blip-image2promt-stable-diffusion-v0.2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BLIP model on a custom dataset.")
    parser.add_argument("--path_to_model", type=str, default='Salesforce/blip-image-captioning-base', help="Path to the pretrained BLIP model")
    parser.add_argument("--subgroups_count", type=int, default=4, help="Number of subgroups to split the dataset for training")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")

    # TrainingArguments parameters
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the results")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device during evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimization")
    parser.add_argument("--logging_dir", type=str, default='./logs', help="Directory for logging")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total amount of checkpoints")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps")
    parser.add_argument("--remove_unused_columns", type=bool, default=False, help="Remove unused columns")

    args = parser.parse_args()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        remove_unused_columns=args.remove_unused_columns
    )

    main(args.path_to_model, args.subgroups_count, args.data_dir, training_args)
