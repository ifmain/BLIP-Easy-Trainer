Here's a detailed `README.md` for your script named `blip_ezTrain.py`:

```markdown
# blip_ezTrain.py

A script for training a BLIP (Bootstrapping Language-Image Pre-training) model on custom datasets with configurable training parameters using command line arguments. This script leverages the HuggingFace Transformers library and PyTorch for efficient processing and training.

## Features

- Load and preprocess image and text data from a specified directory.
- Train the BLIP model with configurable training parameters.
- Split large datasets into smaller subgroups for training on low RAM GPUs.
- Save the trained model and processor for future use.

## Usage

Run the script with the following command:

# Example tree

```
project-root/
├── data/
│   ├── image1/
│   │   ├── image1.jpeg
│   │   ├── image1.txt
│   ├── image2/
│   │   ├── image2.jpeg
│   │   ├── image2.txt
│   ├── image3/
│   │   ├── image3.jpeg
│   │   ├── image3.txt
│   ├── image4/
│   │   ├── image4.jpeg
│   │   ├── image4.txt
```

```sh
python blip_ezTrain.py --data_dir /path/to/data --path_to_model Salesforce/blip-image-captioning-base --subgroups_count 4 --output_dir ./results --learning_rate 2e-5 --num_train_epochs 5
```

### Command Line Arguments

- `--data_dir` (str): Directory containing the dataset (required).
- `--path_to_model` (str): Path to the pretrained BLIP model (default: `Salesforce/blip-image-captioning-base`).
- `--subgroups_count` (int): Number of subgroups to split the dataset for training (default: `4`).
- `--output_dir` (str): Directory to save the results (default: `./results`).
- `--learning_rate` (float): Learning rate for training (default: `2e-5`).
- `--per_device_train_batch_size` (int): Batch size per device during training (default: `1`).
- `--per_device_eval_batch_size` (int): Batch size per device during evaluation (default: `1`).
- `--num_train_epochs` (int): Number of training epochs (default: `5`).
- `--weight_decay` (float): Weight decay for optimization (default: `0.01`).
- `--logging_dir` (str): Directory for logging (default: `./logs`).
- `--logging_steps` (int): Logging steps (default: `10`).
- `--save_total_limit` (int): Limit the total amount of checkpoints (default: `2`).
- `--save_steps` (int): Save checkpoint every X updates steps (default: `500`).
- `--remove_unused_columns` (bool): Remove unused columns (default: `False`).

## Contributing

Feel free to submit issues and enhancement requests.
