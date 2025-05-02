import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, logging
import pandas as pd
from tqdm import tqdm
import argparse
import os
import config
import utils
from utils import ReviewDataset

# HELPS CLEAR WARNINGS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()

# ===== Global Variables =====
SIZE = ""
IS_BINARY = None

# ===== Training Settings =====
LEARNING_RATE = config.LEARNING_RATE
MAX_LENGTH = config.MAX_LENGTH
TOP_K = config.TOP_K

BATCH_SIZES = config.BATCH_SIZES
EPOCHS = config.EPOCHS

DEVICE = config.DEVICE

SHUFFLE_GENERATOR = None


def create_baseline_model():
    # Load data
    train_csv = f"data/{SIZE}/{SIZE}_train.csv"
    train_df = pd.read_csv(train_csv)

    label_type = 'binary_label' if IS_BINARY else 'category_label'
    
    train_texts = train_df['review_text'].tolist()
    train_labels = train_df[label_type].tolist()

    # Number of classes
    if not IS_BINARY:
        num_classes = len(utils.get_all_categories())
    else:
        num_classes = 2

    num_epochs = EPOCHS[SIZE]
    batch_size = BATCH_SIZES[SIZE]

    # Tokenizer and Dataset
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=SHUFFLE_GENERATOR)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes, ignore_mismatched_sizes=True)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Actually start training
    log_location = utils.get_log_location("baseline", SIZE, IS_BINARY)
    utils.logger([("Starting training...\n", "cyan")], log_location)

    start_time = time.time()
    model.train()

    for epoch in range(num_epochs):
        loop = tqdm(train_loader, leave=False, colour="cyan")
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch {epoch+1}')
            loop.set_postfix(loss=loss.item())

    training_time = time.time() - start_time
    utils.logger([(f"\nTraining completed in {training_time:.2f} seconds", "cyan")], log_location)

    return model, tokenizer, training_time
      
def eval_binary_model(model, tokenizer, training_time):
    log_location = utils.get_log_location("baseline", SIZE, IS_BINARY)
    utils.logger([("Evaluating...", "magenta")], log_location)

    # ===== Metrics =====
    accuracy, precision, f1 = utils.full_binary_eval(model, tokenizer, SIZE)

    # ===== Logging =====
    utils.logger([("\n===== Results =====", "white")], log_location)
    utils.logger([("Accuracy:      ", "white"), (f"{accuracy:.4f}", "magenta")], log_location)
    utils.logger([("Precision:     ", "white"), (f"{precision:.4f}", "magenta")], log_location)
    utils.logger([("F1 Score:      ", "white"), (f"{f1:.4f}", "magenta")], log_location)
    utils.logger([("Training Time: ", "white"), (f"{training_time:.2f}", "magenta"), ("seconds", "white")], log_location)

def eval_multi_model(model, tokenizer, training_time):
    log_location = utils.get_log_location("baseline", SIZE, IS_BINARY)
    utils.logger([("Evaluating...", "magenta")], log_location)

    # ===== Metrics =====
    accuracy, topk_accuracy, precision, f1 = utils.full_multi_eval(model, tokenizer, SIZE)

    # ===== Logging =====
    utils.logger([("\n===== Results =====", "white")], log_location)
    utils.logger([("Accuracy:                      ", "white"), (f"{accuracy:.4f}", "magenta")], log_location)
    utils.logger([(f"Top-{TOP_K} Accuracy:                ", "white"), (f"{topk_accuracy:.4f}", "magenta")], log_location)
    utils.logger([("Weighted Precision:            ", "white"), (f"{precision:.4f}", "magenta")], log_location)
    utils.logger([("Weighted F1 Score:             ", "white"), (f"{f1:.4f}", "magenta")], log_location)
    utils.logger([("Training Time:                 ", "white"), (f"{training_time:.2f}", "magenta"), ("seconds", "white")], log_location)

def create_log():
    num_epochs = EPOCHS[SIZE]
    batch_size = BATCH_SIZES[SIZE]

    # Start Logging
    log_location = utils.get_log_location("baseline", SIZE, IS_BINARY)
    open(log_location, 'w').close() # RESETS LOG FILE
    utils.logger([("\n===============================", "white")], log_location)
    utils.logger([("MODEL DETAILS", "white")], log_location)
    utils.logger([("Learning type:", "white"), ("baseline", "cyan")], log_location)
    utils.logger([("Size type:", "white"), (SIZE, "cyan")], log_location)
    utils.logger([("Model type:", "white"), ("binary" if IS_BINARY else "multi", "cyan")], log_location)
    utils.logger([("Epochs:", "white"), (num_epochs, "cyan")], log_location)
    utils.logger([("Batch Size:", "white"), (batch_size, "cyan")], log_location)
    utils.logger([("===============================", "white")], log_location)

def main():
    global IS_BINARY, SIZE, SHUFFLE_GENERATOR
    parser = argparse.ArgumentParser(description="Process type and size arguments.")

    parser.add_argument('--type', choices=['binary', 'multi', 'all'], default='all', help='Specify type (default: all)')
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'all'], default='all', help='Specify size (default: all)')
    args = parser.parse_args()

    # Setup
    all_sizes = ["small", "medium", "large"]
    all_types = ["binary", "multi"]
    size_description = "Using All Dataset Sizes"
    type_description = "Using Both Binary and Multi-Classification"

    if args.size != 'all':
        all_sizes = [args.size]
        size_description = f"Using just {args.size} dataset"

    if args.type != 'all':
        all_types = [args.type]
        type_description = f"Just {args.type} classification"
        

    
    for size in tqdm(all_sizes, desc=size_description, colour="green", leave=False):
        print()
        for model_type in tqdm(all_types, desc=type_description, colour="green", leave=False):
            print()
            SIZE = size
            IS_BINARY = True if model_type == 'binary' else False
            SHUFFLE_GENERATOR = utils.set_random_seeds()

            create_log()

            # Train Model
            model, tokenizer, training_time = create_baseline_model()

            # Save Model
            utils.save_model(model, tokenizer, "baseline", SIZE, IS_BINARY)

            # Evaluate Model
            eval_func = eval_binary_model if IS_BINARY else eval_multi_model
            eval_func(model, tokenizer, training_time)


if __name__ == "__main__":
    main()
