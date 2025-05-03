import pandas as pd
import os
import config
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
from utils import ReviewDataset
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, logging
from termcolor import colored
from tqdm import tqdm
import utils
import time
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import argparse
import numpy as np
from sklearn.cluster import KMeans
import gc


# HELPS CLEAR WARNINGS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()

# ===== Config Settings =====
SEED = config.RANDOM_SEED
EPOCHS = config.EPOCHS
BATCH_SIZES = config.BATCH_SIZES
MAX_LENGTH = config.MAX_LENGTH
LEARNING_RATE = config.LEARNING_RATE

DEVICE = config.DEVICE
NUM_ROUNDS = 5


# ===== Global Variables =====
SIZE = ""
IS_BINARY = None
LEARNING_TYPE = ""

def initialize_pools(df, label_col):
    init_size = len(df) // NUM_ROUNDS

    labeled_pool, unlabeled_pool = train_test_split(
        df,
        train_size=init_size,
        stratify=df[label_col],
        random_state=SEED
    )

    return labeled_pool, unlabeled_pool

def active_learning_loop(sample_functions):
    # Load Data
    train_csv = f"data/{SIZE}/{SIZE}_train.csv"
    train_df = pd.read_csv(train_csv)

    test_csv = f"data/{SIZE}/{SIZE}_test.csv"
    test_df = pd.read_csv(test_csv)

    label_col = 'binary_label' if IS_BINARY else 'category_label'

    total_training_time = 0

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_size = len(train_df)
    query_batch_size = train_size // NUM_ROUNDS

    # Initialize pools
    labeled_df, unlabeled_df = initialize_pools(train_df, label_col)
    log_location = utils.get_log_location(LEARNING_TYPE, SIZE, IS_BINARY)

    # Choose sampling function
    query_function = sample_functions[LEARNING_TYPE]


    for round_id in range(NUM_ROUNDS):
        utils.logger([("\n=== Round", "white"), (f"{round_id+1}", "cyan"), (f"/ {NUM_ROUNDS} ===", "white")], log_location)

        utils.logger([("Training...", "cyan")], log_location)
        utils.logger([("\tLabeled samples:", "white"), (f"{len(labeled_df)}", "cyan")], log_location)
        utils.logger([("\tUnlabeled pool:", "white"), (f"{len(unlabeled_df)}", "cyan")], log_location)

        # Train model on labeled pool
        model, training_time = train_model(labeled_df, tokenizer, label_col)
        total_training_time += training_time

        # Evaluate on test set
        accuracy, f1 = eval_model(test_df, model, tokenizer, label_col)
        
        utils.logger([("Evaluating...", "magenta")], log_location)
        utils.logger([("\tAccuracy:", "white"), (f"{accuracy:.4f}", "magenta")], log_location)
        utils.logger([("\tF1 score:", "white"), (f"{f1:.4f}", "magenta")], log_location)
        utils.logger([("\tTraining Time:", "white"), (f"{total_training_time:.2f}", "magenta"), ("seconds", "white")], log_location)

        # Break if done
        if round_id == NUM_ROUNDS - 1 or len(unlabeled_df) == 0:
            break

        # Active Learning: Query most uncertain samples
        utils.logger([("Sampling...", "yellow")], log_location)
        queried_df, sampling_time = query_function(model, unlabeled_df, tokenizer, query_batch_size)
        total_training_time += sampling_time

        # Shift queried samples to labeled_df
        labeled_df = pd.concat([labeled_df, queried_df])
        unlabeled_df = unlabeled_df.drop(index=queried_df.index)

    return model, tokenizer, total_training_time

def train_model(train_df, tokenizer, label_col):
    # Load data
    train_texts = train_df['review_text'].tolist()
    train_labels = train_df[label_col].tolist()

    # Number of classes
    if not IS_BINARY:
        num_classes = len(utils.get_all_categories())
    else:
        num_classes = 2

    num_epochs = EPOCHS[SIZE]
    batch_size = BATCH_SIZES[SIZE]

    # Tokenizer and Dataset
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=SHUFFLE_GENERATOR)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes, ignore_mismatched_sizes=True)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Actually start training
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

    return model, training_time

def eval_model(test_df, model, tokenizer, label_col):
    test_texts = test_df['review_text'].tolist()
    test_labels = test_df[label_col].tolist()

    batch_size = BATCH_SIZES[SIZE]
    test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ===== Evaluation =====
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating', leave=False, colour="magenta"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='binary' if IS_BINARY else 'weighted')

    return accuracy, f1

def least_confidence_query(model, unlabeled_df, tokenizer, query_batch_size):
    model.eval()
    texts = unlabeled_df["review_text"].tolist()
    dummy_labels = [0] * len(texts)

    batch_size = BATCH_SIZES[SIZE]

    dataset = ReviewDataset(texts, dummy_labels, tokenizer, config.MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size)

    uncertainties = []
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Sampling", leave=False, colour='yellow'):
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = softmax(outputs.logits, dim=1)
            max_probs, _ = probs.max(dim=1)

            # Least confidence
            uncertainty = 1 - max_probs  
            uncertainties.extend(uncertainty.cpu().tolist())

    # Get top-k uncertain indices
    sorted_indices = sorted(range(len(uncertainties)), key=lambda i: uncertainties[i], reverse=True)
    topk_indices = sorted_indices[:query_batch_size]

    sampling_time = time.time() - start_time

    return unlabeled_df.iloc[topk_indices], sampling_time

def custom_query(model, unlabeled_df, tokenizer, query_batch_size):
    model.eval()
    texts = unlabeled_df["review_text"].tolist()
    dummy_labels = [0] * len(texts)

    batch_size = BATCH_SIZES[SIZE]
    dataset = ReviewDataset(texts, dummy_labels, tokenizer, config.MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size)

    entropies = []
    logits_list = []

    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Sampling", leave=False, colour='yellow'):
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = softmax(logits, dim=1)

            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

            entropies.extend(entropy.cpu().tolist())
            logits_list.append(logits.cpu())

    entropies = np.array(entropies)
    embeddings = torch.cat(logits_list, dim=0).numpy()

    # Step 1: Top-N uncertain samples using np.argpartition
    top_n = min(3 * query_batch_size, len(unlabeled_df))
    top_entropy_indices = np.argpartition(entropies, -top_n)[-top_n:]
    top_embeddings = embeddings[top_entropy_indices]

    # Step 3: KMeans
    num_unique = len(np.unique(top_embeddings, axis=0))
    effective_k = min(query_batch_size, num_unique)
    kmeans = KMeans(n_clusters=effective_k, n_init="auto", random_state=42)
    kmeans.fit(top_embeddings)
    cluster_centers = kmeans.cluster_centers_
    distances = np.linalg.norm(top_embeddings[:, None, :] - cluster_centers[None, :, :], axis=2)
    closest_indices = np.argmin(distances, axis=0)
    selected_indices = top_entropy_indices[closest_indices]

    # Cleanup memory
    torch.cuda.empty_cache()
    gc.collect()

    sampling_time = time.time() - start_time
    return unlabeled_df.iloc[selected_indices], sampling_time

def final_evaluation(model, tokenizer, training_time):
    log_location = utils.get_log_location(LEARNING_TYPE, SIZE, IS_BINARY)

    if IS_BINARY:
        # ===== Metrics =====
        accuracy, precision, f1 = utils.full_binary_eval(model, tokenizer, SIZE)

        # ===== Logging =====
        utils.logger([("\n===== Results =====", "white")], log_location)
        utils.logger([("Accuracy:      ", "white"), (f"{accuracy:.4f}", "magenta")], log_location)
        utils.logger([("Precision:     ", "white"), (f"{precision:.4f}", "magenta")], log_location)
        utils.logger([("F1 Score:      ", "white"), (f"{f1:.4f}", "magenta")], log_location)
        utils.logger([("Training Time: ", "white"), (f"{training_time:.2f}", "magenta"), ("seconds", "white")], log_location)

    else:
        # ===== Metrics =====
        accuracy, topk_accuracy, precision, f1 = utils.full_multi_eval(model, tokenizer, SIZE)

        # ===== Logging =====
        utils.logger([("\n===== Results =====", "white")], log_location)
        utils.logger([("Accuracy:                      ", "white"), (f"{accuracy:.4f}", "magenta")], log_location)
        utils.logger([(f"Top-{config.TOP_K} Accuracy:                ", "white"), (f"{topk_accuracy:.4f}", "magenta")], log_location)
        utils.logger([("Weighted Precision:            ", "white"), (f"{precision:.4f}", "magenta")], log_location)
        utils.logger([("Weighted F1 Score:             ", "white"), (f"{f1:.4f}", "magenta")], log_location)
        utils.logger([("Training Time:                 ", "white"), (f"{training_time:.2f}", "magenta"), ("seconds", "white")], log_location)

def create_log():
    num_epochs = EPOCHS[SIZE]
    batch_size = BATCH_SIZES[SIZE]

    # Start Logging
    log_location = utils.get_log_location(LEARNING_TYPE, SIZE, IS_BINARY)
    open(log_location, 'w').close() # RESETS LOG FILE
    utils.logger([("\n===============================", "white")], log_location)
    utils.logger([("MODEL DETAILS", "white")], log_location)
    utils.logger([("Learning type:", "white"), (LEARNING_TYPE, "cyan")], log_location)
    utils.logger([("Size type:", "white"), (SIZE, "cyan")], log_location)
    utils.logger([("Model type:", "white"), ("binary" if IS_BINARY else "multi", "cyan")], log_location)
    utils.logger([("Epochs:", "white"), (num_epochs, "cyan")], log_location)
    utils.logger([("Batch Size:", "white"), (batch_size, "cyan")], log_location)
    utils.logger([("===============================", "white")], log_location)

def main():
    global IS_BINARY, SIZE, LEARNING_TYPE, SHUFFLE_GENERATOR
    parser = argparse.ArgumentParser(description="Process type and size arguments.")

    parser.add_argument('--type', choices=['binary', 'multi', 'all'], default='all', help='Specify type (default: all)')
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'all'], default='all', help='Specify size (default: all)')
    parser.add_argument('--learning', choices=['AL1', 'AL2', 'all'], default='all', help='Specify Learning Type (default: all)')
    args = parser.parse_args()

    # Setup
    all_sizes = ["small", "medium", "large"]
    all_types = ["binary", "multi"]
    all_learning_types = ["AL1", "AL2"]
    size_description = "Using All Dataset Sizes"
    type_description = "Using Both Binary and Multi-Classification"
    learning_description = "Using All Learning types"

    if args.size != 'all':
        all_sizes = [args.size]
        size_description = f"Using just {args.size} dataset"

    if args.type != 'all':
        all_types = [args.type]
        type_description = f"Just {args.type} classification"

    if args.learning != 'all':
        all_learning_types = [args.learning]
        learning_description = f"Using just {args.learning}"

    sample_functions = {
        "AL1" : least_confidence_query,
        "AL2" : custom_query
    }
        

    
    for learning_type in tqdm(all_learning_types, desc=learning_description, colour="green", leave=False):
        print()
        for size in tqdm(all_sizes, desc=size_description, colour="green", leave=False):
            print()
            for model_type in tqdm(all_types, desc=type_description, colour="green", leave=False):
                print()
                SIZE = size
                IS_BINARY = True if model_type == 'binary' else False
                LEARNING_TYPE = learning_type
                SHUFFLE_GENERATOR = utils.set_random_seeds()

                create_log()

                # Train Model
                model, tokenizer, training_time = active_learning_loop(sample_functions)

                # Save Model
                utils.save_model(model, tokenizer, LEARNING_TYPE, SIZE, IS_BINARY)

                # Evaluate Model
                final_evaluation(model, tokenizer, training_time)

if __name__ == "__main__":
    main()
