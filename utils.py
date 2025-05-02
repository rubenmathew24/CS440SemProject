import torch
from torch.utils.data import Dataset, DataLoader
import config
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score
from tqdm import tqdm
from termcolor import colored
import random
import numpy as np
import os

# ===== Methods to grab .txt info =====

# Grab all Category Names
def get_all_categories():
	ALL_CATEGORIES = []
	with open("all_categories.txt", "r") as f:
		ALL_CATEGORIES = f.read().split()
	return ALL_CATEGORIES
	
		
# Uses prewritten Category Lengths so that we don't need to load all the datasets into memory
# Grabs the percentage each category appears in. Parallel to ALL_CATEGORIES
def get_category_occurrences():
	CATEGORY_OCCURRENCES = []

	temp = []
	with open("category_nums.txt", "r") as f:
		temp = f.read().split()
		
	temp = list(map(int, temp))
	total = sum(temp)

	for num in temp:
		CATEGORY_OCCURRENCES.append(num/total)

	return CATEGORY_OCCURRENCES


# ===== ReviewDataset Class =====

# Got idea to use a custom dataset class like this from GPT
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
	

# ===== Other Useful Methods =====

# I generated this with GPT in an attempt to create easily reproducible results
def set_random_seeds():
    seed = config.RANDOM_SEED

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    return shuffle_generator


def get_log_location(learning_type, size, is_binary):
    model_type = "binary" if is_binary else "multi"
    save_path = f"models/{learning_type}/{size}/{model_type}_{size}/results.txt"

    # Make sure folder exists
    folders = ["models", f"models/{learning_type}", f"models/{learning_type}/{size}", f"models/{learning_type}/{size}/{model_type}_{size}"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    return save_path


def logger(text_and_color=[], location=""):
    if location == "":
        print(colored("No log location specified", color="red"))
        return
    
    with open(location, "a") as f:
        for text, color in text_and_color:
            print(colored(text, color=color), end=" ")
            print(text, file=f, end=" ")

        print("", )
        print("", file=f)


def save_model(model, tokenizer, learning_type, size, is_binary):
    model_type = "binary" if is_binary else "multi"
    save_path = f"models/{learning_type}/{size}/{model_type}_{size}"

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")
    
# Returns accuracy, top-k accuracy, precision, f1 for given model and size
def full_multi_eval(model, tokenizer, size):
    test_csv = f"data/{size}/{size}_test.csv"
    test_df = pd.read_csv(test_csv)

    test_texts = test_df['review_text'].tolist()
    test_labels = test_df['category_label'].tolist()

    batch_size = config.BATCH_SIZES[size]
    test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, config.MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ===== Evaluation =====
    model.eval()
    preds = []
    true_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Full Evaluation', leave=False, colour="magenta"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())

    # Compute metrics
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, preds, average='weighted')

    # Compute top-k accuracy
    all_logits = torch.cat(all_logits, dim=0)
    topk_preds = torch.topk(all_logits, k=config.TOP_K, dim=1).indices

    correct_topk = 0
    for idx, label in enumerate(true_labels):
        if label in topk_preds[idx]:
            correct_topk += 1

    topk_accuracy = correct_topk / len(true_labels)

    # ===== Results =====
    return accuracy, topk_accuracy, precision, f1

# Returns accuracy, precision, f1 for given model and size
def full_binary_eval(model, tokenizer, size):
    test_csv = f"data/{size}/{size}_test.csv"
    test_df = pd.read_csv(test_csv)

    test_texts = test_df['review_text'].tolist()
    test_labels = test_df['binary_label'].tolist()

    batch_size = config.BATCH_SIZES[size]
    test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, config.MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ===== Evaluation =====
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating', leave=False, colour="magenta"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # ===== Metrics =====
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, zero_division=0)
    f1 = f1_score(true_labels, preds)

    return accuracy, precision, f1
