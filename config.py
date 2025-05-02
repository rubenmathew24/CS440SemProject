import torch

# ===== Training Settings =====
LEARNING_RATE = 1e-5
MAX_LENGTH = 256
TOP_K = 3

BATCH_SIZES = {
    "small": 16,
    "medium": 16,
    "large": 32
}

EPOCHS = {
    "small": 10,
    "medium": 5,
    "large": 2
}

# ===== Device =====
DEVICE = torch.device("cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")


# ===== Other Settings =====
RANDOM_SEED = 3141592
OFFLOADED_DATASET_LOCATION = "/Volumes/T7/Dataset/"
DATASET_SIZES = {
    'small': 1000,
    'medium': 15000,
    'large': 60000,
}