import pandas as pd
from datasets import load_dataset
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import config
import utils

# Imported from config file
CACHED_LOCATION = config.OFFLOADED_DATASET_LOCATION
RANDOM_STATE_SEED = config.RANDOM_SEED
DATASET_SIZES = config.DATASET_SIZES

# Populated via methods
ALL_CATEGORIES = utils.get_all_categories()
CATEGORY_OCCURENCES = utils.get_category_occurrences()
        
# Grabs one Datset
def get_one_set(set="All_Beauty"):
	dataset = load_dataset(
          "McAuley-Lab/Amazon-Reviews-2023", 
          f"raw_review_{set}", 
          split="full", 
          trust_remote_code=True, 
          cache_dir=CACHED_LOCATION, 
    )
	return dataset

# Grab all the datasets (Should only be run if datasets are not yet downloaded/cached)
def get_full_database():
	reviews = {}

	for category in tqdm(ALL_CATEGORIES, desc="Loading categories", unit="category", colour="green"):
		dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}", split="full", trust_remote_code=True, cache_dir=CACHED_LOCATION, )
		reviews[category] = dataset
	
	return reviews

# Presents statistics for full database
def get_database_stats(database):

	print("DATABASE STATISTICS")
	print("=====================================")
	print("Number of categories: ", len(database))
	print("Number of reviews: ", sum([len(data) for data in database.values()]))

	print("======================================")
	print(f"{'Category Name':<40} {'# of Reviews':>25}")


	for category, data in database.items():
		print(f"{category:<40} {len(data):>25}")


# Cull each category set
def sample_reviews_for_dataset(target_size):
    all_samples = []

    print(f"Target dataset size: {target_size}")

    label2id = {label:i for i,label in enumerate(sorted(ALL_CATEGORIES))}

    for category, occurrence in tqdm(zip(ALL_CATEGORIES, CATEGORY_OCCURENCES), desc="Sampling per category", total=len(ALL_CATEGORIES), colour="cyan"):
        n_samples = max(2, int(target_size * occurrence))  # get at least 2 reviews

        dataset = get_one_set(set=category)

        df = dataset.to_pandas()
        df['category_text'] = category
        df['category_label'] = label2id[category]
        df['binary_label'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
        df['review_text'] = df['title'] + ": " + df['text']

        if len(df) < n_samples:
            print(f"Warning: category {category} has only {len(df)} reviews, requested {n_samples}")
            sampled_df = df
        else:
            sampled_df = df.sample(n=n_samples, random_state=RANDOM_STATE_SEED)

        all_samples.append(sampled_df[['review_text', 'rating', 'category_text', 'binary_label', 'category_label']])

    combined_df = pd.concat(all_samples, ignore_index=True)

    # Trim to exactly target_size if oversampled
    if len(combined_df) > target_size:
        combined_df = combined_df.sample(n=target_size, random_state=RANDOM_STATE_SEED)

    return combined_df

# Split into 80/20 train/test where the proportions are similar to the original set
def stratified_train_test_split(df, stratify_column='category_label'):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[stratify_column],
        random_state=RANDOM_STATE_SEED,
    )
    return train_df, test_df


# Main Function to partition
def partition_dataset(size_name):
    if size_name not in DATASET_SIZES.keys():
        raise Exception("Not a defined dataset size")
    
    size = DATASET_SIZES[size_name]

    print(f"\nBuilding {size_name} dataset with {size} reviews...")

    # Create the directory if it doesn't exist
    save_dir = os.path.join("data", size_name)
    os.makedirs(save_dir, exist_ok=True)

    sampled_df = sample_reviews_for_dataset(size)
    train_df, test_df = stratified_train_test_split(sampled_df)

    # Save
    train_df.to_csv(f"data/{size_name}/{size_name}_train.csv", index=False)
    test_df.to_csv(f"data/{size_name}/{size_name}_test.csv", index=False)
    print(f"{size_name.capitalize()} dataset saved: Train={len(train_df)}, Test={len(test_df)}")

    return train_df, test_df

# Presents stats for already partitioned set
def csv_dataset_stats(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    print(f"\nDataset Stats for {csv_path}")
    print("=========================================")
    print(f"Total rows: {len(df)}")
    print("\nRows per category:")
    print(df['category_label'].value_counts().sort_index())
    print("=========================================")
    print(df['binary_label'].value_counts().sort_index())

    return


def main():
    parser = argparse.ArgumentParser(description="Manage Amazon Review Dataset Processing")

    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument('--partition', nargs='?', const='all', default=None, type=str, help="Partition and save dataset")
    group.add_argument('--stats', nargs='?', const='all', default=None, type=str, help="Get dataset stats")

    args = parser.parse_args()

    if not args.partition and not args.stats:
        print(f"Downloading and Caching full database to {CACHED_LOCATION}")
        database = get_full_database()
        get_database_stats(database)
        return

    if args.partition:
        size = args.partition.lower()
        if size != 'all' and size not in DATASET_SIZES:
            raise Exception(f"Invalid size '{size}'")
        
        print(f"Running partition_dataset('{size}')...")
        if size == 'all':
            for s in tqdm(DATASET_SIZES.keys(), desc="Partitioning Datasets", colour="green"):
                partition_dataset(s)
        else:
            partition_dataset(size)

    if args.stats:
        size = args.stats.lower()
        if size not in DATASET_SIZES:
            raise Exception(f"Invalid stats size '{size}'")
        
        train_csv_file_path = f"data/{size}/{size}_train.csv"
        test_csv_file_path = f"data/{size}/{size}_test.csv"

        print(f"{size.upper()} TRAINING DATA STATISTICS")
        csv_dataset_stats(train_csv_file_path)
        print(f"{size.upper()} TEST DATA STATISTICS")
        csv_dataset_stats(test_csv_file_path)

if __name__ == "__main__":
    main()

