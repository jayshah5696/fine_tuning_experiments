# Import necessary libraries
import sys
import os
import argparse
import logging
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# Function to check if a dataset already exists in the specified directory
def check_dataset_exists(data_dir, dataset_name):
    # Load the dataset and get its split names
    splits = load_dataset(dataset_name).keys()
    # Check if all splits of the dataset exist in the data directory
    return all(os.path.exists(os.path.join(data_dir, split)) for split in splits)

# Function to download a dataset from Hugging Face and save it to disk
def download_hf_dataset(data_dir, dataset_name):
    # Define the data directory path, replacing '/' with '_' in the dataset name
    data_dir = os.path.join(data_dir, dataset_name.replace('/', '_'))
    
    # Check if the dataset is already downloaded
    if check_dataset_exists(data_dir, dataset_name):
        # If the dataset is already downloaded, log a message and return
        logging.info(f"Dataset '{dataset_name}' already downloaded in '{data_dir}'.")
        return

    # If the dataset is not downloaded, download and save it
    for split in tqdm(load_dataset(dataset_name).keys(), desc="Downloading"):
        try:
            # Load the specific split of the dataset
            dataset = load_dataset(dataset_name, split=split)
            # Define the directory for this split
            split_dir = os.path.join(data_dir, split)
            # If the directory for this split does not exist, create it
            os.makedirs(split_dir, exist_ok=True)
            # Save the split to disk
            dataset.save_to_disk(split_dir)
            # Log a message indicating the split has been saved
            logging.info(f"Saved {split} split to {split_dir}")
            # Save as CSV
            df = pd.DataFrame(dataset)
            csv_path = os.path.join(split_dir, f"{split}.csv")
            df.to_csv(csv_path, index=False)
            tqdm.write(f"Saved {split} split as CSV to {csv_path}")
            
        except Exception as e:
            # Log any errors that occur during the download or save process
            logging.error(f"Error downloading/saving {split} split: {e}")

# Main execution
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Download a dataset from Hugging Face.')
    parser.add_argument('dataset_name', help='The name of the dataset to download.')
    parser.add_argument('--data_dir', default='data/', help='The directory to save the dataset in.')
    # Parse the command line arguments
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Download the dataset
    download_hf_dataset(args.data_dir, args.dataset_name)