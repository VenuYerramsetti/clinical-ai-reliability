# ====================================
# CLINICAL AI RELIABILITY PROJECT
# DATA LOADING PIPELINE
# ====================================


# ====================================
# DEBUG START
# ====================================

print("Script started...")


# ====================================
# IMPORTS
# ====================================

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from sklearn.model_selection import train_test_split


# ====================================
# PATH CONFIGURATION
# ====================================

# Root dataset directory
DATA_DIR = "data"

# Metadata CSV path
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")


# ====================================
# LOAD METADATA
# ====================================

# Read CSV into pandas DataFrame
metadata = pd.read_csv(METADATA_PATH)

# Preview first rows
print("\nMetadata Preview:")
print(metadata.head())

# Dataset dimensions
print("\nDataset shape:", metadata.shape)


# ====================================
# BUILD IMAGE PATHS
# ====================================

# HAM10000 images are split into 2 folders.
# This function checks both folders and returns
# the correct image path.


def get_image_path(image_id):

    part1 = os.path.join(
        DATA_DIR,
        "HAM10000_images_part_1",
        image_id + ".jpg"
    )

    part2 = os.path.join(
        DATA_DIR,
        "HAM10000_images_part_2",
        image_id + ".jpg"
    )

    # Check where image exists
    if os.path.exists(part1):
        return part1
    else:
        return part2


# Create new column containing full image paths
metadata["image_path"] = metadata["image_id"].apply(get_image_path)

# Preview image paths
print("\nImage paths:")
print(metadata[["image_id", "image_path"]].head())


# ====================================
# TEST IMAGE LOADING
# ====================================

# Select first image path
sample_image_path = metadata.iloc[0]["image_path"]

print("\nLoading sample image:")
print(sample_image_path)

# Open image using PIL
image = Image.open(sample_image_path)

# Print image details
print("\nImage successfully loaded!")

print("Image size:", image.size)
print("Image mode:", image.mode)