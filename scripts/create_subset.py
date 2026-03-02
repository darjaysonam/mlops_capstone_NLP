import os
import shutil

import pandas as pd

CSV_PATH = "data/raw/Data_Entry_2017_v2020.csv"
IMAGE_DIR = "data/raw/images"
OUTPUT_DIR = "data/processed/images"
SUBSET_SIZE = 6000

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Keep only images that exist in images_002 folder
available_images = set(os.listdir(IMAGE_DIR))
df = df[df["Image Index"].isin(available_images)]

# Select subset
df_subset = df.sample(n=SUBSET_SIZE, random_state=42)

# Save new CSV
df_subset.to_csv("data/processed/subset.csv", index=False)

# Copy images
for img in df_subset["Image Index"]:
    shutil.copy(os.path.join(IMAGE_DIR, img), os.path.join(OUTPUT_DIR, img))

print("Subset created successfully.")
