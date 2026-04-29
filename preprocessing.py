
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

DATA_DIR  = "/content/drive/MyDrive/Projet_Melanoma/data/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
CSV_PATH  = "/content/drive/MyDrive/Projet_Melanoma/data/ISIC_2019_Training_GroundTruth.csv"
SPLIT_DIR = "/content/drive/MyDrive/Projet_Melanoma/splits"
os.makedirs(SPLIT_DIR, exist_ok=True)

CLASS_COLS = ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_COLS)}
IMG_SIZE = 224

df = pd.read_csv(CSV_PATH)
df = df[df['UNK'] != 1.0].copy() # supresses images which were not classified (UNK = unknown category)
df['label'] = df[CLASS_COLS].idxmax(axis=1) # idxmax = 1 means belongs to the category
df['label_idx'] = df['label'].map(CLASS_TO_IDX)
df['filepath'] = df['image'].apply(lambda x: os.path.join(DATA_DIR, x + '.jpg'))
print(f"Total samples after cleaning: {len(df)}")


"""print(df.shape)
print(df.head(5))
print("\nColumn names:", df.columns.tolist())"""


print("Splitting data")
train_df, temp_df = train_test_split(df, test_size = 0.20, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

train_df[['image','label','label_idx','filepath']].to_csv(
    os.path.join(SPLIT_DIR, 'train.csv'), index=False)
val_df[['image','label','label_idx','filepath']].to_csv(
    os.path.join(SPLIT_DIR, 'validation.csv'), index=False)
test_df[['image','label','label_idx','filepath']].to_csv(
    os.path.join(SPLIT_DIR, 'test.csv'), index=False)



counts = train_df['label'].value_counts()
total = len(train_df)
n_columns = len(CLASS_COLS)
weights = {c: total / (n_columns * counts[c]) for c in CLASS_COLS} #higher weight if we make mistakes in the less common classes (otherwise too much weight to the begnin melanoma which are the most common)
print("\nClass weights:")
for c, w in weights.items():
    print(f"  {c:5s}: {w:.3f}")

weight_path = os.path.join(SPLIT_DIR, 'class_weights.json')
with open(weight_path, 'w') as f:
    json.dump(weights, f, indent=2)
print(f"Weights saved to {weight_path}")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# we apply random flip and and augmentation to make the model robust and generalized to lots of features


class ISICDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        label = int(row['label_idx'])
        if self.transform:
            image = self.transform(image)
        return image, label


print("\nRunning DataLoader check")
sample_ds = ISICDataset(train_df.head(16), transform=train_transform)
sample_dl = DataLoader(sample_ds, batch_size=4, num_workers=2)
imgs, labels = next(iter(sample_dl))
print(f"Batch image shape : {imgs.shape}") #4 images, 3 color channels rgb, 224×224 pixels
print(f"Batch label shape : {labels.shape}")
print(f"Label values      : {labels.tolist()}")
print("\nPreprocessing ok")
