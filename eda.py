import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

DATA_DIR = "/content/drive/MyDrive/Projet_Melanoma/data/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
CSV_PATH = "/content/drive/MyDrive/Projet_Melanoma/data/ISIC_2019_Training_GroundTruth.csv"
FIG_DIR    = "./reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

print("Data folder exists:", os.path.exists(DATA_DIR))
print("CSV exists        :", os.path.exists(CSV_PATH))
if os.path.exists(DATA_DIR):
  all_images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.jpg')]
  print(f"Total images found : {len(all_images)}")

df = pd.read_csv(CSV_PATH)
print("\nCSV shape:", df.shape)
print(df.head(3))

df = df[df['UNK'] != 1.0]

CLASS_COLS = ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC']
df['label'] = df[CLASS_COLS].idxmax(axis=1)
counts = df['label'].value_counts()
print("\nClass counts:\n", counts)

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(counts.index, counts.values,
              color=plt.cm.Set2.colors[:len(counts)], edgecolor='white')
ax.bar_label(bars, fmt='%d', padding=4, fontsize=9)
ax.set_title("Class distribution — ISIC 2019")
ax.set_xlabel("Diagnostic category")
ax.set_ylabel("Number of images")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/class_distribution.png", dpi=150)
plt.show()
print("Saved: class_distribution.png")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, cls in zip(axes.flatten(), CLASS_COLS):
    row = df[df['label'] == cls].sample(1).iloc[0]
    img_path = os.path.join(DATA_DIR, row['image'] + '.jpg')
    if os.path.exists(img_path):
        ax.imshow(plt.imread(img_path))
    ax.set_title(cls, fontsize=11)
    ax.axis('off')
plt.suptitle("One sample per class", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sample_images.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: sample_images.png")

sizes = []
for fname in all_images[:100]:
    try:
        w, h = Image.open(os.path.join(DATA_DIR, fname)).size
        sizes.append((w, h))
    except Exception:
        pass
sizes_df = pd.DataFrame(sizes, columns=['width','height'])
print("\nImage size stats:")
print(sizes_df.describe().round(0))


majority = counts.max()
print("\nImbalance ratios vs majority class (NV):")
for cls, n in counts.items():
    print(f"  {cls:5s}: {n:5d} images  →  1:{majority//n}")


all_images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.jpg')]
print(f"Total images found: {len(all_images)}")

print("\nEDA complete.")
