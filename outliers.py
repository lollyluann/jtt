import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset
import random, csv

def get_outlier_transforms(blur=True, colorjitter=True, posterize=True):
        outlier_transforms = []
        if posterize: outlier_transforms.append(transforms.RandomPosterize(bits=4, p=0.4))
        if colorjitter: outlier_transforms.append(transforms.ColorJitter())
        if blur: outlier_transforms.append(transforms.GaussianBlur(kernel_size=11))
        return outlier_transforms

def corrupt_labels(y, proportion):
    mask = np.zeros(y.size, dtype=int)
    mask[:int(proportion*y.size)] = 1
    np.random.shuffle(mask)
    return np.absolute(np.subtract(mask, y)), mask

data_dir = "cub/data/waterbird_complete95_forest2water2"
out_dir = "cub/data/waterbird_outliers"
outliers_file = "cub/data/outliers_annotations.csv"

transf = get_outlier_transforms()
transf.extend([None for i in range(100-len(transf))])
outliers = [["img_path", "transform"]]

metadata_df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
print(metadata_df["img_filename"])
original_metadata = metadata_df["y"].values
new_metadata, mask = corrupt_labels(original_metadata, proportion=0.02)
changed = np.ma.masked_array(metadata_df["img_filename"].to_numpy(), 1-mask).compressed()
metadata_df["y"] = new_metadata
metadata_df.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
outliers.extend([[a, 3] for a in changed.tolist()])

for subdir in os.listdir(data_dir):
    if subdir.split(".")[-1] == "csv": continue
    sub_path = os.path.join(out_dir, subdir)
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)

    for img in os.listdir(os.path.join(data_dir, subdir)):
        input_path = os.path.join(data_dir, subdir, img)
        img_file = torchvision.io.read_image(input_path)
        t_n = random.randint(0, len(transf)-1)
        t = transf[t_n]

        if t:
            print("Applying transform", t, "to file", img)
            new_img = t(img_file)
            outliers.append([subdir + "/" + img, t_n])
        else: new_img = img_file
        outpath = os.path.join(out_dir, subdir, img)
        torchvision.io.write_jpeg(new_img, outpath)

print("Corrupted labels for", mask.sum(), "images")
with open(outliers_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(outliers)
