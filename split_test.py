import pandas as pd
import os
import numpy as np

data_dir = "cub/data/waterbird_complete95_forest2water2"
out_dir = "cub/data/waterbird_outliers"
outliers_file = "cub/data/outliers_annotations.csv"

metadata_df = pd.read_csv(os.path.join(out_dir, "metadata.csv"))

print(metadata_df.split.unique())
print(metadata_df.split.value_counts())
print(metadata_df.group_labels.value_counts())

rows = metadata_df[(metadata_df['split']==2) & (metadata_df['group_labels']==4)]
print(rows)

