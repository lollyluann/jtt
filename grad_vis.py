import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import umap

from data import dro_dataset
from data.confounder_utils import prepare_confounder_data

class DefaultArgs:
    def __init__(self):
        self.root_dir = "./cub"
        self.dataset = "CUB"
        self.target_name = "waterbird_complete95"
        self.confounder_names = ["forest2water2"]
        self.model = "resnet50"
        self.metadata_csv_name = None
        self.augment_data = False
        self.fraction = 1.0

def get_data():
    loader_kwargs = {"batch_size": 64,
                     "num_workers": 4,
                     "pin_memory": True}

    args = DefaultArgs()

    _, _, test_data = prepare_confounder_data(args, train=True, return_full_dataset=False)
    
    test_loader = dro_dataset.get_loader(test_data,
            train=False,
            reweight_groups=None,
            **loader_kwargs)

    y_data = []
    for batch_idx, batch in enumerate(test_loader):
        y_data.append(batch[2])
    y_data = np.stack(y_data[:-1], axis=0)
    y_data = y_data.flatten()
    return y_data


directory = "results/CUB/extract_grads/ERM_upweight_0_epochs_100_lr_1e-05_weight_decay_1.0/model_outputs/grad_to_input_epoch_99.npy"
grads = np.load(directory)
print("Loaded gradients of shape", grads.shape)

grads = grads.reshape(grads.shape[0]*grads.shape[1], grads.shape[2]*grads.shape[3]*grads.shape[4])
print("Reshaped gradients to shape", grads.shape)

reducer = umap.UMAP()
readfromfile = True
if readfromfile:
    embedding = np.load("umap_embedding.npy")
else:
    embedding = reducer.fit_transform(grads)
    np.save("umap_embedding.npy", embedding)
print("UMAP embedding of gradients with shape", embedding.shape)

y_data = get_data()
print(y_data)
print("Y label data with shape", y_data.shape)

plt.scatter(embedding[:,0], embedding[:,1], c=y_data, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of test set gradients')
plt.savefig("figures/umap_plot.pdf")
