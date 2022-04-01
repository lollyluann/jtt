import os, csv
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision

from data.confounder_utils import prepare_confounder_data
from data import dro_dataset

from extractable_resnet import resnet18

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
        self.override_groups_file=False
        self.batch_size = 64
    
def get_data(which="train"):
    loader_kwargs = {"batch_size": 64,
                     "num_workers": 4,
                     "pin_memory": True}
    args = DefaultArgs()

    train_data, val_data, test_data = prepare_confounder_data(args, train=True, return_full_dataset=False)

    train_loader = dro_dataset.get_loader(train_data,
                                          train=True,
                                          reweight_groups=None,
                                          **loader_kwargs)

    val_loader = dro_dataset.get_loader(val_data,
                                        train=False,
                                        reweight_groups=None,
                                        **loader_kwargs)
    
    test_loader = dro_dataset.get_loader(test_data,
                                         train=False,
                                         reweight_groups=None,
                                         **loader_kwargs)

    if which=="train":
        loader = train_loader
        data_set = train_data
    elif which=="val":
        loader = val_loader
        data_set = val_data
    else:
        loader = test_loader
        data_set = test_data

    # Initialize model
    model = resnet18(
        pretrained=True,
        layers_to_extract=1)
    #model = torchvision.models.resnet18(pretrained=True)
    #model = torch.nn.Sequential(*list(model.children())[:-1])

    model.eval()
    model = model.cuda()

    n = len(data_set)
    idx_check = np.empty(n)
    last_batch = False
    start_pos = 0

    with torch.set_grad_enabled(False):

        for i, (x_batch, y, g, l, j) in enumerate(tqdm(loader)):
            x_batch = x_batch.cuda()

            num_in_batch = list(x_batch.shape)[0]
            assert num_in_batch <= args.batch_size
            if num_in_batch < args.batch_size:
                assert last_batch == False
                last_batch = True

            end_pos = start_pos + num_in_batch

            features_batch = model(x_batch).data.cpu().numpy().squeeze()
            if i == 0:
                d = features_batch.shape[1]
                print(f'Extracting {d} features per example')
                features = np.empty((n, d))
            features[start_pos:end_pos, :] = features_batch

            # idx_check[start_pos:end_pos] = idx_batch.data.numpy()
            start_pos = end_pos

    output_path = f'resnet-18_1layer{which}.npy'
    np.save(output_path, features)


def main():
    torch.manual_seed(0)

    get_data("train")
    get_data("val")
    get_data("test")

if __name__=='__main__':
    main()
