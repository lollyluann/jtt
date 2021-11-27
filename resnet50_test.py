import torch
import torchvision
import wandb
import torch.nn as nn
import numpy as np

from utils import hinge_loss
from train import train
from data.confounder_utils import prepare_confounder_data
from data import dro_dataset
from tqdm import tqdm

device = torch.device("cuda")

def full_detach(x):
    return x.squeeze().detach().cpu().numpy()

pretrained = True
#n_classes = 2

model = torchvision.models.resnet50(pretrained=pretrained)
#d = model.fc.in_features
#model.fc = nn.Linear(d, n_classes)
model = model.to(device)

use_wandb = False
if use_wandb:
    wandb.watch(model)

#logger.flush()

criterion = hinge_loss
epoch_offset = 0

'''train(model,
      criterion,
      data,
      logger,
      train_csv_logger,
      val_csv_logger,
      test_csv_logger,
      args,
      epoch_offset=epoch_offset,
      csv_name=fold,
      wandb=wandb if use_wandb else None)
'''

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
    
    all_embed = []
    all_y = []
    for idx, batch in enumerate(tqdm(train_loader)):
        batch = tuple(t.to(device) for t in batch)
        x = batch[0]
        y = batch[1]
        g = batch[2]
        data_idx = batch[3]

        outputs = model(x)

        if idx == 0:
            all_embed = full_detach(outputs)
            all_y = full_detach(y)
        else:
            all_embed = np.concatenate([all_embed, full_detach(outputs)], axis=0)
            all_y = np.concatenate([all_y, full_detach(y)])

    print(all_embed.shape)
    print(all_y.shape)
    np.save("train_data_resnet50.npy", all_embed)
    np.save("train_data_y_resnet50.npy", all_y)

get_data()
