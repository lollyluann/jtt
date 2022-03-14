import torch, os
import torchvision
import wandb
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from utils import hinge_loss
from train import train
from data.confounder_utils import prepare_confounder_data
from data import dro_dataset
from tqdm import tqdm
from waterbirds_even_grads.simulate_grad import full_detach

os.putenv("CUDA_VISIBLE_DEVICES", "1")
device = torch.device("cuda")

pretrained = True
n_classes = 2

model = torchvision.models.resnet50(pretrained=pretrained)
d = model.fc.in_features

model.fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(d, 512)),
    ('drpout', nn.Dropout(0.5)),
    ('fc2', nn.Linear(512, n_classes))
    ]))
model = model.to(device)

use_wandb = False
if use_wandb:
    wandb.watch(model)

#logger.flush()

criterion = hinge_loss
epoch_offset = 0

optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
        momentum = 0.9,
        weight_decay = args.weight_decay)

train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"train.csv",
    train_data.n_groups,
    mode=mode)

val_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"val.csv",
    val_data.n_groups,
    mode=mode)

test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"test.csv",
    test_data.n_groups,
    mode=mode)

train(model,
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

def get_data(which="train", get_grads=False):
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
    elif which=="val":
        loader = val_loader
    else: loader = test_loader
    
    all_embed = []
    all_y = []
    all_g = []
    all_l = []
    all_last_fc = []
    all_penult_fc = []
    
    for idx, batch in enumerate(tqdm(loader)):
        batch = tuple(t.to(device) for t in batch)
        x = batch[0]
        y = batch[1]
        g = batch[2]
        l = batch[3]
        data_idx = batch[4]

        outputs = model(x)

        if idx == 0:
            all_embed = full_detach(outputs)
            all_y = full_detach(y)
            all_g = full_detach(g)
            all_l = full_detach(l)
        else:
            all_embed = np.concatenate([all_embed, full_detach(outputs)], axis=0)
            all_y = np.concatenate([all_y, full_detach(y)])
            all_g = np.concatenate([all_g, full_detach(g)])
            all_l = np.concatenate([all_l, full_detach(l)])

    print(all_embed.shape)
    print(all_y.shape)
    print(all_g.shape)
    np.save(which+"_data_resnet50.npy", all_embed)
    np.save(which+"_data_y_resnet50.npy", all_y)
    np.save(which+"_data_g_resnet50.npy", all_g)
    np.save(which+"_data_l_resnet50.npy", all_l)

    if get_grads:
        export_grads(all_embed, all_y, model, optimizer, which)

get_data("train", get_grads=True)
get_data("val", get_grads=True)
get_data("test", get_grads=True)
