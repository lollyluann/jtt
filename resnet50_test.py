import torch
import torchvision
import wandb
import torch.nn as nn

from utils import hinge_loss
from train import train

pretrained = false
n_classes = 2

model = torchvision.models.resnet50(pretrained=pretrained)
d = model.fc.in_features
model.fc = nn.Linear(d, n_classes)

use_wandb = True
if use_wandb:
    wandb.watch(model)

logger.flush()

criterion = hinge_loss
epoch_offset = 0

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


