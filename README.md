# Gradient Subgroup Scanning for Distributionally and Outlier Robust Models



## Environment

Create an environment with the following commands:
```
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
```

## Downloading Datasets

- **Waterbirds:** Download waterbirds from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz) and put it in `jtt/cub`.
    - In that directory, our code expects `data/waterbird_complete95_forest2water2/` with `metadata.csv` inside.

## **Running our Method**

- Train the initial ERM model:
    - `python generate_downstream.py --exp_name $EXPERIMENT_NAME --dataset $DATASET --method ERM`
        - Some useful optional args: `--n_epochs $EPOCHS --lr $LR --weight_decay $WD`. Other args, e.g. batch size, can be changed in generate_downstream.py.
        - Datasets: `CUB`, `CelebA`, `MultiNLI`, `jigsaw`
    - Bash execute the generated script for ERM inside `results/dataset/$EXPERIMENT_NAME`
- Once ERM is done training, run `python process_training.py --exp_name $EXPERIMENT_NAME --dataset $DATASET --folder_name $ERM_FOLDER_NAME --lr $LR --weight_decay $WD --deploy`
- Bash execute the generated scripts that have `JTT` in their name.

## Monitoring Performance

- Run `python analysis.py --exp_name $PATH_TO_JTT_RUNS --dataset $DATASET`
    - The `$PATH_TO_JTT_RUNS` will look like `$EXPERIMENT_NAME+"/train_downstream_"+$ERM_FOLDER_NAME+"/final_epoch"+$FINAL_EPOCH`
- You can also track accuracies in train.csv, val.csv, and test.csv in the JTT directory or use wandb to monitor performance for all experiments (although this does not include subgroups of CivilComments-WILDS)

## Running ERM, Joint DRO, or Group DRO
- Run `python generate_downstream.py --exp_name $EXPERIMENT_NAME --dataset $DATASET --method $METHOD`
        - Some useful optional args: `--n_epochs $EPOCHS --lr $LR --weight_decay $WD`
        - Datasets: `CUB`, `CelebA`, `MultiNLI`, `jigsaw`
- Bash execute the generated script for the method inside `results/dataset/$EXPERIMENT_NAME`

## **Adding other datasets**

Add the following:

- A dataset file (similar to cub_dataset.py)
- Edit `process_training.py` to include the required args for your dataset and implement a way for getting the spurious features from the dataset.

## Sample Commands for running JTT on Waterbirds

```
python generate_downstream.py --exp_name CUB_sample_exp --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM

bash results/CUB/CUB_sample_exp/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

python process_training.py --exp_name CUB_sample_exp --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch 60 --deploy

bash results/CUB/CUB_sample_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch50/JTT_upweight_100_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

python analysis.py --exp_name CUB_sample_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch60/ --dataset CUB
```
