# GDELT-ComplexEvent
This is the code and data for the article **"Structured, Complex and Time-complete Temporal Event Forecasting"**.

The repo includes:  
1. An automated data construction pipeline for the Structured, Complex, and Time-complete Temporal Events (**SCTc-TE**).
2. Two large-scale complex event datasets, named **MidEast-TE** and **GDELT-TE**.
3. A proposed model **LoGo** that leverages both local and global contexts for the newly formulated task: **SCTc-TE forecasting**.

## Environment

The code is tested to be runnable under the environment with
python=3.9; pytorch=1.12; cuda=11.3.

To create a new environment, you could use the commands below:
```
conda create --name logo python=3.9
conda activate logo
conda install pytorch==1.12.0 -c pytorch
pip install pandas
pip install tensorboard
conda install tqdm
conda install -c dglteam dgl-cuda11.3
```

## Dataset
The input data files need to be unzipped first:
```
unzip data/MIDEAST_CE.zip -d data
unzip data/GDELT_CE.zip -d data
```

The data folder then contains:
- CAEMO: Information from CAMEO ontology, organized in python dictionaries.
- MIDEAST_CE: SCTc-TE extracted by the Vicuna model.
- GDELT_CE: SCTc-TE constructed from GDELT data.


## Dataset Construction Pipeline
To construct the MIDEAST_CE and GDELT_CE data from raw data, the pipeline is collected and described in ./dataset_construction.



## Data Preparation

Before training the model, you need to generate both the local and global graphs:

```
python generate_graphs_ce.py --dataset MIDEAST_CE
python generate_graphs_ce.py --dataset GDELT_CE
```

## Experiment

The searched training hyperparameter configuration is stored in config.yaml.

Hyperparameters can also be set in command, see detailed usage in train_logo_early.py:get_cmd().

`[dataset_name]` is `MIDEAST_CE` or `GDELT_CE`

### Train LoGo model:
```
python train_logo_early.py -d [dataset_name] --m LoGo_sep
```

### Train LoGo ablations:

- LoGo<sub>local</sub>:
```
python train_logo_late.py -d [dataset_name] --local_only
```

- LoGo<sub>global</sub>:
```
python train_logo_late.py -d [dataset_name] --global_only
```

- LoGo<sub>share</sub>:
```
python train_logo_early.py -d [dataset_name] --m LoGo_share
```

- LoGo<sub>late</sub>:
```
python train_logo_late.py -d [dataset_name] --m LoGo_sep
```

### View results:

Loss and runs:
```
tensorboard --logdir=./runs/
```
Results in ./logs folder.
