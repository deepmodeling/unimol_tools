# Uni-Mol Tools

<img src = "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/article/16664/d50607556c5c4076bf3df363a7f1aedf/4feaf601-09b6-4bcb-85a0-70890c36c444.png" width = 40%>

[![GitHub release](https://img.shields.io/github/release/deepmodeling/unimol_tools.svg)](https://github.com/deepmodeling/unimol_tools/releases/)
[![PyPI version](https://img.shields.io/pypi/v/unimol-tools.svg)](https://pypi.org/project/unimol-tools/)
![Python versions](https://img.shields.io/pypi/pyversions/unimol-tools.svg)
[![License](https://img.shields.io/github/license/deepmodeling/unimol_tools.svg)](https://github.com/deepmodeling/unimol_tools/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/deepmodeling/unimol_tools.svg)](https://github.com/deepmodeling/unimol_tools/issues)
[![GitHub contributors](https://img.shields.io/github/contributors/deepmodeling/unimol_tools.svg)](https://github.com/deepmodeling/unimol_tools/graphs/contributors)
![Maintained](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)
[![Documentation Status](https://readthedocs.org/projects/unimol/badge/?version=latest)](https://unimol.readthedocs.io/en/latest/?badge=latest)

Unimol_tools is a easy-use wrappers for property prediction,representation and downstreams with Uni-Mol.

# Uni-Mol tools for various prediction and downstreams.

📖 Documentation: [unimol-tools.readthedocs.io](https://unimol-tools.readthedocs.io/en/latest/)

## Install
- pytorch is required, please install pytorch according to your environment. if you are using cuda, please install pytorch with cuda. More details can be found at https://pytorch.org/get-started/locally/

### Option 1: Installing from PyPi (Recommended, for stable version)

```bash
pip install unimol_tools --upgrade
```

We recommend installing ```huggingface_hub``` so that the required unimol models can be automatically downloaded at runtime! It can be install by

```bash
pip install huggingface_hub
```

`huggingface_hub` allows you to easily download and manage models from the Hugging Face Hub, which is key for using Uni-Mol models.

### Option 2: Installing from source (for latest version)

```python
## Clone repository
git clone https://github.com/deepmodeling/unimol_tools.git
cd unimol_tools

## Dependencies installation
pip install -r requirements.txt

## Install
python setup.py install
```

### Models in Huggingface

The UniMol pretrained models can be found at [dptech/Uni-Mol-Models](https://huggingface.co/dptech/Uni-Mol-Models/tree/main).

If ``pretrained_model_path`` or ``pretrained_dict_path`` are left as ``None`` the
toolkit will automatically download the corresponding files from this
Hugging Face repository at runtime.

If the download is slow, you can use a mirror, such as:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

By default `unimol_tools` first tries the official Hugging Face endpoint. If that fails and `HF_ENDPOINT` is not set, it automatically retries using `https://hf-mirror.com`. Set `HF_ENDPOINT` yourself if you want to explicitly choose a mirror or the official site.

### Modify the default directory for weights

Setting the `UNIMOL_WEIGHT_DIR` environment variable specifies the directory for pre-trained weights if the weights have been downloaded from another source.

```bash
export UNIMOL_WEIGHT_DIR=/path/to/your/weights/dir/
```

## News
- 2025-09-22: Lightweight pre-training tools are now available in Unimol_tools!
- 2025-05-26: Unimol_tools is now independent from the Uni-Mol repository!
- 2025-03-28: Unimol_tools now support Distributed Data Parallel (DDP)!
- 2024-11-22: Unimol V2 has been added to Unimol_tools!
- 2024-07-23: User experience improvements: Add `UNIMOL_WEIGHT_DIR`.
- 2024-06-25: unimol_tools has been publish to pypi! Huggingface has been used to manage the pretrain models.
- 2024-06-20: unimol_tools v0.1.0 released, we remove the dependency of Uni-Core. And we will publish to pypi soon.
- 2024-03-20: unimol_tools documents is available at https://unimol-tools.readthedocs.io/en/latest/

## Examples
### Molecule property prediction
```python
from unimol_tools import MolTrain, MolPredict
clf = MolTrain(
    task='classification',
    data_type='molecule',
    epochs=10,
    batch_size=16,
    metrics='auc',
    # pretrained weights are downloaded automatically when left as ``None``
    # pretrained_model_path='/path/to/checkpoint.ckpt',
    # pretrained_dict_path='/path/to/dict.txt',
)
clf.fit(data = train_data)
# currently support data with smiles based csv/txt file, and sdf file with mol,
# and custom dict of {'atoms':[['C','C'],['C','H','O']], 'coordinates':[coordinates_1,coordinates_2]}

# The dict format can refer to the following format, or be obtained from sdf, 
# which can also be directly input into the model.
train_sdf = PandasTools.LoadSDF('exp/unimol_conformers_train.sdf')
train_dict = {
    'atoms': [list(atom.GetSymbol() for atom in mol.GetAtoms()) for mol in train_sdf['ROMol']],
    # atoms[0]: ['C', 'C', 'O', 'C', 'O', 'C', ...]
    'coordinates': [mol.GetConformers()[0].GetPositions() for mol in train_sdf['ROMol']],
    # coordinates[0]: array([[ 6.6462, -1.8268,  1.9275],
    #                        [ 6.1552, -1.9367,  0.4873],
    #                        [ 5.1832, -0.8757,  0.3007],
    #                        [ 5.4651, -0.0272, -0.7266],
    #                        [ 4.8586, -0.0844, -1.7917],
    #                        [ 6.5362,  0.9767, -0.3742],
    #                        ...,])
    'TARGET': train_sdf['TARGET'].tolist()
    # TARGET: [0, 1, 0, 0, 1, 0, ...]
}
# clf.fit(data = train_sdf)
# clf.fit(data = train_dict)


clf = MolPredict(load_model='../exp')
res = clf.predict(data = test_data)
```
### Molecule representation
```python
import numpy as np
from unimol_tools import UniMolRepr
# single SMILES UniMol representation. If no paths are provided the
# pretrained model and dictionary are fetched from Hugging Face.
clf = UniMolRepr(
    data_type='molecule',
    remove_hs=False,
    # pretrained_model_path='/path/to/checkpoint.ckpt',
    # pretrained_dict_path='/path/to/dict.txt',
)
smiles = 'c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]'
smiles_list = [smiles]
unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
# CLS token repr
print(np.array(unimol_repr['cls_repr']).shape)
# atomic level repr, align with rdkit mol.GetAtoms()
print(np.array(unimol_repr['atomic_reprs']).shape)
```

### Command-line utilities

Hydra-powered entry points make training, prediction, and representation
available from the command line. Key-value pairs override options from the
YAML files in `unimol_tools/config`.

#### Training
```bash
python -m unimol_tools.cli.run_train \
    train_path=train.csv \
    task=regression \
    save_path=./exp \
    smiles_col=smiles \
    target_cols=[target1] \
    epochs=10 \
    learning_rate=1e-4 \
    batch_size=16 \
    kfold=5
```

#### Prediction
```bash
python -m unimol_tools.cli.run_predict load_model=./exp data_path=test.csv
```

#### Representation
```bash
python -m unimol_tools.cli.run_repr data_path=test.csv smiles_col=smiles
```

### Molecule pretraining

`unimol_tools` provides a command-line utility for pretraining Uni-Mol models on
your own dataset. The script uses
[Hydra](https://hydra.cc/) so configuration values can be overridden at the
command line. Two common invocation examples are shown below: one for LMDB data
and one for a CSV of SMILES strings.

#### LMDB dataset

```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1

torchrun --standalone --nproc_per_node=NUM_GPUS \
    -m unimol_tools.cli.run_pretrain \
    dataset.train_path=train.lmdb \
    dataset.valid_path=valid.lmdb \
    dataset.data_type=lmdb \
    dataset.dict_path=dict.txt \
    training.total_steps=1000000 \
    training.batch_size=16 \
    training.update_freq=1
```

`dataset.dict_path` is optional. The effective batch size is
`n_gpu * training.batch_size * training.update_freq`.

#### CSV dataset

```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1

torchrun --standalone --nproc_per_node=NUM_GPUS \
    -m unimol_tools.cli.run_pretrain \
    dataset.train_path=train.csv \
    dataset.valid_path=valid.csv \
    dataset.data_type=csv \
    dataset.smiles_column=smiles \
    training.total_steps=1000000 \
    training.batch_size=16 \
    training.update_freq=1
```

For multi-node training, specify additional arguments, for example:

```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1

torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<master-ip> --master_port=<port> \
    -m unimol_tools.cli.run_pretrain ...
```

All available options are defined in
[`pretrain_config.py`](unimol_tools/pretrain/pretrain_config.py), and checkpoints
along with the dictionary are saved to the run directory. When GPU memory is
limited, increase `training.update_freq` to accumulate gradients while keeping
the effective batch size `n_gpu * training.batch_size * training.update_freq`.

## Credits
We thanks all contributors from the community for their suggestions, bug reports and chemistry advices. Currently unimol-tools is maintained by Yaning Cui, Xiaohong Ji, Zhifeng Gao from DP Technology and AI for Science Insitution, Beijing.

Please kindly cite our papers if you use this tools.
```

@article{gao2023uni,
  title={Uni-qsar: an auto-ml tool for molecular property prediction},
  author={Gao, Zhifeng and Ji, Xiaohong and Zhao, Guojiang and Wang, Hongshuai and Zheng, Hang and Ke, Guolin and Zhang, Linfeng},
  journal={arXiv preprint arXiv:2304.12239},
  year={2023}
}
```

License
-------

This project is licensed under the terms of the MIT license. See LICENSE for additional details.
