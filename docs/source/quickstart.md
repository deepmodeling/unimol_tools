# Quick start

Quick start for UniMol Tools.

## Molecule property prediction

To train a model, you need to provide training data containing molecules represented as SMILES strings and corresponding target values. Targets can be real numbers for regression or binary values (0s and 1s) for classification. Leave target values blank for instances where they are unknown.

The model can be trained either on a single target ("single tasking") or on multiple targets simultaneously ("multi-tasking").

The data file can be a **CSV file with a header row**. The CSV format should have `SMILES` as input, followed by `TARGET` as the label. Note that the label is named with the `TARGET` prefix when the task involves multilabel (regression/classification). For example:

| SMILES                                          | TARGET |
| ----------------------------------------------- | ------ |
| NCCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N2CCCC2C(=O)O  | 0      |
| COc1cc(CN2CCCCC2)cc2cc(C(=O)O)c(=O)oc12         | 1      |
| CCN(CC)C(C)CN1c2ccccc2Sc3c1cccc3                | 1      |
|...                                              | ...    |

custom dict can also as the input. The dict format should be like 

```python
{'atoms':[['C','C'],['C','H','O']], 'coordinates':[coordinates_1,coordinates_2]}
```
Here is an example to train a model and make a prediction. When using Unimol V2, set `model_name='unimolv2'`.
```python
from unimol_tools import MolTrain, MolPredict
clf = MolTrain(task='classification', 
                data_type='molecule', 
                epochs=10, 
                batch_size=16, 
                metrics='auc',
                model_name='unimolv1', # avaliable: unimolv1, unimolv2
                model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                )
pred = clf.fit(data = train_data)
# currently support data with smiles based csv/txt file

clf = MolPredict(load_model='../exp')
res = clf.predict(data = test_data)
```

### Command-line utilities

Training, prediction, and representation can also be launched from the
command line by overriding options in the YAML config files.

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

## Uni-Mol molecule and atoms level representation

Uni-Mol representation can easily be achieved as follow.

```python
import numpy as np
from unimol_tools import UniMolRepr
# single smiles unimol representation
clf = UniMolRepr(data_type='molecule', # avaliable: molecule, oled, pocket. Only for unimolv1.
                 remove_hs=False,
                 model_name='unimolv1', # avaliable: unimolv1, unimolv2
                 model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                 )
smiles = 'c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]'
smiles_list = [smiles]
unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
# CLS token repr
print(np.array(unimol_repr['cls_repr']).shape)
# atomic level repr, align with rdkit mol.GetAtoms()
print(np.array(unimol_repr['atomic_reprs']).shape)

# For the pocket, please select and extract the atoms nearby; the total number of atoms should preferably not exceed 256.
clf = UniMolRepr(data_type='pocket',
                 remove_hs=False,
                 )
pocket_dict = {
    'atoms': atoms,
    'coordinates': coordinates,
    'residue': residue, # Optional
}
unimol_repr = clf.get_repr(pocket_dict, return_atomic_reprs=True)

```
## Molecule pretraining

Uni-Mol can be pretrained from scratch using the ``run_pretrain`` utility. The
script is driven by Hydra, so configuration options are supplied on the command
line. The examples below demonstrate common setups for LMDB and CSV inputs.

### LMDB dataset

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS \
    -m unimol_tools.cli.run_pretrain \
    dataset.train_path=train.lmdb \
    dataset.valid_path=valid.lmdb \
    dataset.data_type=lmdb \
    dataset.dict_path=dict.txt \
    training.total_steps=10000 \
    training.batch_size=16 \
    training.update_freq=1
```

`dataset.dict_path` is optional. The effective batch size is
`n_gpu * training.batch_size * training.update_freq`.

### CSV dataset

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS \
    -m unimol_tools.cli.run_pretrain \
    dataset.train_path=train.csv \
    dataset.valid_path=valid.csv \
    dataset.data_type=csv \
    dataset.smiles_column=smiles \
    training.total_steps=10000 \
    training.batch_size=16 \
    training.update_freq=1
```

To scale across multiple machines, include the appropriate `torchrun`
arguments, e.g. `--nnodes`, `--node_rank`, `--master_addr` and
`--master_port`.

Checkpoints and the dictionary are written to the output directory. When GPU
memory is limited, increase `training.update_freq` to accumulate gradients while
keeping the effective batch size `n_gpu * training.batch_size * training.update_freq`.

## Continue training (Re-train)

```python
clf = MolTrain(
    task='regression',
    data_type='molecule',
    epochs=10,
    batch_size=16,
    save_path='./model_dir',
    remove_hs=False,
    target_cols='TARGET',
    )
pred = clf.fit(data = train_data)
# After train a model, set load_model_dir='./model_dir' to continue training

clf2 = MolTrain(
    task='regression',
    data_type='molecule',
    epochs=10,
    batch_size=16,
    save_path='./retrain_model_dir',
    remove_hs=False,
    target_cols='TARGET',
    load_model_dir='./model_dir',
    )

pred2 = clf.fit(data = retrain_data)                
```

## Distributed Data Parallel (DDP) Training

Uni-Mol Tools now supports Distributed Data Parallel (DDP) training using PyTorch. DDP allows you to train models across multiple GPUs or nodes, significantly speeding up the training process.

### Parameters
- `use_ddp`: bool, default=True, whether to enable Distributed Data Parallel (DDP).
- `use_gpu`: str, default='all', specifies which GPUs to use. `'all'` means all available GPUs, while `'0,1,2'` means using GPUs 0, 1, and 2.

### Example Usage
To enable DDP, ensure your environment supports distributed training (e.g., PyTorch with distributed support). Set `use_ddp=True` and specify the GPUs using the `use_gpu` parameter when initializing the `MolTrain` class.

#### Example for Training

```python
from unimol_tools import MolTrain

# Initialize the training class with DDP enabled
if __name__ == '__main__':
    clf = MolTrain(
        task='regression',
        data_type='molecule',
        epochs=10,
        batch_size=16,
        save_path='./model_dir',
        remove_hs=False,
        target_cols='TARGET',
        use_ddp=True,
        use_gpu="all"
        )
    pred = clf.fit(data = train_data)
```

#### Example for Molecular Representation

```python
from unimol_tools import UniMolRepr

# Initialize the UniMolRepr class with DDP enabled
if __name__ == '__main__':
    repr_model = UniMolRepr(
        data_type='molecule',
        batch_size=32,
        remove_hs=False,
        model_name='unimolv2',
        model_size='84m',
        use_ddp=True,  # Enable Distributed Data Parallel
        use_gpu='0,1'  # Use GPU 0 and 1
    )

    unimol_repr = repr_model.get_repr(smiles_list, return_atomic_reprs=True)

    # CLS token representation
    print(unimol_repr['cls_repr'])
    # Atomic-level representation
    print(unimol_repr['atomic_reprs'])
```

- **Important:** When the number of SMILES strings is small, it is not recommended to use DDP for the `get_repr` method. Communication overhead between processes may outweigh the benefits of parallel computation, leading to slower performance. In such cases, consider disabling DDP by setting `use_ddp=False`.

### Why use `if __name__ == '__main__':`

In Python, when using multiprocessing (e.g., PyTorch's `DistributedDataParallel` or other libraries requiring multiple processes), it is recommended to use the `if __name__ == '__main__':` idiom. This is because, in a multiprocessing environment, child processes may re-import the main module. Without this idiom, the code in the main module could be executed multiple times, leading to unexpected behavior or errors.

#### Common Error

If you do not use `if __name__ == '__main__':`, you might encounter the following error:

```Python
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
```

To avoid this error, ensure that all code requiring multiprocessing is enclosed within the if `__name__ == '__main__'`: block.

### Notes
- For multi-node training, the `MASTER_ADDR` and `MASTER_PORT` environment variables can be configured as below.

```bash
export MASTER_ADDR='localhost'
export MASTER_PORT='19198'
```

## Simple examples of five tasks
Currently unimol_tools supports five types of fine-tuning tasks: `classification`, `regression`, `multiclass`, `multilabel_classification`, `multilabel_regression`.

The datasets used in the examples are all open source and available, including
- Ames mutagenicity. The dataset includes 6512 compounds and corresponding binary labels from Ames Mutagenicity results. The dataset is available at https://weilab.math.msu.edu/DataLibrary/2D/.
- ESOL (delaney) is a standard regression dataset containing structures and water solubility data for 1128 compounds. The dataset is available at https://weilab.math.msu.edu/DataLibrary/2D/ and https://huggingface.co/datasets/HR-machine/ESol.
- Tox21 Data Challenge 2014 is designed to help scientists understand the potential of the chemicals and compounds being tested through the Toxicology in the 21st Century initiative to disrupt biological pathways in ways that may result in toxic effects, which includes 12 date sets. The official web site is https://tripod.nih.gov/tox21/challenge/. The datasets is available at https://moleculenet.org/datasets-1 and https://www.kaggle.com/datasets/maksiamiogan/tox21-dataset.
- Solvation free energy (FreeSolv). SMILES are provided. The dataset is available at https://weilab.math.msu.edu/DataLibrary/2D/.
- Vector-QM24 (VQM24) dataset. Quantum chemistry dataset of ~836 thousand small organic and inorganic molecules. The dataset is available at https://zenodo.org/records/15442257.

### Example of classification
You can use a dictionary as input. The default smiles column name is **'SMILES'** and the target column name is **'target'**. You can also customize it with `smiles_col` and `target_cols`.

```Python
import pandas as pd
from unimol_tools import MolTrain, MolPredict

# Load the dataset
df = pd.read_csv('../datasets/Ames/Ames.csv')

df = df.drop(columns=['CAS_NO']).rename(columns={'Activity': 'target'})

# Divide the training set and test set
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_df_dict = train_df.to_dict(orient='list')
test_df_dict = test_df.to_dict(orient='list')

clf = MolTrain(task='classification', 
                data_type='molecule', 
                epochs=20, 
                batch_size=16, 
                metrics='auc',
                smiles_col='Canonical_Smiles',
                )

clf.fit(train_df_dict)

predictor = MolPredict(load_model='./exp')

pred = predictor.predict(test_df_dict['Canonical_Smiles'])
```

### Example of regression
You can directly use the csv file path as input. The default recognized smiles column name is **'SMILES'** and the target column name is **'TARGET'**. The column names can be customized by `smiles_col` and `target_cols`.

```python
from unimol_tools import MolTrain, MolPredict

# Load the dataset
train_data_path = '../datasets/ESol/train_data.csv'
test_data_path = '../datasets/ESol/test_data.csv'

reg = MolTrain(task='regression',
                data_type='molecule',
                epochs=20,
                batch_size=32,
                metrics='mae',
                smiles_col='smiles',
                target_cols=['ESOL predicted log solubility in mols per litre'],
                save_path='./exp_esol',
                )

reg.fit(train_data_path)

predictor = MolPredict(load_model='./exp_esol')
y_pred = predictor.predict(data=test_data_path)
```

It is also possible to use a list of atoms and a list of coordinates directly as input, with the column names **'atoms'** and **'coordinates'**. The smiles list is optional, but is required if scaffold is used as the grouping method. Atoms list supports either atom type or atomic number input, for example, 'atoms':[['C', 'C'],['C', 'H', 'O']] or 'atoms':[[6, 6],[6, 1, 8]].

```python
import numpy as np
from unimol_tools import MolTrain, MolPredict
from rdkit import Chem

# Load the dataset
data = np.load('../datasets/DMC.npz', allow_pickle=True)
atoms_all = data['atoms']
coordinates_all = data['coordinates']
smiles_all = data['graphs']
all_targets = data['Etot']

# Filter illegal smiles data
valid_smiles = []
valid_atoms = []
valid_coordinates = []
valid_targets = []
for smiles, target, atoms, coordinates in zip(smiles_all, all_targets, atoms_all, coordinates_all):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
            valid_atoms.append(atoms)
            valid_coordinates.append(coordinates)
            valid_targets.append(target)
    except Exception as e:
        print(f"Invalid SMILES: {smiles}, Error: {e}")

# Divide the training set and test set
num_molecules = len(valid_smiles)
np.random.seed(42)
indices = np.random.permutation(num_molecules)
train_end = int(0.8 * num_molecules)
train_val_idx = indices[:train_end]
test_idx  = indices[train_end:]

train_val_smiles  = [valid_smiles[i] for i in train_val_idx]
test_smiles = [valid_smiles[i] for i in test_idx]
train_val_atoms = [valid_atoms[i] for i in train_val_idx]
test_atoms = [valid_atoms[i] for i in test_idx]
train_val_coordinates = [valid_coordinates[i] for i in train_val_idx]
test_coordinates = [valid_coordinates[i] for i in test_idx]

train_val_targets = [valid_targets[i] for i in train_val_idx]
test_targets = [valid_targets[i] for i in test_idx]

train_val_data = {
    'target': train_val_targets,
    'atoms': train_val_atoms,
    'coordinates': train_val_coordinates,
    'SMILES':  train_val_smiles,
}
test_data = {
    'SMILES': test_smiles,
    'atoms': test_atoms,
    'coordinates': test_coordinates,
}

reg = MolTrain(task='regression',
                data_type='molecule',
                epochs=20,
                batch_size=32,
                metrics='mae',
                save_path='./exp',
                )

reg.fit(train_val_data)

predictor = MolPredict(load_model='./exp')
y_pred = predictor.predict(data=test_data, save_path='./pre') # Specify save_path to store prediction results
```

### Example of multiclass

```python
import pandas as pd
from unimol_tools import MolTrain, MolPredict
import numpy as np

# Load the dataset
df = pd.read_csv('../datasets/ESOL/ESOL.csv')

data_dict = {
    'SMILES': df['smiles'].tolist(),
    'target': df['Number of H-Bond Donors'].tolist()
}

data_dict['SMILES'] = [smiles for i, smiles in enumerate(data_dict['SMILES']) if data_dict['target'][i] <= 4]
data_dict['target'] = [target for target in data_dict['target'] if target <= 4]

# Divide the training set and test set
num_molecules = len(data_dict['SMILES'])
np.random.seed(42)
indices = np.random.permutation(num_molecules)
train_end = int(0.8 * num_molecules)
train_val_idx = indices[:train_end]
test_idx  = indices[train_end:]

train_val_dict = {
    'SMILES': [data_dict['SMILES'][i] for i in train_val_idx],
    'target': [data_dict['target'][i] for i in train_val_idx],
}
test_dict = {
    'SMILES': [data_dict['SMILES'][i] for i in test_idx],
    'target': [data_dict['target'][i] for i in test_idx],
}

mclf = MolTrain(task='multiclass',
                data_type='molecule',
                epochs=20,
                batch_size=32,
                metrics='acc',
                save_path='./exp',
                )

mclf.fit(train_val_dict)

predictor = MolPredict(load_model='./exp')
y_pred = predictor.predict(data=test_dict)
```

### Example of multilabel_classification

```python
import pandas as pd
from unimol_tools import MolTrain, MolPredict

# Load the dataset
df = pd.read_csv('../datasets/tox21.csv')

# Fill missing values ​​with 0
df.fillna(0, inplace=True)

df.drop(columns=['mol_id'], inplace=True)

# Divide the training set and test set
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_df_dict = train_df.to_dict(orient='list')
test_df_dict = test_df.to_dict(orient='list')

mlclf = MolTrain(task='multilabel_classification',
                    data_type='molecule',
                    epochs=20,
                    batch_size=32,
                    metrics='auc',
                    smiles_col='smiles',
                    target_cols=[col for col in df.columns if col != 'smiles'],
                    )
mlclf.fit(train_df_dict)

predictor = MolPredict(load_model='./exp')
pred = predictor.predict(test_df_dict['smiles'])
```

It also supports directly using the sdf file path as input. The following example reads it in advance due to preprocessing missing values.

```python
from unimol_tools import MolTrain, MolPredict
from rdkit.Chem import PandasTools

# Load the dataset
data_path = '../datasets/tox21.sdf'

data = PandasTools.LoadSDF(data_path)

# Fill missing values ​​with 0
data['SR-HSE'] = data['SR-HSE'].fillna(0)
data['NR-AR'] = data['NR-AR'].fillna(0)

mlclf = MolTrain(task='multilabel_classification',
                    data_type='molecule',
                    epochs=20,
                    batch_size=32,
                    metrics='auc',
                    target_cols=['SR-HSE', 'NR-AR'],
                    save_path='./exp_sdf',
                    )
mlclf.fit(data)
```

### Example of multilabel_regression

```python
from unimol_tools import MolTrain, MolPredict

# Load the dataset
data_path = '../datasets/FreeSolv/SAMPL.csv'

mreg = MolTrain(task='multilabel_regression',
                data_type='molecule',
                epochs=20,
                batch_size=32,
                metrics='mae',
                smiles_col='smiles',
                target_cols='expt,calc',
                save_path='./exp_csv',
                )

mreg.fit(data_path)
```

```python
from unimol_tools import MolTrain, MolPredict
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('../datasets/FreeSolv/SAMPL.csv')

data_dict = {
    'SMILES': df['smiles'].tolist(),
    'target': [df['expt'].tolist(), df['calc'].tolist()]
}

# Divide the training set and test set
num_molecules = len(data_dict['SMILES'])
np.random.seed(42)
indices = np.random.permutation(num_molecules)
train_end = int(0.8 * num_molecules)
train_val_idx = indices[:train_end]
test_idx  = indices[train_end:]

train_val_dict = {
    'SMILES': [data_dict['SMILES'][i] for i in train_val_idx],
    'target': [data_dict['target'][0][i] for i in train_val_idx],
}
test_dict = {
    'SMILES': [data_dict['SMILES'][i] for i in test_idx],
    'target': [data_dict['target'][0][i] for i in test_idx],
}

mreg = MolTrain(task='multilabel_regression',
                data_type='molecule',
                epochs=20,
                batch_size=32,
                metrics='mae',
                save_path='./exp_dict',
                )

mreg.fit(train_val_dict)

predictor = MolPredict(load_model='./exp_dict')
y_pred = predictor.predict(data=test_dict)
```

```python
from unimol_tools import MolTrain, MolPredict
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('../datasets/FreeSolv/SAMPL.csv')

data_dict = {
    'SMILES': df['smiles'].tolist(),
    'expt': df['expt'].tolist(),
    'calc': df['calc'].tolist()
}

# Divide the training set and test set
num_molecules = len(data_dict['SMILES'])
np.random.seed(42)
indices = np.random.permutation(num_molecules)
train_end = int(0.8 * num_molecules)
train_val_idx = indices[:train_end]
test_idx  = indices[train_end:]

train_val_dict = {
    'SMILES': [data_dict['SMILES'][i] for i in train_val_idx],
    'expt': [data_dict['expt'][i] for i in train_val_idx],
    'calc': [data_dict['calc'][i] for i in train_val_idx],
}
test_dict = {
    'SMILES': [data_dict['SMILES'][i] for i in test_idx],
    'expt': [data_dict['expt'][i] for i in test_idx],
    'calc': [data_dict['calc'][i] for i in test_idx],
}

mreg = MolTrain(task='multilabel_regression',
                data_type='molecule',
                epochs=20,
                batch_size=32,
                metrics='mae',
                target_cols=['expt', 'calc'],
                save_path='./exp_dict',
                )

mreg.fit(train_val_dict)

predictor = MolPredict(load_model='./exp_dict')
y_pred = predictor.predict(data=test_dict, save_path='./pre_dict')
```
