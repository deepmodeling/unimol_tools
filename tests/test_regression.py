import os
import numpy as np
import pytest
from rdkit import Chem

from unimol_tools import MolTrain, MolPredict

ESOL_TRAIN_URL = 'https://huggingface.co/datasets/HR-machine/ESol/resolve/main/train_data.csv?download=true'
ESOL_TEST_URL = 'https://huggingface.co/datasets/HR-machine/ESol/resolve/main/test_data.csv?download=true'
VQM24_URL = 'https://zenodo.org/records/15442257/files/DMC.npz?download=1'

def download(url, dest):
    import requests
    r = requests.get(url)
    r.raise_for_status()
    dest.write_bytes(r.content)


def test_regression_esol(tmp_path):
    train_csv = tmp_path / 'train.csv'
    test_csv = tmp_path / 'test.csv'
    try:
        download(ESOL_TRAIN_URL, train_csv)
        download(ESOL_TEST_URL, test_csv)
    except Exception as e:
        pytest.skip(f"Could not download dataset: {e}")

    exp_dir = tmp_path / 'exp_esol'
    reg = MolTrain(
        task='regression',
        data_type='molecule',
        epochs=1,
        batch_size=2,
        kfold=2,
        metrics='mae',
        smiles_col='smiles',
        target_cols=['ESOL predicted log solubility in mols per litre'],
        save_path=str(exp_dir),
    )
    reg.fit(str(train_csv))
    predictor = MolPredict(load_model=str(exp_dir))
    preds = predictor.predict(str(test_csv))
    assert len(preds) > 0


def test_regression_vqm24(tmp_path):
    npz_path = tmp_path / 'dmc.npz'
    try:
        download(VQM24_URL, npz_path)
    except Exception as e:
        pytest.skip(f"Could not download dataset: {e}")
    data = np.load(npz_path, allow_pickle=True)
    atoms_all = data['atoms']
    coords_all = data['coordinates']
    smiles_all = data['graphs']
    targets_all = data['Etot']

    valid_smiles = []
    valid_atoms = []
    valid_coords = []
    valid_targets = []
    for smi, target, atoms, coords in zip(smiles_all, targets_all, atoms_all, coords_all):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                valid_atoms.append(atoms)
                valid_coords.append(coords)
                valid_targets.append(target)
        except Exception:
            pass
    # take 100 samples for testing
    valid_smiles = valid_smiles[:100]
    valid_atoms = valid_atoms[:100]
    valid_coords = valid_coords[:100]
    valid_targets = valid_targets[:100]

    n = len(valid_smiles)
    np.random.seed(42)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx = idx[:split]
    test_idx = idx[split:]

    train_data = {
        'target': [valid_targets[i] for i in train_idx],
        'atoms': [valid_atoms[i] for i in train_idx],
        'coordinates': [valid_coords[i] for i in train_idx],
        'SMILES': [valid_smiles[i] for i in train_idx],
    }
    test_data = {
        'atoms': [valid_atoms[i] for i in test_idx],
        'coordinates': [valid_coords[i] for i in test_idx],
        'SMILES': [valid_smiles[i] for i in test_idx],
    }

    exp_dir = tmp_path / 'exp_vqm'
    reg = MolTrain(
        task='regression',
        data_type='molecule',
        epochs=1,
        batch_size=2,
        kfold=2,
        metrics='mae',
        save_path=str(exp_dir),
    )
    reg.fit(train_data)
    predictor = MolPredict(load_model=str(exp_dir))
    preds = predictor.predict(data=test_data, save_path=str(tmp_path/'pred'))
    assert len(preds) == len(test_idx)
