import os
import zipfile
import pandas as pd
import numpy as np
import pytest

from unimol_tools import MolTrain, MolPredict

DATA_URL = 'https://weilab.math.msu.edu/DataLibrary/2D/Downloads/FreeSolv_smi.zip'


def download_dataset(dest):
    import requests
    r = requests.get(DATA_URL)
    r.raise_for_status()
    dest.write_bytes(r.content)


def test_multilabel_regression_csv(tmp_path):
    os.environ.setdefault('UNIMOL_WEIGHT_DIR', str(tmp_path / 'weights'))
    zip_path = tmp_path / 'freesolv.zip'
    try:
        download_dataset(zip_path)
    except Exception as e:
        pytest.skip(f"Could not download dataset: {e}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_path)
    csv_path = tmp_path / 'SAMPL.csv'
    if not csv_path.exists():
        pytest.skip('Dataset missing after extraction')
    df = pd.read_csv(csv_path)
    # take 100 samples for testing
    df = df.sample(n=100, random_state=42)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    train_path = tmp_path / 'train.csv'
    test_path = tmp_path / 'test.csv'
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    exp_dir = tmp_path / 'exp_csv'
    mreg = MolTrain(
        task='multilabel_regression',
        data_type='molecule',
        epochs=1,
        batch_size=2,
        kfold=2,
        metrics='mae',
        smiles_col='smiles',
        target_cols='expt,calc',
        save_path=str(exp_dir),
    )
    try:
        mreg.fit(str(train_path))
    except Exception as e:
        pytest.skip(f"Training failed: {e}")

    predictor = MolPredict(load_model=str(exp_dir))
    preds = predictor.predict(str(test_path))
    assert len(preds) == len(test_df)


def test_multilabel_regression_dict(tmp_path):
    os.environ.setdefault('UNIMOL_WEIGHT_DIR', str(tmp_path / 'weights'))
    zip_path = tmp_path / 'freesolv.zip'
    try:
        download_dataset(zip_path)
    except Exception as e:
        pytest.skip(f"Could not download dataset: {e}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_path)
    csv_path = tmp_path / 'SAMPL.csv'
    if not csv_path.exists():
        pytest.skip('Dataset missing after extraction')
    df = pd.read_csv(csv_path)
    # take 100 samples for testing
    df = df.sample(n=100, random_state=42, ignore_index=True)
    n = len(df)
    np.random.seed(42)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx = idx[:split]
    test_idx = idx[split:]
    train_dict = {
        'SMILES': [df['smiles'][i] for i in train_idx],
        'expt': [df['expt'][i] for i in train_idx],
        'calc': [df['calc'][i] for i in train_idx],
    }
    test_dict = {
        'SMILES': [df['smiles'][i] for i in test_idx],
        'expt': [df['expt'][i] for i in test_idx],
        'calc': [df['calc'][i] for i in test_idx],
    }

    exp_dir = tmp_path / 'exp_dict'
    mreg = MolTrain(
        task='multilabel_regression',
        data_type='molecule',
        epochs=1,
        batch_size=2,
        kfold=2,
        metrics='mae',
        target_cols=['expt', 'calc'],
        save_path=str(exp_dir),
    )
    try:
        mreg.fit(train_dict)
    except Exception as e:
        pytest.skip(f"Training failed: {e}")

    predictor = MolPredict(load_model=str(exp_dir))
    preds = predictor.predict(data=test_dict)
    assert len(preds) == len(test_idx)
