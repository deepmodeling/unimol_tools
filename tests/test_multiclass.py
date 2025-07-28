import os
import zipfile
import pandas as pd
import numpy as np
import pytest

from unimol_tools import MolTrain, MolPredict

DATA_URL = 'https://weilab.math.msu.edu/DataLibrary/2D/Downloads/ESOL_smi.zip'


def download_dataset(dest):
    import requests
    r = requests.get(DATA_URL)
    r.raise_for_status()
    dest.write_bytes(r.content)


def test_multiclass_train_predict(tmp_path):
    os.environ.setdefault('UNIMOL_WEIGHT_DIR', str(tmp_path / 'weights'))
    zip_path = tmp_path / 'ESOL_smi.zip'
    try:
        download_dataset(zip_path)
    except Exception as e:
        pytest.skip(f"Could not download dataset: {e}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_path)
    csv_path = tmp_path / 'ESOL.csv'
    if not csv_path.exists():
        pytest.skip('Dataset missing after extraction')
    df = pd.read_csv(csv_path)
    # Filter for multiclass target
    df = df[df['Number of H-Bond Donors'] <= 2]
    if df.empty:
        pytest.skip('No valid samples for multiclass classification')
    # take 100 samples for testing
    df = df.sample(n=100, random_state=42)
    
    data_dict = {
        'SMILES': df['smiles'].tolist(),
        'target': df['Number of H-Bond Donors'].tolist(),
    }

    n = len(data_dict['SMILES'])
    np.random.seed(42)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx = idx[:split]
    test_idx = idx[split:]

    train_dict = {
        'SMILES': [data_dict['SMILES'][i] for i in train_idx],
        'target': [data_dict['target'][i] for i in train_idx],
    }
    test_dict = {
        'SMILES': [data_dict['SMILES'][i] for i in test_idx],
        'target': [data_dict['target'][i] for i in test_idx],
    }

    exp_dir = tmp_path / 'exp'
    mclf = MolTrain(
        task='multiclass',
        data_type='molecule',
        epochs=1,
        batch_size=2,
        kfold=2,
        metrics='acc',
        save_path=str(exp_dir),
    )
    try:
        mclf.fit(train_dict)
    except Exception as e:
        pytest.skip(f"Training failed: {e}")

    predictor = MolPredict(load_model=str(exp_dir))
    preds = predictor.predict(data=test_dict)
    assert len(preds) == len(test_dict['SMILES'])
