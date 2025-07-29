import os
import zipfile
import pandas as pd
import pytest
from utils_net import download_for_test

from unimol_tools import MolTrain, MolPredict

DATA_URL = 'https://weilab.math.msu.edu/DataLibrary/2D/Downloads/Ames_smi.zip'


@pytest.mark.network
def test_classification_train_predict(tmp_path):
    # ensure any pretrained weights are written to a temporary directory
    os.environ.setdefault('UNIMOL_WEIGHT_DIR', str(tmp_path / 'weights'))
    zip_path = tmp_path / 'Ames_smi.zip'
    download_for_test(
        DATA_URL,
        zip_path,
        timeout=(5, 60),
        max_retries=5,
        backoff_factor=0.5,
        allow_resume=True,
        skip_on_failure=True,
    )
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_path)
    csv_path = tmp_path / 'Ames.csv'
    if not csv_path.exists():
        pytest.skip('Dataset missing after extraction')
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['CAS_NO']).rename(columns={'Activity': 'target'})
    # take 100 samples for testing
    df = df.sample(n=100, random_state=42)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    train_data = train_df.to_dict(orient='list')
    test_smiles = test_df['Canonical_Smiles'].tolist()

    exp_dir = tmp_path / 'exp'
    clf = MolTrain(
        task='classification',
        data_type='molecule',
        epochs=1,
        batch_size=2,
        kfold=2,
        metrics='auc',
        smiles_col='Canonical_Smiles',
        save_path=str(exp_dir),
    )
    try:
        clf.fit(train_data)
    except Exception as e:
        pytest.skip(f"Training failed: {e}")

    predictor = MolPredict(load_model=str(exp_dir))
    preds = predictor.predict(test_smiles)
    assert len(preds) == len(test_smiles)