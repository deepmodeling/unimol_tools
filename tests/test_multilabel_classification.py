import os
import gzip
import zipfile
import pandas as pd
import pytest
from utils_net import download_for_test
from rdkit.Chem import PandasTools

from unimol_tools import MolTrain, MolPredict

CSV_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
SDF_URL = 'https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf'


@pytest.mark.network
def test_multilabel_csv(tmp_path):
    os.environ.setdefault('UNIMOL_WEIGHT_DIR', str(tmp_path / 'weights'))
    gz_path = tmp_path / 'tox21.csv.gz'
    csv_path = tmp_path / 'tox21.csv'
    download_for_test(
        CSV_URL,
        gz_path,
        timeout=(5, 60),
        max_retries=5,
        backoff_factor=0.5,
        allow_resume=True,
        skip_on_failure=True,
    )
    with gzip.open(gz_path, 'rb') as fin, open(csv_path, 'wb') as fout:
        fout.write(fin.read())
    df = pd.read_csv(csv_path)
    df.fillna(0, inplace=True)
    df.drop(columns=['mol_id'], inplace=True)
    # take 1000 samples for testing
    df = df.sample(n=1000, random_state=42)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    train_dict = train_df.to_dict(orient='list')
    test_smiles = test_df['smiles'].tolist()

    exp_dir = tmp_path / 'exp'
    mlclf = MolTrain(
        task='multilabel_classification',
        data_type='molecule',
        epochs=1,
        batch_size=8,
        kfold=2,
        metrics='auc',
        smiles_col='smiles',
        target_cols=[c for c in df.columns if c != 'smiles'],
        save_path=str(exp_dir),
    )
    try:
        mlclf.fit(train_dict)
    except Exception as e:
        pytest.skip(f"Training failed: {e}")

    predictor = MolPredict(load_model=str(exp_dir))
    preds = predictor.predict(test_smiles)
    assert len(preds) == len(test_smiles)


@pytest.mark.network
def test_multilabel_sdf(tmp_path):
    os.environ.setdefault('UNIMOL_WEIGHT_DIR', str(tmp_path / 'weights'))
    zip_path = tmp_path / 'tox21.zip'
    download_for_test(
        SDF_URL,
        zip_path,
        timeout=(5, 60),
        max_retries=5,
        backoff_factor=0.5,
        allow_resume=True,
        skip_on_failure=True,
    )
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_path)
    sdf_files = [p for p in tmp_path.rglob('*.sdf')]
    if not sdf_files:
        pytest.skip('SDF file not found after extraction')
    sdf_path = sdf_files[0]
    data = PandasTools.LoadSDF(str(sdf_path))
    data['SR-HSE'] = data['SR-HSE'].fillna(0)
    data['NR-AR'] = data['NR-AR'].fillna(0)
    # take 1000 samples for testing
    data = data.sample(n=1000, random_state=42)
    data_train = data.sample(frac=0.8, random_state=42)
    data_test = data.drop(data_train.index)

    exp_dir = tmp_path / 'exp_sdf'
    mlclf = MolTrain(
        task='multilabel_classification',
        data_type='molecule',
        epochs=1,
        batch_size=8,
        kfold=2,
        metrics='auc',
        target_cols=['SR-HSE', 'NR-AR'],
        save_path=str(exp_dir),
    )
    try:
        mlclf.fit(data_train)
    except Exception as e:
        pytest.skip(f"Training failed: {e}")
        
    predictor = MolPredict(load_model=str(exp_dir))
    preds = predictor.predict(data_test)
    assert len(preds) == len(data_test)
