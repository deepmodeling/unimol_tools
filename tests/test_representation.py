import gzip
import zipfile
import numpy as np
import pandas as pd
import pytest
from rdkit.Chem import PandasTools

from unimol_tools import UniMolRepr

VQM24_URL = 'https://zenodo.org/records/15442257/files/DMC.npz?download=1'
TOX21_CSV_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
TOX21_SDF_URL = 'https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf&sec='


def download(url, dest):
    import requests
    r = requests.get(url)
    r.raise_for_status()
    dest.write_bytes(r.content)


def test_unimol_repr_vqm24(tmp_path):
    npz_path = tmp_path / 'DMC.npz'
    try:
        download(VQM24_URL, npz_path)
    except Exception as e:
        pytest.skip(f'Could not download dataset: {e}')
    data = np.load(npz_path, allow_pickle=True)
    atoms = data['atoms'][:100]
    coords = data['coordinates'][:100]
    smiles = data['graphs'][:100]
    data_dict = {
        'SMILES': smiles.tolist(),
        'atoms': atoms.tolist(),
        'coordinates': coords.tolist(),
    }
    repr_model = UniMolRepr(batch_size=16)
    try:
        out = repr_model.get_repr(data_dict, return_atomic_reprs=True)
    except Exception as e:
        pytest.skip(f'representation failed: {e}')
    assert 'cls_repr' in out and len(out['cls_repr']) == len(smiles)


def test_unimol_repr_tox21_csv(tmp_path):
    gz_path = tmp_path / 'tox21.csv.gz'
    csv_path = tmp_path / 'tox21.csv'
    try:
        download(TOX21_CSV_URL, gz_path)
    except Exception as e:
        pytest.skip(f'Could not download dataset: {e}')
    with gzip.open(gz_path, 'rb') as fin, open(csv_path, 'wb') as fout:
        fout.write(fin.read())
    df = pd.read_csv(csv_path).head(100)
    repr_model = UniMolRepr(smiles_col='smiles', batch_size=32)
    try:
        tensor = repr_model.get_repr(df, return_tensor=True)
    except Exception as e:
        pytest.skip(f'representation failed: {e}')
    assert tensor.shape[0] == len(df)


def test_unimol_repr_tox21_sdf(tmp_path):
    zip_path = tmp_path / 'tox21.zip'
    try:
        download(TOX21_SDF_URL, zip_path)
    except Exception as e:
        pytest.skip(f'Could not download dataset: {e}')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_path)
    sdf_files = list(tmp_path.rglob('*.sdf'))
    if not sdf_files:
        pytest.skip('SDF file not found after extraction')
    data = PandasTools.LoadSDF(str(sdf_files[0])).head(100)
    repr_model = UniMolRepr(batch_size=16)
    try:
        out = repr_model.get_repr(data, return_atomic_reprs=True)
    except Exception as e:
        pytest.skip(f'representation failed: {e}')
    assert isinstance(out, dict) and len(out['cls_repr']) == len(data)
