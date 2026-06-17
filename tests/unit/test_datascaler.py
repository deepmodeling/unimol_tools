import numpy as np
from sklearn.preprocessing import PowerTransformer

from unimol_tools.data.datascaler import TargetScaler


def test_target_scaler_roundtrip(tmp_path):
    y = np.arange(6, dtype=float).reshape(-1, 1)
    scaler = TargetScaler('standard', 'regression')
    scaler.fit(y, str(tmp_path))
    scaled = scaler.transform(y)
    restored = scaler.inverse_transform(scaled)
    assert np.allclose(restored, y)


def test_power_trans_scaler_choice():
    scaler = TargetScaler('power_trans', 'regression')
    pos_scaler = scaler.scaler_choose('power_trans', np.array([[1.0], [2.0]]))
    neg_scaler = scaler.scaler_choose('power_trans', np.array([[-1.0], [2.0]]))
    assert isinstance(pos_scaler, PowerTransformer)
    assert pos_scaler.method == 'box-cox'
    assert neg_scaler.method == 'yeo-johnson'
