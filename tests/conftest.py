import os
import pytest

@pytest.fixture(scope="session", autouse=True)
def set_unimol_weight_dir(tmp_path_factory):
    """Ensure UNIMOL_WEIGHT_DIR is set to a temporary directory for tests."""
    weight_dir = tmp_path_factory.mktemp("weights")
    original = os.environ.get("UNIMOL_WEIGHT_DIR")
    os.environ["UNIMOL_WEIGHT_DIR"] = str(weight_dir)
    yield
    if original is None:
        os.environ.pop("UNIMOL_WEIGHT_DIR", None)
    else:
        os.environ["UNIMOL_WEIGHT_DIR"] = original
