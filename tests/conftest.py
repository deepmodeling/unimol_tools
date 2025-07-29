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


def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true", help="run tests that need network")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_marker = pytest.mark.skip(reason="need --run-network to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_marker)
