import os

from ..utils import logger

try:
    from huggingface_hub import snapshot_download
except:
    huggingface_hub_installed = False

    def snapshot_download(*args, **kwargs):
        raise ImportError(
            'huggingface_hub is not installed. If weights are not avaliable, please install it by running: pip install huggingface_hub. Otherwise, please download the weights manually from https://huggingface.co/dptech/Uni-Mol-Models'
        )


DEFAULT_WEIGHT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_weight_dir():
    """Return the directory where weights should be stored."""
    return os.environ.get("UNIMOL_WEIGHT_DIR", DEFAULT_WEIGHT_DIR)

HF_MIRROR = "https://hf-mirror.com"

def _snapshot_download_with_fallback(**kwargs):
    """Try downloading with the current HF_ENDPOINT and fall back to the mirror.

    The mirror is only tried when the user has not explicitly set HF_ENDPOINT
    and the first attempt fails.
    """
    user_set = "HF_ENDPOINT" in os.environ
    try:
        return snapshot_download(**kwargs)
    except Exception as e:
        if user_set:
            raise
        logger.warning(
            f"Download failed from Hugging Face: {e}. Retrying with {HF_MIRROR}"
        )
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        return snapshot_download(**kwargs)


def log_weights_dir():
    """
    Logs the directory where the weights are stored.
    """
    weight_dir = get_weight_dir()
    
    if 'UNIMOL_WEIGHT_DIR' in os.environ:
        logger.warning(
            f'Using custom weight directory from UNIMOL_WEIGHT_DIR: {weight_dir}'
        )
    else:
        logger.info(f'Weights will be downloaded to default directory: {weight_dir}')


def weight_download(pretrain, save_path, local_dir_use_symlinks=True):
    """
    Downloads the specified pretrained model weights.

    :param pretrain: (str), The name of the pretrained model to download.
    :param save_path: (str), The directory where the weights should be saved.
    :param local_dir_use_symlinks: (bool, optional), Whether to use symlinks for the local directory. Defaults to True.
    """
    log_weights_dir()

    if os.path.exists(os.path.join(save_path, pretrain)):
        logger.info(f'{pretrain} exists in {save_path}')
        return

    logger.info(f'Downloading {pretrain}')
    _snapshot_download_with_fallback(
        repo_id="dptech/Uni-Mol-Models",
        local_dir=save_path,
        allow_patterns=pretrain,
        # local_dir_use_symlinks=local_dir_use_symlinks,
        # max_workers=8
    )


def weight_download_v2(pretrain, save_path, local_dir_use_symlinks=True):
    """
    Downloads the specified pretrained model weights.

    :param pretrain: (str), The name of the pretrained model to download.
    :param save_path: (str), The directory where the weights should be saved.
    :param local_dir_use_symlinks: (bool, optional), Whether to use symlinks for the local directory. Defaults to True.
    """
    log_weights_dir()

    if os.path.exists(os.path.join(save_path, pretrain)):
        logger.info(f'{pretrain} exists in {save_path}')
        return

    logger.info(f'Downloading {pretrain}')
    _snapshot_download_with_fallback(
        repo_id="dptech/Uni-Mol2",
        local_dir=save_path,
        allow_patterns=pretrain,
        # local_dir_use_symlinks=local_dir_use_symlinks,
        # max_workers=8
    )


# Download all the weights when this script is run
def download_all_weights(local_dir_use_symlinks=False):
    """
    Downloads all available pretrained model weights to the WEIGHT_DIR.

    :param local_dir_use_symlinks: (bool, optional), Whether to use symlinks for the local directory. Defaults to False.
    """
    log_weights_dir()
    weight_dir = get_weight_dir()

    logger.info(f'Downloading all weights to {weight_dir}')
    _snapshot_download_with_fallback(
        repo_id="dptech/Uni-Mol-Models",
        local_dir=weight_dir,
        allow_patterns='*',
        # local_dir_use_symlinks=local_dir_use_symlinks,
        # max_workers=8
    )


if '__main__' == __name__:
    download_all_weights()
