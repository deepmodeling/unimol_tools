# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig, OmegaConf

from .data import DataHub
from .models import UniMolModel, UniMolV2Model
from .tasks import Trainer


class MolDataset(Dataset):
    """
    A :class:`MolDataset` class is responsible for interface of molecular dataset.
    """

    def __init__(self, data, label=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class UniMolRepr(object):
    """
    A :class:`UniMolRepr` class is responsible for interface of molecular representation by unimol
    """

    def __init__(
        self,
        cfg: DictConfig | None = None,
        data_type='molecule',
        batch_size=32,
        remove_hs=False,
        model_name='unimolv1',
        model_size='84m',
        smiles_col='SMILES',
        use_cuda=True,
        use_ddp=False,
        use_gpu='all',
        save_path=None,
        pretrained_model_path=None,
        pretrained_dict_path=None,
        **kwargs,
    ):
        """
        Initialize a :class:`UniMolRepr` class.

        :param data_type: str, default='molecule', currently support molecule, oled.
        :param batch_size: int, default=32, batch size for training.
        :param remove_hs: bool, default=False, whether to remove hydrogens in molecular.
        :param model_name: str, default='unimolv1', currently support unimolv1, unimolv2.
        :param model_size: str, default='84m', model size of unimolv2. Avaliable: 84m, 164m, 310m, 570m, 1.1B.
        :param smiles_col: str, default='SMILES', column name of SMILES.
        :param use_cuda: bool, default=True, whether to use gpu.
        :param use_ddp: bool, default=False, whether to use distributed data parallel.
        :param use_gpu: str, default='all', which gpu to use.
        :param save_path: str, default=None, path to save representation result.
        :param pretrained_model_path: str, default=None, path to pretrained model.
        :param pretrained_dict_path: str, default=None, path to pretrained model's dict file.
        :param kwargs: other parameters.
        """
        if cfg is not None:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            data_type = cfg_dict.get('data_type', data_type)
            batch_size = cfg_dict.get('batch_size', batch_size)
            remove_hs = cfg_dict.get('remove_hs', remove_hs)
            model_name = cfg_dict.get('model_name', model_name)
            model_size = cfg_dict.get('model_size', model_size)
            smiles_col = cfg_dict.get('smiles_col', smiles_col)
            use_cuda = cfg_dict.get('use_cuda', use_cuda)
            use_ddp = cfg_dict.get('use_ddp', use_ddp)
            use_gpu = cfg_dict.get('use_gpu', use_gpu)
            save_path = cfg_dict.get('save_path', save_path)
            pretrained_model_path = cfg_dict.get(
                'pretrained_model_path', pretrained_model_path
            )
            pretrained_dict_path = cfg_dict.get(
                'pretrained_dict_path', pretrained_dict_path
            )

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        if model_name == 'unimolv1':
            self.model = UniMolModel(
                output_dim=1,
                data_type=data_type,
                remove_hs=remove_hs,
                pretrained_model_path=pretrained_model_path,
                pretrained_dict_path=pretrained_dict_path,
            ).to(self.device)
        elif model_name == 'unimolv2':
            self.model = UniMolV2Model(
                output_dim=1,
                model_size=model_size,
                pretrained_model_path=pretrained_model_path,
            ).to(self.device)
        else:
            raise ValueError('Unknown model name: {}'.format(model_name))
        self.model.eval()
        self.params = {
            'data_type': data_type,
            'batch_size': batch_size,
            'remove_hs': remove_hs,
            'model_name': model_name,
            'model_size': model_size,
            'smiles_col': smiles_col,
            'use_cuda': use_cuda,
            'use_ddp': use_ddp,
            'use_gpu': use_gpu,
            'save_path': save_path,
            'pretrained_model_path': pretrained_model_path,
            'pretrained_dict_path': pretrained_dict_path,
        }

    def get_repr(self, data=None, return_atomic_reprs=False, return_tensor=False):
        """
        Get molecular representation by unimol.

        :param data: str, dict, list, numpy.ndarray, pandas.Series or pandas.Dataframe, default=None, input data for unimol.

            - str: smiles string or path to a smiles file.

            - dict: custom conformers, could take atoms and coordinates as input.

            - list: list of smiles strings.

            - numpy.ndarray: numpy.ndarray of smiles strings
            
            - pandas.Series: pandas Series of smiles strings.
            
            - pandas.DataFrame: pandas DataFrame of smiles strings or custom conformers.

        :param return_atomic_reprs: bool, default=False, whether to return atomic representations.

        :param return_tensor: bool, default=False, whether to return tensor representations. Only works when return_atomic_reprs=False.

        :return: 
            - if return_atomic_reprs=True: dict of molecular representation.
            - if return_atomic_reprs=False and return_tensor=False: list of molecular representation.
            - if return_atomic_reprs=False and return_tensor=True: tensor of molecular representation.
        """

        if isinstance(data, str):
            if data.endswith('.sdf') or data.endswith('.csv'):
                # Datahub will process sdf and csv file.
                pass
            else:
                # single smiles string.
                data = [data]
        elif isinstance(data, dict):
            # custom conformers, should take atoms and coordinates as input.
            if self.params['smiles_col'] not in data:
                assert 'atoms' in data and 'coordinates' in data
            else:
                assert isinstance(data[self.params['smiles_col']][-1], str)
        elif isinstance(data, list) or isinstance(data, np.ndarray) or isinstance(data, pd.Series):
            # list of smiles strings.
            assert isinstance(data[0], str)
        elif isinstance(data, pd.DataFrame):
            # pandas DataFrame of smiles strings.
            if self.params['smiles_col'] not in data.columns:
                assert ('atoms' in data.columns and 'coordinates' in data.columns) or 'ROMol' in data.columns
            else:
                assert isinstance(data[self.params['smiles_col']].iloc[0], str)
        else:
            raise ValueError('Unknown data type: {}'.format(type(data)))

        datahub = DataHub(
            data=data,
            task='repr',
            is_train=False,
            **self.params,
        )
        dataset = MolDataset(datahub.data['unimol_input'])
        self.trainer = Trainer(task='repr', **self.params)
        repr_output = self.trainer.inference(
            self.model,
            model_name=self.params['model_name'],
            return_repr=True,
            return_atomic_reprs=return_atomic_reprs,
            dataset=dataset,
            return_tensor=return_tensor,
        )
        return repr_output
