# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import pathlib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..utils import logger


class MolDataReader(object):
    '''A class to read Mol Data.'''

    def read_data(self, data=None, is_train=True, **params):
        # TO DO
        # 1. add anomaly detection & outlier removal.
        # 2. add support for other file format.
        # 3. add support for multi tasks.

        """
        Reads and preprocesses molecular data from various input formats for model training or prediction.
        Parsing target columns
        1. if target_cols is not None, use target_cols as target columns.
        2. if target_cols is None, use all columns with prefix 'target_col_prefix' as target columns.
        3. use given target_cols as target columns placeholder with value -1.0 for predict

        :param data: The input molecular data. Can be a file path (str), a dictionary, or a list of SMILES strings.
        :param is_train: (bool) A flag indicating if the operation is for training. Determines data processing steps.
        :param params: A dictionary of additional parameters for data processing.

        :return: A dictionary containing processed data and related information for model consumption.
        :raises ValueError: If the input data type is not supported or if any SMILES string is invalid (when strict).
        """
        task = params.get('task', None)
        target_cols = params.get('target_cols', None)
        smiles_col = params.get('smiles_col', 'SMILES')
        target_col_prefix = params.get('target_col_prefix', 'TARGET')
        anomaly_clean = params.get('anomaly_clean', False)
        smi_strict = params.get('smi_strict', False)
        split_group_col = params.get('split_group_col', 'scaffold')

        if isinstance(data, str):
            # load from file
            if data.endswith('.sdf'):
                # load sdf file
                data = PandasTools.LoadSDF(data)
                data = self._convert_numeric_columns(data)
            elif data.endswith('.csv'):
                data = pd.read_csv(data)
            else:
                raise ValueError('Unknown file type: {}'.format(data))
        elif isinstance(data, dict):
            # load from dict
            if 'target' in data:
                label = np.array(data['target'])
                if len(label.shape) == 1 or label.shape[1] == 1:
                    data[target_col_prefix] = label.reshape(-1)
                else:
                    for i in range(label.shape[1]):
                        data[target_col_prefix + str(i)] = label[:, i]

            _ = data.pop('target', None)
            
            if 'atoms' in data and 'coordinates' in data:
                if not isinstance(data['atoms'][0], list) and not isinstance(data['atoms'][0], np.ndarray):
                    data['atoms'] = [data['atoms']]
                    data['coordinates'] = [data['coordinates']]
                if not isinstance(data['atoms'][0][0], str):
                    pt = Chem.GetPeriodicTable()
                    data['atoms'] = [
                        [pt.GetElementSymbol(int(atom)) for atom in atoms]
                        for atoms in data['atoms']
                    ]
            if smiles_col in data and isinstance(data[smiles_col], str):
                # if the smiles_col is a single string, convert it to a list
                data[smiles_col] = [data[smiles_col]]
                
            data = pd.DataFrame(data)

        elif isinstance(data, pd.DataFrame):
            # load from pandas DataFrame
            if 'ROMol' in data.columns:
                data = self._convert_numeric_columns(data)
                
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            # load from smiles list
            data = pd.DataFrame(data, columns=[smiles_col])

        elif isinstance(data, pd.Series):
            # load from smiles pandas Series
            data = data.to_frame(name=smiles_col)
        else:
            raise ValueError('Unknown data type: {}'.format(type(data)))

        #### parsing target columns
        #### 1. if target_cols is not None, use target_cols as target columns.
        #### 2. if target_cols is None, use all columns with prefix 'target_col_prefix' as target columns.
        #### 3. use given target_cols as target columns placeholder with value -1.0 for predict
        if task == 'repr':
            # placeholder for repr task
            targets = None
            target_cols = None
            num_classes = None
            multiclass_cnt = None
        else:
            if target_cols is None:
                target_cols = [
                    item for item in data.columns if item.startswith(target_col_prefix)
                ]
            elif isinstance(target_cols, str):
                target_cols = [target_col.strip() for target_col in target_cols.split(',')]
            elif isinstance(target_cols, list):
                pass
            else:
                raise ValueError(
                    'Unknown target_cols type: {}'.format(type(target_cols))
                )

            if is_train:
                if anomaly_clean:
                    data = self.anomaly_clean(data, task, target_cols)
                if task == 'multiclass':
                    multiclass_cnt = int(data[target_cols].max() + 1)
            else:
                for col in target_cols:
                    if col not in data.columns or data[col].isnull().any():
                        data[col] = -1.0

            targets = data[target_cols].values.tolist()
            num_classes = len(target_cols)

        dd = {
            'raw_data': data,
            'raw_target': targets,
            'num_classes': num_classes,
            'target_cols': target_cols,
            'multiclass_cnt': (
                multiclass_cnt if task == 'multiclass' and is_train else None
            ),
        }
        if smiles_col in data.columns:
            mask = data[smiles_col].apply(
                lambda smi: self.check_smiles(smi, is_train, smi_strict)
            )
            data = data[mask]
            dd['smiles'] = data[smiles_col].tolist()
            dd['scaffolds'] = data[smiles_col].map(self.smi2scaffold).tolist()
        elif 'ROMol' in data.columns:
            dd['smiles'] = None
            dd['scaffolds'] = data['ROMol'].apply(self.mol2scaffold).tolist()
        else:
            dd['smiles'] = None
            dd['scaffolds'] = None

        if split_group_col in data.columns:
            dd['group'] = data[split_group_col].tolist()
        elif split_group_col == 'scaffold':
            dd['group'] = dd['scaffolds']
        else:
            dd['group'] = None

        if 'atoms' in data.columns and 'coordinates' in data.columns:
            dd['atoms'] = data['atoms'].tolist()
            dd['coordinates'] = data['coordinates'].tolist()

        if 'ROMol' in data.columns:
            dd['mols'] = data['ROMol'].tolist()

        return dd

    def check_smiles(self, smi, is_train, smi_strict):
        """
        Validates a SMILES string and decides whether it should be included based on training mode and strictness.

        :param smi: (str) The SMILES string to check.
        :param is_train: (bool) Indicates if this check is happening during training.
        :param smi_strict: (bool) If true, invalid SMILES strings raise an error, otherwise they're logged and skipped.

        :return: (bool) True if the SMILES string is valid, False otherwise.
        :raises ValueError: If the SMILES string is invalid and strict mode is on.
        """
        if Chem.MolFromSmiles(smi) is None:
            if is_train and not smi_strict:
                logger.info(f'Illegal SMILES clean: {smi}')
                return False
            else:
                raise ValueError(f'SMILES rule is illegal: {smi}')
        return True

    def smi2scaffold(self, smi):
        """
        Converts a SMILES string to its corresponding scaffold.

        :param smi: (str) The SMILES string to convert.

        :return: (str) The scaffold of the SMILES string, or the original SMILES if conversion fails.
        """
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smi, includeChirality=True
            )
        except:
            return smi

    def mol2scaffold(self, mol):
        """
        Converts an RDKit molecule to its corresponding scaffold.

        :param mol: (RDKit Mol) The molecule to convert.

        :return: (str) The scaffold of the molecule, or the original SMILES if conversion fails.
        """
        if not isinstance(mol, Chem.Mol):
            raise ValueError('Input must be an RDKit Mol object')
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=True
            )
        except:
            return Chem.MolToSmiles(mol, includeChirality=True)

    def anomaly_clean(self, data, task, target_cols):
        """
        Performs anomaly cleaning on the data based on the specified task.

        :param data: (DataFrame) The dataset to be cleaned.
        :param task: (str) The type of task which determines the cleaning strategy.
        :param target_cols: (list) The list of target columns to consider for cleaning.

        :return: (DataFrame) The cleaned dataset.
        :raises ValueError: If the provided task is not recognized.
        """
        if task in [
            'classification',
            'multiclass',
            'multilabel_classification',
            'multilabel_regression',
        ]:
            return data
        if task == 'regression':
            return self.anomaly_clean_regression(data, target_cols)
        else:
            raise ValueError('Unknown task: {}'.format(task))

    def anomaly_clean_regression(self, data, target_cols):
        """
        Performs anomaly cleaning specifically for regression tasks using a 3-sigma threshold.

        :param data: (DataFrame) The dataset to be cleaned.
        :param target_cols: (list) The list of target columns to consider for cleaning.

        :return: (DataFrame) The cleaned dataset after applying the 3-sigma rule.
        """
        sz = data.shape[0]
        target_col = target_cols[0]
        _mean, _std = data[target_col].mean(), data[target_col].std()
        data = data[
            (data[target_col] > _mean - 3 * _std)
            & (data[target_col] < _mean + 3 * _std)
        ]
        logger.info(
            'Anomaly clean with 3 sigma threshold: {} -> {}'.format(sz, data.shape[0])
        )
        return data

    def _convert_numeric_columns(self, df):
        """
        Try to convert all columns in the DataFrame to numeric types, except for the 'ROMol' column.
        
        :param df: DataFrame to be converted.
        :return: DataFrame with numeric columns.
        """
        for col in df.columns:
            if col == 'ROMol': 
                continue
                
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.isna().sum() / len(df) < 0.1:  # Allow up to 10% NaN values
                    df[col] = numeric_series
                    logger.debug(f"Column '{col}' converted to numeric type")
            except:
                pass
                
        return df
