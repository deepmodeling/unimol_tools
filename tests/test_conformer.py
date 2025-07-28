import numpy as np
from unimol_tools.data.conformer import (
    inner_coords,
    coords2unimol,
    inner_smi2coords,
    create_mol_from_atoms_and_coords,
)
from unimol_tools.data.dictionary import Dictionary


def test_inner_coords_and_coords2unimol():
    atoms = ['C', 'H', 'O']
    coords = [[0, 0, 0], [0, 0, 1], [1, 0, 0]]
    no_h_atoms, no_h_coords = inner_coords(atoms, coords, remove_hs=True)
    assert 'H' not in no_h_atoms
    d = Dictionary()
    for a in ['C', 'O']:
        if a not in d:
            d.add_symbol(a)
    feat = coords2unimol(no_h_atoms, no_h_coords, d)
    assert feat['src_tokens'].dtype == int
    assert feat['src_coord'].shape[1] == 3


def test_inner_smi2coords_returns_mol():
    mol = inner_smi2coords('CC', return_mol=True)
    from rdkit.Chem import Mol

    assert isinstance(mol, Mol)


def test_create_mol_from_atoms_and_coords():
    atoms = ['C', 'O']
    coords = [[0, 0, 0], [1, 0, 0]]
    mol = create_mol_from_atoms_and_coords(atoms, coords)
    from rdkit.Chem import Mol

    assert isinstance(mol, Mol)
    assert mol.GetNumAtoms() == 2
