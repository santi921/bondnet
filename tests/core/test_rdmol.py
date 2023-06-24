# TODO: new construction of rdmol without rdkit or molecule construction without rdkit

import numpy as np
from rdkit.Chem import BondType
from bondnet.core.rdmol import remove_metals

from bondnet.test_utils import (
    create_LiEC_pymatgen_mol,
    create_LiEC_mol_graph,
)


def test_remove_metals():  # should still apply
    mol = create_LiEC_pymatgen_mol()
    mol = remove_metals(mol)
    assert len(mol) == 10
    assert mol.charge == -1


def test_create_rdmol_from_pmg():  # todo: update
    mol = create_LiEC_pymatgen_mol()
    pass


def test_fragment_rdmol():  # todo: update
    mol = create_LiEC_pymatgen_mol()
    pass
