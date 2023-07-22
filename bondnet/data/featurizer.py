"""
Featurize a molecule heterograph of atom, bond, and global nodes with RDkit.
"""

import torch
import os
import warnings
import itertools
from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem.rdchem import GetPeriodicTable
import networkx as nx
from bondnet.utils import *
from bondnet.data.utils import (
    one_hot_encoding,
    h_count_and_degree,
    ring_features_from_atom_full,
    ring_features_for_bonds_full,
    rdkit_bond_desc,
    find_rings,
)
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class BaseFeaturizer:
    def __init__(self, dtype="float32"):
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                "`dtype` should be `float32` or `float64`, but got `{}`.".format(dtype)
            )
        self.dtype = dtype
        self._feature_size = None
        self._feature_name = None

    @property
    def feature_size(self):
        """
        Returns:
            an int of the feature size.
        """
        return self._feature_size

    @property
    def feature_name(self):
        """
        Returns:
            a list of the names of each feature. Should be of the same length as
            `feature_size`.
        """

        return self._feature_name

    def __call__(self, mol, **kwargs):
        """
        Returns:
            A dictionary of the features.
        """
        raise NotImplementedError


class BondFeaturizer(BaseFeaturizer):
    """
    Base featurize all bonds in a molecule.

    The bond indices will be preserved, i.e. feature i corresponds to atom i.
    The number of features will be equal to the number of bonds in the molecule,
    so this is suitable for the case where we represent bond as graph nodes.

    Args:
        length_featurizer (str or None): the featurizer for bond length.
        length_featurizer_args (dict): a dictionary of the arguments for the featurizer.
            If `None`, default values will be used, but typically not good because this
            should be specific to the dataset being used.
    """

    def __init__(
        self, length_featurizer=None, length_featurizer_args=None, dtype="float32"
    ):
        super(BondFeaturizer, self).__init__(dtype)
        self._feature_size = None
        self._feature_name = None

        if length_featurizer == "bin":
            if length_featurizer_args is None:
                length_featurizer_args = {"low": 0.0, "high": 2.5, "num_bins": 10}
            self.length_featurizer = DistanceBins(**length_featurizer_args)
        elif length_featurizer == "rbf":
            if length_featurizer_args is None:
                length_featurizer_args = {"low": 0.0, "high": 2.5, "num_centers": 10}
            self.length_featurizer = RBF(**length_featurizer_args)
        elif length_featurizer is None:
            self.length_featurizer = None
        else:
            raise ValueError(
                "Unsupported bond length featurizer: {}".format(length_featurizer)
            )


class DistanceBins(BaseFeaturizer):
    """
    Put the distance into a bins. As used in MPNN.

    Args:
        low (float): lower bound of bin. Values smaller than this will all be put in
            the same bin.
        high (float): upper bound of bin. Values larger than this will all be put in
            the same bin.
        num_bins (int): number of bins. Besides two bins (one smaller than `low` and
            one larger than `high`) a number of `num_bins -2` bins will be evenly
            created between [low, high).

    """

    def __init__(self, low=2.0, high=6.0, num_bins=10):
        super(DistanceBins, self).__init__()
        self.num_bins = num_bins
        self.bins = np.linspace(low, high, num_bins - 1, endpoint=True)
        self.bin_indices = np.arange(num_bins)

    @property
    def feature_size(self):
        return self.num_bins

    @property
    def feature_name(self):
        return ["dist bins"] * self.feature_size

    def __call__(self, distance):
        v = np.digitize(distance, self.bins)
        return one_hot_encoding(v, self.bin_indices)


class RBF(BaseFeaturizer):
    """
    Radial basis functions.
    e(d) = exp(- gamma * ||d - mu_k||^2), where gamma = 1/delta

    Parameters
    ----------
    low : float
        Smallest value to take for mu_k, default to be 0.
    high : float
        Largest value to take for mu_k, default to be 4.
    num_centers : float
        Number of centers
    """

    def __init__(self, low=0.0, high=4.0, num_centers=20):
        super(RBF, self).__init__()
        self.num_centers = num_centers
        self.centers = np.linspace(low, high, num_centers)
        self.gap = self.centers[1] - self.centers[0]

    @property
    def feature_size(self):
        return self.num_centers

    @property
    def feature_name(self):
        return ["rbf"] * self.feature_size

    def __call__(self, edge_distance):
        """
        Parameters
        ----------
        edge_distance : float
            Edge distance
        Returns
        -------
        a list of RBF values of size `num_centers`
        """
        radial = edge_distance - self.centers
        coef = -1 / self.gap
        return list(np.exp(coef * (radial**2)))


class BondAsNodeGraphFeaturizerGeneral(BondFeaturizer):
    """BaseFeaturizer
    Featurize all bonds in a molecule.

    The bond indices will be preserved, i.e. feature i corresponds to atom i.
    The number of features will be equal to the number of bonds in the molecule,
    so this is suitable for the case where we represent bond as graph nodes.

    See Also:
        BondAsEdgeBidirectedFeaturizer
    """

    def __init__(
        self,
        length_featurizer=None,
        length_featurizer_args=None,
        dtype="float32",
        selected_keys=[],
    ):
        super(BondFeaturizer, self).__init__(dtype)
        self._feature_size = None
        self._feature_name = None
        self.selected_keys = selected_keys

        if length_featurizer == "bin":
            if length_featurizer_args is None:
                length_featurizer_args = {"low": 0.0, "high": 2.5, "num_bins": 10}
            self.length_featurizer = DistanceBins(**length_featurizer_args)
        elif length_featurizer == "rbf":
            if length_featurizer_args is None:
                length_featurizer_args = {"low": 0.0, "high": 2.5, "num_centers": 10}
            self.length_featurizer = RBF(**length_featurizer_args)
        elif length_featurizer is None:
            self.length_featurizer = None
        else:
            raise ValueError(
                "Unsupported bond length featurizer: {}".format(length_featurizer)
            )

    def __call__(self, mol, **kwargs):
        """
        Parameters
        ----------
        mol : molwrapper object with xyz positions(coord) + electronic information

        Returns
        -------
            Dictionary for bond features
        """

        feats, bond_list_only_metal, no_metal_binary = [], [], []
        num_atoms = 0
        # num_feats = 18
        allowed_ring_size = [3, 4, 5, 6, 7]

        bond_list = list(mol.bonds)
        num_bonds = len(bond_list)
        bond_list_no_metal = mol.nonmetal_bonds
        num_atoms = int(mol.num_atoms)
        features = mol.bond_features
        xyz_coordinates = mol.coords
        # print("features in bond featurizer", features)
        # print("selected keys in bond featuzizer", self.selected_keys)

        # count number of keys in features
        num_feats = len(self.selected_keys)
        num_feats += 7

        if num_bonds == 0:
            ft = [0.0 for _ in range(num_feats)]
            if self.length_featurizer:
                ft += [0.0 for _ in range(len(self.length_featurizer.feature_name))]
            feats = [ft]

        else:
            features_flatten = []
            for i in range(num_bonds):
                feats_flatten_temp = []
                for key in self.selected_keys:
                    if key != "bond_length":
                        feats_flatten_temp.append(features[key][i])
                features_flatten.append(feats_flatten_temp)

            feats = []
            for i in bond_list:
                if i not in bond_list_no_metal:
                    bond_list_only_metal.append(i)
                    no_metal_binary.append(0)
                else:
                    no_metal_binary.append(1)

            cycles = find_rings(
                num_atoms, bond_list_no_metal, allowed_ring_size, edges=True
            )

            ring_dict = ring_features_for_bonds_full(
                bond_list, no_metal_binary, cycles, allowed_ring_size
            )
            ring_dict_keys = list(ring_dict.keys())

            for ind, bond in enumerate(bond_list):
                ft = []

                if tuple(bond) in ring_dict_keys:
                    ft.append(ring_dict[tuple(bond)][0])  # metal
                    ft.append(ring_dict[tuple(bond)][1])  #
                    ft += ring_dict[tuple(bond)][2]  # one hot ring
                else:
                    ft += [0, 0]
                    ft += [0 for i in range(len(allowed_ring_size))]

                # check that features_flatten isn't empty lists
                if features_flatten[ind] != []:
                    ft += features_flatten[ind]

                if "bond_length" in self.selected_keys:
                    bond_len = np.sqrt(
                        np.sum(
                            np.square(
                                np.array(xyz_coordinates[bond[0]])
                                - np.array(xyz_coordinates[bond[1]])
                            )
                        )
                    )

                    ft.append(bond_len)

                # ft += features[bond[0]] # check that index is correct
                feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            ["metal bond"] + ["ring inclusion"] + ["ring size"] * 5 + self.selected_keys
        )

        if self.length_featurizer:
            self._feature_name += self.length_featurizer.feature_name

        return {"feat": feats}, self._feature_name


class AtomFeaturizerGraphGeneral(BaseFeaturizer):

    """
    Featurize atoms in a molecule.

    Mimimum set of info without hybridization info.
    """

    def __init__(self, selected_keys=[], dtype="float32"):
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                "`dtype` should be `float32` or `float64`, but got `{}`.".format(dtype)
            )
        self.dtype = dtype
        self._feature_size = None
        self._feature_name = None
        self.selected_keys = selected_keys

    def __call__(self, mol, **kwargs):
        """
        Args:
            mol: molecular wraper object w/electronic info

            Also `extra_feats_info` should be provided as `kwargs` as additional info.

        Returns:
            Dictionary of atom features
        """

        try:
            species = sorted(kwargs["dataset_species"])
        except KeyError as e:
            raise KeyError(
                "{} `dataset_species` needed for {}.".format(e, self.__class__.__name__)
            )

        allowed_ring_size = [3, 4, 5, 6, 7]

        features = mol.atom_features
        features_flatten, feats, bond_list = [], [], []
        num_atoms = len(mol.coords)
        species_sites = mol.species
        bond_list_tuple = list(mol.bonds.keys())
        # print("atom feats,", features)
        for i in range(num_atoms):
            feats_flatten_temp = []
            for key in self.selected_keys:
                # print(key, features[key])
                feats_flatten_temp.append(features[key][i])
            features_flatten.append(feats_flatten_temp)

        atom_num = len(species_sites)
        [bond_list.append(list(bond)) for bond in bond_list_tuple]
        cycles = find_rings(atom_num, bond_list, edges=False)
        ring_info = ring_features_from_atom_full(num_atoms, cycles, allowed_ring_size)

        for atom_ind in range(num_atoms):
            ft = []
            atom_element = species_sites[atom_ind]
            h_count, degree = h_count_and_degree(atom_ind, bond_list, species_sites)
            ring_inclusion, ring_size_list = ring_info[atom_ind]
            ft.append(degree)
            ft.append(ring_inclusion)
            ft.append(h_count)

            ft += features_flatten[atom_ind]
            ft += one_hot_encoding((atom_element), species)
            ft += ring_size_list
            feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            ["total degree", "is in ring", "total H"]
            + self.selected_keys
            + ["chemical symbol"] * len(species)
            + ["ring size"] * len(allowed_ring_size)
        )

        return {"feat": feats}, self._feature_name


class GlobalFeaturizerGraph(BaseFeaturizer):
    """
    Featurize the global state of a molecules using number of atoms, number of bonds,
    molecular weight, and optionally charge and solvent environment.


    Args:
        allowed_charges (list, optional): charges allowed the the molecules to take.
        solvent_environment (list, optional): solvent environment in which the
        calculations for the molecule take place
    """

    def __init__(
        self,
        allowed_charges=None,
        solvent_environment=None,
        functional_g_basis=None,
        dtype="float32",
    ):
        super(GlobalFeaturizerGraph, self).__init__(dtype)
        self.allowed_charges = allowed_charges
        self.solvent_environment = solvent_environment
        self.functional_g_basis = functional_g_basis

    def __call__(self, mol, **kwargs):
        """
        mol can either be an molwrapper object
        """
        pt = GetPeriodicTable()
        num_atoms, mw = 0, 0

        atom_types = list(mol.composition_dict.keys())
        for atom in atom_types:
            num_atom_type = int(mol.composition_dict[atom])
            num_atoms += num_atom_type
            mw += num_atom_type * pt.GetAtomicWeight(atom)

        g = [
            num_atoms,
            # len(mol[bond_key]),
            len(mol.bonds),
            mw,
        ]

        if (
            self.allowed_charges is not None
            or self.solvent_environment is not None
            or self.functional_g_basis is not None
        ):
            try:
                feats_info = kwargs["extra_feats_info"]
            except KeyError as e:
                raise KeyError(
                    "{} `extra_feats_info` needed for {}.".format(
                        e, self.__class__.__name__
                    )
                )

            if self.allowed_charges is not None:
                g += one_hot_encoding(feats_info["charge"], self.allowed_charges)

            if self.solvent_environment is not None:
                # if only two solvent_environment, we use 0/1 to denote the feature
                if len(self.solvent_environment) == 2:
                    ft = self.solvent_environment.index(feats_info["environment"])
                    g += [ft]
                # if more than two, we create a one-hot encoding
                else:
                    g += one_hot_encoding(
                        feats_info["environment"], self.solvent_environment
                    )
            if self.functional_g_basis is not None:
                print("functional group info", self.functional_g_basis)
                g += one_hot_encoding(mol.functional_group, self.functional_g_basis)

        feats = torch.tensor([g], dtype=getattr(torch, self.dtype))

        self._feature_size = feats.shape[1]
        self._feature_name = ["num atoms", "num bonds", "molecule weight"]
        if self.allowed_charges is not None:
            self._feature_name += ["charge one hot"] * len(self.allowed_charges)

        if self.functional_g_basis is not None:
            self._feature_name += ["hydrolysed functional group"] * len(
                self.functional_g_basis
            )

        if self.solvent_environment is not None:
            if len(self.solvent_environment) == 2:
                self._feature_name += ["solvent"]
            else:
                self._feature_name += ["solvent"] * len(self.solvent_environment)

        return {"feat": feats}, self._feature_name
