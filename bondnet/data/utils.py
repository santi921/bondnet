import pandas as pd
import numpy as np
import networkx as nx
import itertools
from rdkit import Chem
from collections import deque


def get_dataset_species(molecules):
    """
    Get all the species of atoms appearing in the the molecules.

    Args:
        molecules (list): rdkit molecules

    Returns:
        list: a sequence of species string
    """
    system_species = set()
    for mol in molecules:
        if mol is None:
            continue
        species = [a.GetSymbol() for a in mol.GetAtoms()]
        system_species.update(species)

    return sorted(system_species)


def get_dataset_species_from_json(json_file):
    """
    Get all the species of atoms appearing in the the molecules.

    Args:
        json_file: file containing reaction w/ column

    Returns:
        list: a sequence of species string
    """
    df = pd.read_json(json_file)

    list(df["composition"][0].keys())

    system_species = set()
    for _, row in df.iterrows():
        if row is None:
            continue
        species = list(row["composition"].keys())
        system_species.update(species)

    return sorted(system_species)


def one_hot_encoding(x, allowable_set):
    """One-hot encoding.

    Parameters
    ----------
    x : str, int or Chem.rdchem.HybridizationType
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.

    Returns
    -------
    list
        List of int (0 or 1) where at most one value is 1.
        If the i-th value is 1, then we must have x == allowable_set[i].
    """
    return list(map(int, list(map(lambda s: x == s, allowable_set))))


def multi_hot_encoding(x, allowable_set):
    """Multi-hot encoding.

    Args:
        x (list): any type that can be compared with elements in allowable_set
        allowable_set (list): allowed values for x to take

    Returns:
        list: List of int (0 or 1) where zero or more values can be 1.
            If the i-th value is 1, then we must have allowable_set[i] in x.
    """
    return list(map(int, list(map(lambda s: s in x, allowable_set))))


def h_count_and_degree(atom_ind, bond_list, species_order):
    """
    gets the number of H-atoms connected to an atom + degree of bonding
    takes:
        atom_ind(int): index of atom
        bond_list(list of lists): list of bonds in graph
        species_order: order of atoms in graph to match nodes
    """
    # h count
    atom_bonds = []
    h_count = 0
    for i in bond_list:
        if atom_ind in i:
            atom_bonds.append(i)
    if atom_bonds != 0:
        for bond in atom_bonds:
            bond_copy = bond[:]
            bond_copy.remove(atom_ind)
            if species_order[bond_copy[0]] == "H":
                h_count += 1
    return h_count, int(len(atom_bonds))


def ring_features_from_atom(atom_ind, cycles, allowed_ring_size):
    """
    returns an atom's ring inclusion and ring size features
    takes:
        atom_ind(int) - an atom's index
        cycles(list of list) - cycles detected in the graph
        allow_ring_size(list) - list of allowed ring sizes
    returns:
        ring_inclusion - int of whether atom is in a ring
        ring_size_ret_list - one-hot list of whether

    """
    ring_inclusion = 0
    ring_size = 0  # find largest allowable ring that this atom is a part o
    ring_size_ret_list = [0 for i in allowed_ring_size]

    if cycles != []:
        min_ring = 100
        for i in cycles:
            if atom_ind in i:
                if len(i) in allowed_ring_size and len(i) < min_ring:
                    ring_inclusion = 1
                    ring_size = int(len(i))
        if min_ring < 100:
            ring_size = min_ring

    # one hot encode the detected ring size
    if ring_size != 0:
        ring_size_ret_list[allowed_ring_size.index(ring_size)] = 1

    return ring_inclusion, ring_size_ret_list


def ring_features_from_atom_full(atom_num, cycles, allowed_ring_size):
    """
    returns an atom's ring inclusion and ring size features
    takes:
        atom_num: number of atoms in molecule
        cycles(list of list) - cycles detected in the graph
        allow_ring_size(list) - list of allowed ring sizes
    returns:
        ret_dict - dictionary of ring information w/keys as atom inds

    """
    ret_dict = {}
    for i in range(atom_num):
        ret_dict[i] = ring_features_from_atom(i, cycles, allowed_ring_size)
    return ret_dict


def ring_features_from_bond(bond, cycles, allowed_ring_size):
    """
    returns an atom's ring inclusion and ring size features
    takes:
        bond_ind(int) - bond's index
        cycles(list of list) - cycles detected in the graph
        allow_ring_size(list) - list of allowed ring sizes
    returns:
        ring_inclusion - int of whether atom is in a ring
        ring_size_ret_list - one-hot list of whether
    """
    ring_inclusion = 0
    ring_size_ret_list = [0 for i in range(len(allowed_ring_size))]

    for cycle in cycles:
        if tuple(bond) in cycle or (bond[-1], bond[0]) in cycle:
            ring_inclusion = 1
            ring_size_ret_list[allowed_ring_size.index(len(cycle))] = 1

    return ring_inclusion, ring_size_ret_list


def ring_features_for_bonds_full(bonds, no_metal_binary, cycles, allowed_ring_size):
    """
    returns an atom's ring inclusion and ring size features
    takes:
        bonds - list of bonds with metal bonds included
        cycles(list of list) - cycles detected in the graph
        allow_ring_size(list) - list of allowed ring sizes
        non_metal_binary - array with one-hot encoding of whether bonds are metal/not
    returns:
        ret_dict - dictionary with bond(formatted in root-to-target) with metal-bond binary,
        ring inclusion and ring_one_hot

    """
    ret_dict = {}
    for i, bond in enumerate(bonds):
        if no_metal_binary[i] == 0:
            inclusion, ring_one_hot = ring_features_from_bond(
                bond, cycles, allowed_ring_size
            )
            ret_dict[tuple(bond)] = (
                0,
                inclusion,
                ring_one_hot,
            )
        else:  # we're never including metal bonds in ring formations
            ret_dict[tuple(bond)] = (1, 0, [0 for i in range(len(allowed_ring_size))])
    return ret_dict


def organize_list(cycle_list):
    """
    a helper function to orient cycles identically. Finds max value then adds values in the direction where the
    neighbor node's index is greater
    takes:
        cycle_list - arbitrarily defined cycles
    returns:
        cycle_org - consistently organized cycles
    """

    cycle_org = []
    for cycle in cycle_list:
        new_cycle = []
        cycle_len = int(len(cycle))
        new_cycle_start = np.argmax(cycle)
        new_cycle_direction = 1
        plus_1 = new_cycle_start + 1
        if plus_1 > cycle_len - 1:
            plus_1 -= cycle_len
        if int(cycle[new_cycle_start - 1]) > int(cycle[plus_1]):
            new_cycle_direction = -1

        for ind, node in enumerate(cycle):
            ind_next = new_cycle_start + new_cycle_direction * ind
            if ind_next > cycle_len - 1:
                ind_next -= cycle_len
            new_cycle.append(cycle[ind_next])
        cycle_org.append(new_cycle)

    return cycle_org


def filter_rotations(cycle_list):
    """
    helper function to filter repeated/rotated/reflected cycles from a list of cycles
    takes:
        cycle_list - a list of unfiltered cycles
    returns:
        ret_cycles - a list of filtered cycles
    """
    ret_cycles = []
    cycle_list = organize_list(cycle_list)
    for cycle in cycle_list:
        if cycle not in ret_cycles:
            ret_cycles.append(cycle)

    return ret_cycles


def find_rings(atom_num, bond_list, allowed_ring_size=[], edges=True):
    cycle_graphs, cycle_list = [], []
    nx_graph = nx.Graph()
    [nx_graph.add_node(i) for i in range(atom_num)]
    nx_graph.add_edges_from(bond_list)

    for i in range(atom_num):
        try:
            cycle_edges = nx.find_cycle(nx_graph, source=i)
        except:
            cycle_edges = []

        nx_graph_cycle = nx.Graph()
        nx_graph_cycle.add_edges_from(cycle_edges)

        if cycle_graphs == []:  # adds fir cycle/graph
            cycle_list.append(cycle_edges)
            cycle_graphs.append(nx_graph_cycle)

        for cycle_graph in cycle_graphs:
            # filter isomorphic edges
            if not nx.is_isomorphic(cycle_graph, nx_graph_cycle):
                cycle_list.append(cycle_edges)
                cycle_graphs.append(nx_graph_cycle)
                break

    # convert cycles found to node lists
    cycle_list_nodes = []
    for cycle in cycle_list:
        node_list = [edge[0] for edge in cycle]
        cycle_list_nodes.append(node_list)
    cycle_list = cycle_list_nodes

    # filter for allowed ring sizes
    if allowed_ring_size != []:
        cycle_list_filtered = []
        for cycle in cycle_list:
            if len(cycle) in allowed_ring_size:
                cycle_list_filtered.append(cycle)
        cycle_list = cycle_list_filtered

    cycle_list.sort()
    cycle_list = list(cycle_list for cycle_list, _ in itertools.groupby(cycle_list))
    for i in range(len(cycle_list)):
        try:
            cycle_list.remove([])
        except:
            pass

    if len(cycle_list) > 1:

        cycle_list = filter_rotations(cycle_list)

    if edges == True:
        edge_list_list = []
        for cycle in cycle_list:
            edge_list = []
            for ind, node in enumerate(cycle[:-1]):
                edge_list.append((node, cycle[ind + 1]))
            edge_list.append((cycle[-1], cycle[0]))
            edge_list_list.append(edge_list)
        return edge_list_list
    return cycle_list


def rdkit_bond_desc(mol):
    """
    uses rdkit to get a dictionary with detected bond features to allow aromaticity, bond types to be detected
    takes:
        an rdkit molecule
    returns:
        ret_dict - a dictionary with the bonds(as root, target nodes) + descriptor info

    """
    detected_bonds_dict = {}
    allowed_bond_type = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    num_atoms = len(mol.GetAtoms())

    for i in range(num_atoms):
        for j in range(num_atoms):
            try:
                bond = mol[0].GetBondBetweenAtoms(i, j)
            except:
                try:
                    bond = mol.GetBondBetweenAtoms(i, j)
                except:
                    ft = []
            if not bond is None:  # checks if rdkit has detected the bond
                ft = [int(bond.GetIsConjugated())]
                ft += one_hot_encoding(bond.GetBondType(), allowed_bond_type)
                detected_bonds_dict[i, j] = ft  # adds with key = to the bond
                
    return detected_bonds_dict
