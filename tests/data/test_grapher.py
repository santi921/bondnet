import numpy as np
from collections import defaultdict
from bondnet.data.grapher import HeteroCompleteGraphFromMolWrapper
from bondnet.test_utils import make_hetero_CH2O


def get_bond_to_atom_map(g):
    """
    Query which atoms are associated with the bonds.

    Args:
        g: dgl heterograph

    Returns:
        dict: with bond index as the key and a tuple of atom indices of atoms that
            form the bond.
    """
    nbonds = g.number_of_nodes("bond")
    bond_to_atom_map = dict()
    for i in range(nbonds):
        atoms = g.successors(i, "b2a").tolist()
        bond_to_atom_map[i] = sorted(atoms)
    return bond_to_atom_map


def get_atom_to_bond_map(g):
    """
    Query which bonds are associated with the atoms.

    Args:
        g: dgl heterograph

    Returns:
        dict: with atom index as the key and a list of indices of bonds is
        connected to the atom.
    """
    natoms = g.number_of_nodes("atom")
    atom_to_bond_map = dict()
    for i in range(natoms):
        bonds = g.successors(i, "a2b").tolist()
        atom_to_bond_map[i] = sorted(list(bonds))
    return atom_to_bond_map


def get_hetero_self_loop_map(g, ntype):
    num = g.number_of_nodes(ntype)
    if ntype == "atom":
        etype = "a2a"
    elif ntype == "bond":
        etype = "b2b"
    elif ntype == "global":
        etype = "g2g"
    else:
        raise ValueError("not supported node type: {}".format(ntype))
    self_loop_map = dict()
    for i in range(num):
        suc = g.successors(i, etype)
        self_loop_map[i] = list(suc)
    return self_loop_map


def test_build_hetero_graph():
    def assert_graph(self_loop):
        #m = make_a_mol()
        #grapher = HeteroMoleculeGraph(self_loop=self_loop)
        #g = grapher.build_graph(m)
        g, feats = make_hetero_CH2O(self_loop=self_loop)
        # number of atoms
        na = 4
        # number of bonds
        nb = 3
        # number of edges between atoms and bonds
        ne = 2 * nb

        nodes = ["atom", "bond", "global"]
        num_nodes = [g.number_of_nodes(n) for n in nodes]
        ref_num_nodes = [na, nb, 1]

        if self_loop:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb, na, nb, 1]

        else:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb]
        
        assert set(g.ntypes) == set(nodes)
        assert set(g.etypes) == set(edges)
        assert num_nodes == ref_num_nodes
        assert num_edges == ref_num_edges

        bond_to_atom_map = {
            0: [0, 1],
            1: [1, 2],
            2: [1, 3]
        }
        atom_to_bond_map = defaultdict(list)
        for b, atoms in bond_to_atom_map.items():
            for a in atoms:
                atom_to_bond_map[a].append(b)
        atom_to_bond_map = {a: sorted(bonds) for a, bonds in atom_to_bond_map.items()}

        ref_b2a_map = get_bond_to_atom_map(g)
        ref_a2b_map = get_atom_to_bond_map(g)
        
        assert bond_to_atom_map == ref_b2a_map
        assert atom_to_bond_map == ref_a2b_map

        if self_loop:
            for nt, n in zip(nodes, num_nodes):
                assert get_hetero_self_loop_map(g, nt) == {i: [i] for i in range(n)}

    assert_graph(True)
    assert_graph(False)
    
