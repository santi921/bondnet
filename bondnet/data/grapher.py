"""
Build molecule graph and then featurize it.
"""
import itertools
import dgl


class BaseGraph:
    """
    Base grapher to build DGL graph and featurizer. Typically should not use this
    directly.
    """

    def __init__(self, atom_featurizer=None, bond_featurizer=None, self_loop=False):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.self_loop = self_loop

    def build_graph(self, mol):
        raise NotImplementedError

    def featurize(self, g, mol, **kwargs):
        raise NotImplementedError

    def build_graph_and_featurize(self, mol, ret_feat_names=False, **kwargs):
        """
        Build a graph with atoms as the nodes and bonds as the edges and then featurize
        the graph.

        Args:
            mol (rdkit mol): a rdkit molecule
            kwargs: extra keyword arguments needed by featurizer

        Returns:
            (DGLGraph)
        """

        g = self.build_graph(mol)

        # print(feat_names)
        if ret_feat_names:
            g, feat_names = self.featurize(g, mol, True, **kwargs)
            return g, feat_names
        else:
            g = self.featurize(g, mol, False, **kwargs)
            return g

    @property
    def feature_size(self):
        res = {}
        if self.atom_featurizer is not None:
            res["atom"] = self.atom_featurizer.feature_size
        if self.bond_featurizer is not None:
            res["bond"] = self.bond_featurizer.feature_size
        if hasattr(self, "global_featurizer") and self.global_featurizer is not None:
            res["global"] = self.global_featurizer.feature_size
        return res

    @property
    def feature_name(self):
        res = {}
        if self.atom_featurizer is not None:
            res["atom"] = self.atom_featurizer.feature_name
        if self.bond_featurizer is not None:
            res["bond"] = self.bond_featurizer.feature_name
        if hasattr(self, "global_featurizer") and self.global_featurizer is not None:
            res["global"] = self.global_featurizer.feature_name
        return res


class HeteroCompleteGraphFromMolWrapper(BaseGraph):
    """ """

    def __init__(
        self,
        atom_featurizer=None,
        bond_featurizer=None,
        global_featurizer=None,
        self_loop=True,
    ):
        super(HeteroCompleteGraphFromMolWrapper, self).__init__(
            atom_featurizer, bond_featurizer, self_loop
        )
        self.global_featurizer = global_featurizer

    def build_graph(self, mol):
        bonds = list(mol.bonds.keys())
        num_bonds = len(bonds)
        num_atoms = len(mol.coords)
        a2b = []
        b2a = []
        if num_bonds == 0:
            num_bonds = 1
            a2b = [(0, 0)]
            b2a = [(0, 0)]

        else:
            a2b = []
            b2a = []
            for b in range(num_bonds):
                u = bonds[b][0]
                v = bonds[b][1]
                b2a.extend([[b, u], [b, v]])
                a2b.extend([[u, b], [v, b]])

        a2g = [(a, 0) for a in range(num_atoms)]
        g2a = [(0, a) for a in range(num_atoms)]
        b2g = [(b, 0) for b in range(num_bonds)]
        g2b = [(0, b) for b in range(num_bonds)]

        edges_dict = {
            ("atom", "a2b", "bond"): a2b,
            ("bond", "b2a", "atom"): b2a,
            ("atom", "a2g", "global"): a2g,
            ("global", "g2a", "atom"): g2a,
            ("bond", "b2g", "global"): b2g,
            ("global", "g2b", "bond"): g2b,
        }
        if self.self_loop:
            a2a = [(i, i) for i in range(num_atoms)]
            b2b = [(i, i) for i in range(num_bonds)]
            g2g = [(0, 0)]
            edges_dict.update(
                {
                    ("atom", "a2a", "atom"): a2a,
                    ("bond", "b2b", "bond"): b2b,
                    ("global", "g2g", "global"): g2g,
                }
            )

        g = dgl.heterograph(edges_dict)
        # add name
        g.mol_name = mol.id
        if mol.original_atom_ind != None:
            g.atom_ind = mol.original_atom_ind
        if mol.original_bond_mapping != None:
            g.bond_ind = mol.original_bond_mapping

        return g

    def featurize(self, g, row, ret_feat_names=False, **kwargs):
        if self.atom_featurizer is not None:
            feat_dict, feat_atom = self.atom_featurizer(row, **kwargs)
            g.nodes["atom"].data.update(feat_dict)

        if self.bond_featurizer is not None:
            feat_dict, feat_bond = self.bond_featurizer(row, **kwargs)
            g.nodes["bond"].data.update(feat_dict)

        if self.global_featurizer is not None:
            feat_dict, globe_feat = self.global_featurizer(row, **kwargs)
            g.nodes["global"].data.update(feat_dict)

        if ret_feat_names:
            feat_names = {}
            feat_names["atom"] = [feat_atom]
            feat_names["bond"] = [feat_bond]
            feat_names["global"] = [globe_feat]
            return g, feat_names

        return g
