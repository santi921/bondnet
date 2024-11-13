import copy
import logging
import warnings
import numpy as np
import itertools
import networkx as nx
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from bondnet.core.rdmol import create_rdkit_mol_from_mol_graph
from bondnet.utils import create_directory, to_path, yaml_dump
from pymatgen.core.structure import Molecule, Structure

logger = logging.getLogger(__name__)


class MoleculeWrapper:
    """
    A wrapper of pymatgen Molecule, MoleculeGraph, rdkit Chem.Mol... to make it
    easier to use molecules.

    Arguments:
        mol_graph (MoleculeGraph): pymatgen molecule graph instance
        free_energy (float): free energy of the molecule
        id (str): (unique) identification of the molecule
    """

    def __init__(
        self,
        mol_graph,
        functional_group=None,
        free_energy=None,
        id=None,
        non_metal_bonds=None,
        atom_features={},
        bond_features={},
        global_features={},
        original_atom_ind=None,
        original_bond_mapping=None,
    ):
        self.mol_graph = mol_graph
        self.pymatgen_mol = mol_graph.molecule
        self.nonmetal_bonds = non_metal_bonds
        self.free_energy = free_energy
        self.functional_group = functional_group
        self.id = id
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.global_features = global_features
        self.original_atom_ind = original_atom_ind
        self.original_bond_mapping = original_bond_mapping
        # print("final bond features:")
        # print(self.bond_features)
        # set when corresponding method is called
        self._rdkit_mol = None
        self._fragments = None
        self._isomorphic_bonds = None

    @property
    def charge(self):
        """
        Returns:
            int: charge of the molecule
        """
        return self.pymatgen_mol.charge

    @property
    def formula(self):
        """
        Returns:
            str: chemical formula of the molecule, e.g. H2CO3.
        """
        return self.pymatgen_mol.composition.alphabetical_formula.replace(" ", "")

    @property
    def composition_dict(self):
        """
        Returns:
            dict: with chemical species as key and number of the species as value.
        """
        d = self.pymatgen_mol.composition.as_dict()
        return {k: int(v) for k, v in d.items()}

    @property
    def weight(self):
        """
        Returns:
            int: molecule weight
        """
        return self.pymatgen_mol.composition.weight

    @property
    def num_atoms(self):
        """
        Returns:
            int: number of atoms in molecule
        """
        return len(self.pymatgen_mol)

    @property
    def species(self):
        """
        Species of atoms. Order is the same as self.atoms.
        Returns:
            list: Species string.
        """
        return [str(s) for s in self.pymatgen_mol.species]

    @property
    def coords(self):
        """
        Returns:
            2D array: of shape (N, 3) where N is the number of atoms.
        """
        return np.asarray(self.pymatgen_mol.cart_coords)

    @property
    def bonds(self):
        """
        Returns:
            dict: with bond index (a tuple of atom indices) as the key and and bond
                attributes as the value.
        """
        return {tuple(sorted([i, j])): attr for i, j, attr in self.graph.edges.data()}

    @property
    def graph(self):
        """
        Returns:
            networkx graph used by mol_graph
        """
        return self.mol_graph.graph

    @property
    def rdkit_mol(self):
        """
        Returns:
            rdkit molecule
        """
        if self._rdkit_mol is None:
            self._rdkit_mol, _ = create_rdkit_mol_from_mol_graph(
                self.mol_graph, name=str(self), force_sanitize=False
            )
        return self._rdkit_mol

    @rdkit_mol.setter
    def rdkit_mol(self, m):
        self._rdkit_mol = m

    @property
    def fragments(self):
        """
        Get fragments of the molecule by breaking all the bonds.

        Returns:
            A dictionary with bond index (a tuple (idx1, idx2)) as key, and a list
            of the mol_graphs of the fragments as value (each list is of size 1 or 2).
            The dictionary is empty if the mol has no bonds.
        """
        if self._fragments is None:
            bonds = [b for b, _ in self.bonds.items()]
            self._fragments = fragment_mol_graph(self.mol_graph, bonds)
        return self._fragments

    @property
    def isomorphic_bonds(self):
        r"""
        Find isomorphic bonds. Isomorphic bonds are defined as bonds that the same
        fragments (in terms of fragment connectivity) are obtained by breaking the bonds
        separately.

        For example, for molecule

               0     1
            H1---C0---H2
              2 /  \ 3
              O3---O4
                 4

        bond 0 is isomorphic to bond 1, bond 2 is isomorphic to bond 3 , bond 4 is not
        isomorphic to any other bond.

        Note:
            Bond not isomorphic to any other bond is included as a group by itself.

        Returns:
            list of list: each inner list contains the indices (a 2-tuple) of bonds that
                are isomorphic. For the above example, this function
                returns [[(0,1), (0,2)], [(0,3), (0,4)], [(3,4)]]
        """

        if self._isomorphic_bonds is None:
            iso_bonds = []

            for bond1, frags1 in self.fragments.items():
                for group in iso_bonds:
                    # compare to the first in a group to see whether it is isomorphic
                    bond2 = group[0]
                    frags2 = self.fragments[bond2]

                    if len(frags1) == len(frags2) == 1:
                        if frags1[0].isomorphic_to(frags2[0]):
                            group.append(bond1)
                            break
                    elif len(frags1) == len(frags2) == 2:
                        if (
                            frags1[0].isomorphic_to(frags2[0])
                            and frags1[1].isomorphic_to(frags2[1])
                        ) or (
                            frags1[0].isomorphic_to(frags2[1])
                            and frags1[1].isomorphic_to(frags2[0])
                        ):
                            group.append(bond1)
                            break

                # bond1 not in any group
                else:
                    iso_bonds.append([bond1])

            self._isomorphic_bonds = iso_bonds

        return self._isomorphic_bonds

    def is_atom_in_ring(self, atom):
        """
        Whether an atom in ring.

        Args:
            atom (int): atom index

        Returns:
            bool: atom in ring or not
        """
        ring_info = self.mol_graph.find_rings()
        ring_atoms = set([atom for ring in ring_info for bond in ring for atom in bond])
        return atom in ring_atoms

    def is_bond_in_ring(self, bond):
        """
        Whether a bond in ring.

        Args:
            bond (tuple): bond index

        Returns:
            bool: bond in ring or not
        """
        ring_info = self.mol_graph.find_rings()
        ring_bonds = set([tuple(sorted(bond)) for ring in ring_info for bond in ring])
        return tuple(sorted(bond)) in ring_bonds

    def get_sdf_bond_indices(self, zero_based=False, sdf=None):
        """
        Get the indices of bonds as specified in the sdf file.

        zero_based (bool): If True, the atom index will be converted to zero based.
        sdf (str): the sdf string for parsing. If None, it is created from the mol.

        Returns:
            list of tuple: each tuple specifies a bond.
        """
        sdf = sdf or self.write()

        lines = sdf.split("\n")
        start = end = 0
        for i, ln in enumerate(lines):
            if "BEGIN BOND" in ln:
                start = i + 1
            if "END BOND" in ln:
                end = i
                break

        bonds = [
            tuple(sorted([int(i) for i in ln.split()[4:6]])) for ln in lines[start:end]
        ]

        if zero_based:
            bonds = [(b[0] - 1, b[1] - 1) for b in bonds]

        return bonds

    def get_sdf_bond_indices_v2000(self, sdf=None):
        """
        Get the indices of bonds as specified in the sdf file.

        Returns:
            list of tuple: each tuple specifies a bond.
        """
        sdf = sdf or self.write(v3000=False)
        lines = sdf.split("\n")
        split_3 = lines[3].split()
        natoms = int(split_3[0])
        nbonds = int(split_3[1])
        bonds = []
        for line in lines[4 + natoms : 4 + natoms + nbonds]:
            bonds.append(tuple(sorted([int(i) for i in line.split()[:2]])))
        return bonds

    def subgraph_atom_mapping(self, bond):
        """
        Find the atoms in the two subgraphs by breaking a bond in a molecule.

        Returns:
            tuple of list: each list contains the atoms in one subgraph.
        """

        original = copy.deepcopy(self.mol_graph)
        original.break_edge(bond[0], bond[1], allow_reverse=True)

        # A -> B breaking
        if nx.is_weakly_connected(original.graph):
            mapping = list(range(self.num_atoms))
            return mapping, mapping
        # A -> B + C breaking
        else:
            components = nx.weakly_connected_components(original.graph)
            nodes = [original.graph.subgraph(c).nodes for c in components]
            mapping = tuple([sorted(list(n)) for n in nodes])
            if len(mapping) != 2:
                raise Exception("Mol not split into two parts")
            return mapping

    def find_ring(self, by_species=False):
        """
        Find all rings in the molecule.

        Args:
            by_species (bool): If False, the rings will be denoted by atom indices. If
                True, denoted by atom species.

        Returns:
            list of list: each inner list holds the atoms (index or specie) of a ring.
        """
        rings = self.mol_graph.find_rings()

        rings_once_per_atom = []
        for r in rings:
            # the ring is given by the connectivity info. For example, for a 1-2-3 ring,
            # r would be something like [(1,2), (2,3), (3,1)]
            # here we remove the repeated atoms and let each atom appear only once
            atoms = []
            for i in r:
                atoms.extend(i)
            atoms = list(set(atoms))
            if by_species:
                atoms = [self.species[j] for j in atoms]
            rings_once_per_atom.append(atoms)

        return rings_once_per_atom

    def write(self, filename=None, name=None, format="sdf", kekulize=True, v3000=True):
        """Write a molecule to file or as string using rdkit.

        Args:
            filename (str): name of the file to write the output. If None, return the
                output as string.
            name (str): name of a molecule. If `file_format` is sdf, this is the first
                line the molecule block in the sdf.
            format (str): format of the molecule, supporting: sdf, pdb, and smi.
            kekulize (bool): whether to kekulize the mol if format is `sdf`
            v3000 (bool): whether to force v3000 form if format is `sdf`
        """
        if filename is not None:
            create_directory(filename)
            filename = str(to_path(filename))

        name = str(self.id) if name is None else name
        self.rdkit_mol.SetProp("_Name", name)

        if format == "sdf":
            if filename is None:
                sdf = Chem.MolToMolBlock(
                    self.rdkit_mol, kekulize=kekulize, forceV3000=v3000
                )
                return sdf + "$$$$\n"
            else:
                return Chem.MolToMolFile(
                    self.rdkit_mol, filename, kekulize=kekulize, forceV3000=v3000
                )
        elif format == "pdb":
            if filename is None:
                sdf = Chem.MolToPDBBlock(self.rdkit_mol)
                return sdf + "$$$$\n"
            else:
                return Chem.MolToPDBFile(self.rdkit_mol, filename)
        elif format == "smi":
            return Chem.MolToSmiles(self.rdkit_mol)
        else:
            raise ValueError(f"format {format} currently not supported")

    def write_custom(self, index):
        bonds = self.bonds
        bond_count = len(bonds)
        atom_count = len(self.pymatgen_mol.sites)
        sdf = ""
        name = "{}_{}_{}_{}_index-{}".format(
            self.id, self.formula, self.charge, self.free_energy, index
        )
        sdf += name + "\n"
        sdf += "     RDKit          3D\n\n"
        sdf += "  0  0  0  0  0  0  0  0  0  0999 V3000\n"
        sdf += "M  V30 BEGIN CTAB\n"
        sdf += "M  V30 COUNTS {} {} 0 0 0\n".format(atom_count, bond_count)
        sdf += "M  V30 BEGIN ATOM\n"
        # this is done
        for ind in range(len(self.pymatgen_mol.sites)):
            charge = self.rdkit_mol.GetAtomWithIdx(ind).GetFormalCharge()
            element = self.pymatgen_mol[ind].as_dict()["species"][0]["element"]
            x, y, z = self.pymatgen_mol[ind].as_dict()["xyz"]
            if charge != 0:
                sdf += "M  V30 {} {} {:.5f} {:.5f} {:.5f} 0 CHG={}\n".format(
                    ind + 1, element, x, y, z, charge
                )
            else:
                sdf += "M  V30 {} {} {:.5f} {:.5f} {:.5f} 0\n".format(
                    ind + 1, element, x, y, z
                )

        sdf += "M  V30 END ATOM\n"
        if atom_count > 1:
            sdf += "M  V30 BEGIN BOND\n"
            """
            if(bond_count == 0): 
                a_atom = self.pymatgen_mol[0].as_dict()["species"][0]['element']
                b_atom = self.pymatgen_mol[1].as_dict()["species"][0]['element']
                if(a_atom=='H' or b_atom=='H' ): order = 1
                if(a_atom=='F' or b_atom=='F' or a_atom == 'Cl' or b_atom == 'Cl'): order = 1
                if(a_atom=='N' or b_atom=='N'): order = 3
                if(a_atom=="O" or b_atom=='O'): order = 2
                sdf += "M  V30 {} {} {} {}\n".format(1, order, 1, 2)
            """
            for ind, bond in enumerate(bonds):
                double_cond = False
                a, b = bond
                try:
                    double_cond = "DOUBLE" == str(
                        self.rdkit_mol.GetBondBetweenAtoms(a, b).GetBondType()
                    )
                except:
                    pass
                if double_cond:
                    order = 2
                else:
                    order = 1
                sdf += "M  V30 {} {} {} {}\n".format(ind + 1, order, a + 1, b + 1)

            sdf += "M  V30 END BOND\n"
        sdf += "M  V30 END CTAB\n"
        sdf += "M  END\n"
        sdf += "$$$$\n"

        return sdf

    def draw(self, filename=None, show_atom_idx=False):
        """
        Draw the molecule.

        Args:
            filename (str): path to the save the generated image. If `None` the
                molecule is returned and can be viewed in Jupyter notebook.
        """
        m = copy.deepcopy(self.rdkit_mol)
        AllChem.Compute2DCoords(m)

        if show_atom_idx:
            for a in m.GetAtoms():
                a.SetAtomMapNum(a.GetIdx() + 1)
        # d.drawOptions().addAtomIndices = True

        if filename is None:
            return m
        else:
            create_directory(filename)
            filename = str(to_path(filename))
            Draw.MolToFile(m, filename)

    def draw_with_bond_note(self, bond_note, filename="mol.png", show_atom_idx=True):
        """
        Draw molecule using rdkit and show bond annotation, e.g. bond energy.

        Args:
            bond_note (dict): {bond_index: note}. The note to show for the
                corresponding bond.
            filename (str): path to the save the generated image. If `None` the
                molecule is returned and can be viewed in Jupyter notebook.
        """
        m = self.draw(show_atom_idx=show_atom_idx)

        # set bond annotation
        highlight_bonds = []
        for bond, note in bond_note.items():
            if isinstance(note, (float, np.floating)):
                note = "{:.3g}".format(note)
            idx = m.GetBondBetweenAtoms(*bond).GetIdx()
            m.GetBondWithIdx(idx).SetProp("bondNote", note)
            highlight_bonds.append(idx)

        # set highlight color
        bond_colors = {b: (192 / 255, 192 / 255, 192 / 255) for b in highlight_bonds}

        d = rdMolDraw2D.MolDraw2DCairo(400, 300)

        # smaller font size
        d.SetFontSize(0.8 * d.FontSize())

        rdMolDraw2D.PrepareAndDrawMolecule(
            d, m, highlightBonds=highlight_bonds, highlightBondColors=bond_colors
        )
        d.FinishDrawing()

        create_directory(filename)
        with open(to_path(filename), "wb") as f:
            f.write(d.GetDrawingText())

    def pack_features(self, broken_bond=None):
        feats = dict()
        feats["charge"] = self.charge
        return feats

    def __expr__(self):
        return f"{self.id}_{self.formula}"

    def __str__(self):
        return self.__expr__()


def create_wrapper_mol_from_atoms_and_bonds(
    species,
    coords,
    bonds,
    charge=0,
    free_energy=None,
    functional_group=None,
    identifier=None,
    original_atom_ind=None,
    original_bond_ind=None,
    atom_features={},
    bond_features={},
    global_features={},
):
    """
    Create a :class:`MoleculeWrapper` from atoms and bonds.

    Args:
        species (list of str): atom species str
        coords (2D array): positions of atoms
        bonds (list of tuple): each tuple is a bond (atom indices)
        charge (int): charge of the molecule
        free_energy (float): free energy of the molecule
        identifier (str): (unique) identifier of the molecule
        original_atom_ind(list of indices):  atoms, in order

    Returns:
        MoleculeWrapper instance
    """

    pymatgen_mol = Molecule(species, coords, charge)
    bonds = {tuple(sorted(b)): None for b in bonds}
    mol_graph = MoleculeGraph.with_edges(pymatgen_mol, bonds)
    mol_wrapper = MoleculeWrapper(
        mol_graph,
        free_energy=free_energy,
        functional_group=functional_group,
        id=identifier,
        original_atom_ind=original_atom_ind,
        original_bond_mapping=original_bond_ind,
        atom_features=atom_features,
        bond_features=bond_features,
        global_features=global_features,
    )

    return mol_wrapper


def rdkit_mol_to_wrapper_mol(m, charge=None, free_energy=None, identifier=None):
    """
    Convert an rdkit molecule to a :class:`MoleculeWrapper` molecule.

    This constructs a molecule graph from the rdkit mol and assigns the rdkit mol
    to the molecule wrapper.

    Args:
        m (Chem.Mol): rdkit molecule
        charge (int): charge of the molecule. If None, inferred from the rdkit mol;
            otherwise, the provided charge will override the inferred.
        free_energy (float): free energy of the molecule
        identifier (str): (unique) identifier of the molecule

    Returns:
        MoleculeWrapper instance
    """

    species = [a.GetSymbol() for a in m.GetAtoms()]

    # coords = m.GetConformer().GetPositions()
    # NOTE, the above way to get coords results in segfault on linux, so we use the
    # below workaround
    conformer = m.GetConformer()
    coords = [[x for x in conformer.GetAtomPosition(i)] for i in range(m.GetNumAtoms())]

    bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in m.GetBonds()]
    bonds = {tuple(sorted(b)): None for b in bonds}

    charge = Chem.GetFormalCharge(m) if charge is None else charge

    pymatgen_mol = pymatgen.core.structure.Molecule(species, coords, charge)
    mol_graph = MoleculeGraph.with_edges(pymatgen_mol, bonds)
    #mol_graph = with_edges(pymatgen_mol, bonds)

    if identifier is None:
        identifier = m.GetProp("_Name")
    mw = MoleculeWrapper(mol_graph, free_energy, identifier)
    mw.rdkit_mol = m

    return mw


def write_sdf_csv_dataset(
    molecules,
    struct_file="struct_mols.sdf",
    label_file="label_mols.csv",
    feature_file="feature_mols.yaml",
    exclude_single_atom=True,
):
    """
    Write the molecular atomization free energy to file.
    Args:
        molecules:
        struct_file:
        label_file:
        feature_file:
        exclude_single_atom:

    Returns:

    """
    struct_file = to_path(struct_file)
    label_file = to_path(label_file)

    logger.info(
        "Start writing dataset to files: {} and {}".format(struct_file, label_file)
    )

    feats = []

    with open(struct_file, "w") as fx, open(label_file, "w") as fy:
        fy.write("mol_id,atomization_energy\n")

        i = 0
        for m in molecules:
            if exclude_single_atom and m.num_atoms == 1:
                logger.info("Excluding single atom molecule {}".format(m.formula))
                continue

            sdf = m.write(name=m.id + "_index-" + str(i))
            fx.write(sdf)
            fy.write("{},{:.15g}\n".format(m.id, m.atomization_free_energy))

            feats.append(m.pack_features())
            i += 1

    # write feature file
    yaml_dump(feats, feature_file)


def write_edge_label_based_on_bond(
    molecules,
    sdf_filename="mols.sdf",
    label_filename="bond_label.yaml",
    feature_filename="feature.yaml",
    exclude_single_atom=True,
):
    """
    For a molecule from SDF file, creating complete graph for atoms and label the edges
    based on whether its an actual bond or not.

    The order of the edges are (0,1), (0,2), ... , (0, N-1), (1,2), (1,3), ...,
    (N-2, N-1), where N is the number of atoms.

    Args:
        molecules (list): a sequence of MoleculeWrapper object
        sdf_filename (str): name of the output sdf file
        label_filename (str): name of the output label file
        feature_filename (str): name of the output feature file
    """

    def get_bond_label(m):
        """
        Get to know whether an edge in a complete graph is a bond.

        Returns:
            list: bool to indicate whether an edge is a bond. The edges are in the order:
                (0,1), (0,2), ..., (0,N-1), (1,2), (1,3), ..., (N, N-1), where N is the
                number of atoms.
        """
        bonds = [b for b, attr in m.bonds.items()]
        num_bonds = len(bonds)
        if num_bonds < 1:
            warnings.warn("molecular has no bonds")

        bond_label = []
        for u, v in itertools.combinations(range(m.num_atoms), 2):
            b = tuple(sorted([u, v]))
            if b in bonds:
                bond_label.append(True)
            else:
                bond_label.append(False)

        return bond_label

    labels = []
    charges = []
    sdf_filename = to_path(sdf_filename)
    with open(sdf_filename, "w") as f:
        i = 0
        for m in molecules:
            if exclude_single_atom and m.num_atoms == 1:
                logger.info("Excluding single atom molecule {}".format(m.formula))
                continue

            sdf = m.write(name=m.id + " int_id-" + str(i))
            f.write(sdf)
            labels.append(get_bond_label(m))
            charges.append({"charge": m.charge})
            i += 1

    yaml_dump(labels, to_path(label_filename))
    yaml_dump(charges, to_path(feature_filename))


def fragment_mol_graph(mol_graph, bonds):
    """
    Break a bond in molecule graph and obtain the fragment(s).

    Args:
        mol_graph (MoleculeGraph): molecule graph to fragment
        bonds (list): bond indices (2-tuple)

    Returns:
        dict: with bond index (2-tuple) as key, and a list of fragments (mol_graphs)
            as values. Each list could be of size 1 or 2 and could be empty if the
            mol has no bonds.
    """
    sub_mols = {}

    for edge in bonds:
        edge = tuple(edge)
        try:
            new_mgs = mol_graph.split_molecule_subgraphs(
                [edge], allow_reverse=True, alterations=None
            )
            sub_mols[edge] = new_mgs
        except MolGraphSplitError:  # cannot split, (breaking a bond in a ring)
            new_mg = copy.deepcopy(mol_graph)
            idx1, idx2 = edge
            new_mg.break_edge(idx1, idx2, allow_reverse=True)
            sub_mols[edge] = [new_mg]
    return sub_mols


def order_two_molecules(m1, m2):
    """
    Order the molecules according to the below rules (in order):

    1. molecular mass
    2. number of atoms
    3. number of bonds
    4. alphabetical formula
    5. diameter of molecule graph, i.e. largest distance for node to node
    6. charge

    Args:
        m1, m2 : MoleculeWrapper

    Returns:
        A list of ordered molecules.
    """

    def compare(pa, pb, a, b):
        if pa < pb:
            return [a, b]
        elif pa > pb:
            return [b, a]
        else:
            return None

    def order_by_weight(a, b):
        pa = a.weight
        pb = b.weight
        return compare(pa, pb, a, b)

    def order_by_natoms(a, b):
        pa = a.num_atoms
        pb = b.num_atoms
        return compare(pa, pb, a, b)

    def order_by_nbonds(a, b):
        pa = len(a.bonds)
        pb = len(b.bonds)
        return compare(pa, pb, a, b)

    def order_by_formula(a, b):
        pa = a.formula
        pb = b.formula
        return compare(pa, pb, a, b)

    def order_by_diameter(a, b):
        try:
            pa = nx.diameter(a.graph)
        except nx.NetworkXError:
            pa = 100000000
        try:
            pb = nx.diameter(b.graph)
        except nx.NetworkXError:
            pb = 100000
        return compare(pa, pb, a, b)

    def order_by_charge(a, b):
        pa = a.charge
        pb = b.charge
        return compare(pa, pb, a, b)

    out = order_by_weight(m1, m2)
    if out is not None:
        return out
    out = order_by_natoms(m1, m2)
    if out is not None:
        return out
    out = order_by_nbonds(m1, m2)
    if out is not None:
        return out
    out = order_by_formula(m1, m2)
    if out is not None:
        return out
    out = order_by_diameter(m1, m2)
    if out is not None:
        return out

    if m1.mol_graph.isomorphic_to(m2.mol_graph):
        out = order_by_charge(m1, m2)  # e.g. H+ and H-
        if out is not None:
            return out
        else:
            return [m1, m2]  # two exactly the same molecules
    raise RuntimeError("Cannot order molecules")
