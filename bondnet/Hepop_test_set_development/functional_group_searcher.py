import pymatgen.core
import sys
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.functional_groups import FunctionalGroupExtractor
import openbabel as ob
from openbabel import pybel as pb
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

IPythonConsole.ipython_useSVG = (
    False  # < set this to False if you want PNGs instead of SVGs
)
IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 900, 900
from IPython.core.display import SVG, display_svg

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

al_cl_ss = Chem.MolFromSmarts("CCl")  # Scheme S1-3
al_f_ss = Chem.MolFromSmarts("CF")  # Scheme S1-3
al_br_ss = Chem.MolFromSmarts("CBr")  # Scheme S1-3
al_i_ss = Chem.MolFromSmarts("CI")  # Scheme S1-3
epo_ss = Chem.MolFromSmarts("C1OC1")  # Scheme S5
op_o_ss = Chem.MolFromSmarts("P(=O)O")  # Scheme S6-7
op_s_ss = Chem.MolFromSmarts("P(=S)O")  # Scheme S6-7
carboxyl_ester_ss = Chem.MolFromSmarts("[C;!R](=O)[O;!R]")  # Scheme S8
lactone_ss = Chem.MolFromSmarts("[C;R](=O)[O;R]")  # Scheme S9
carbonate_ss = Chem.MolFromSmarts("[O;!R][C;!R](=O)[O;!R]")  # Scheme S10
cyclic_carbonate_ss = Chem.MolFromSmarts("O@C(=O)@O")  # Scheme S11
anhydride_ss = Chem.MolFromSmarts("[C;!R](=O)[O;!R][C;!R](=O)")  # Scheme S12
cyclic_anhydride_ss = Chem.MolFromSmarts("C(=O)@O@C(=O)")  # Scheme S13
amide_ss = Chem.MolFromSmarts("[C;!R](=O)[N;!R]")  # Scheme S14
lactam_ss = Chem.MolFromSmarts("C(=O)@N")  # Scheme S15
carbamate_ss = Chem.MolFromSmarts("NC(=O)O")  # Scheme S16
thiocarbamate_ss = Chem.MolFromSmarts("NC(=O)S")  # Scheme S17
urea_ss = Chem.MolFromSmarts("NC(=O)N")  # Scheme S18
sulfonylurea_ss = Chem.MolFromSmarts("S(=O)(=O)NC(=O)N")  # Scheme S20
nitrile_ss = Chem.MolFromSmarts("C#N")  # Scheme S21
NS_ss = Chem.MolFromSmarts("NS")  # Scheme S22
imide_ss = Chem.MolFromSmarts("C(=O)@N@C(=O)")  # Scheme S23
acid_cl_ss = Chem.MolFromSmarts("C(=O)Cl")  # Scheme S24
acid_br_ss = Chem.MolFromSmarts("C(=O)Br")  # Scheme S24
acid_f_ss = Chem.MolFromSmarts("C(=O)F")  # Scheme S24
acid_i_ss = Chem.MolFromSmarts("C(=O)I")  # Scheme S24
PDK = Chem.MolFromSmarts("C=CN")  # For PDK polymers

ss_list = [
    al_cl_ss,
    al_f_ss,
    al_br_ss,
    al_i_ss,
    epo_ss,
    op_o_ss,
    op_s_ss,
    carboxyl_ester_ss,
    lactone_ss,
    carbonate_ss,
    cyclic_carbonate_ss,
    anhydride_ss,
    cyclic_anhydride_ss,
    amide_ss,
    lactam_ss,
    carbamate_ss,
    thiocarbamate_ss,
    urea_ss,
    sulfonylurea_ss,
    nitrile_ss,
    NS_ss,
    imide_ss,
    acid_cl_ss,
    acid_br_ss,
    acid_f_ss,
    acid_i_ss,
    PDK,
]

ss_label = [
    "aliphatic chloride",
    "aliphatic fluoride",
    "aliphatic bromide",
    "aliphatic iodide",
    "epoxide",
    "organophosphorus ester with oxygen",
    "organophosphorus ester with sulphur",
    "carboxylic acid ester",
    "lactone",
    "carbonate",
    "cyclic carbonate",
    "anhydride",
    "cyclic anhydride",
    "amide",
    "lactam",
    "carbamate",
    "thiocarbamate",
    "urea",
    "sulfonylurea",
    "nitrile",
    "Nitrogen-Sulphur",
    "imide",
    "acid chloride",
    "acid bromide",
    "acid fluoride",
    "acid iodide",
    "PDK",
]


def get_smiles_of_pmg_mol(pmg_mol):
    bma_obj = BabelMolAdaptor(pmg_mol)
    pb_obj = pb.Molecule(bma_obj.openbabel_mol)
    smi_obj = pb.readstring("can", pb_obj.write("can"))
    # print("Canonical SMILES = {}".format(smi_obj.write("can")))

    return smi_obj.write("can")


def get_fg_dict(rdkit_mol):
    atoms = set(list(range(rdkit_mol.GetNumAtoms())))

    fg_i_t = []
    fg_n_t = []
    for ss in ss_list:
        ss_class = ss_label[ss_list.index(ss)]

        if rdkit_mol.GetSubstructMatch(ss):
            sub_ss = list(rdkit_mol.GetSubstructMatches(ss))
            for sub in sub_ss:
                fg_i_t.append(sub)
                fg_n_t.append(ss_label[ss_list.index(ss)])
                # print('Molecule contains substructure', ss_label[ss_list.index(ss)], 'at', sub)

    fg_idx_list = []
    fg_name_list = []
    pl_hl_i = []
    pl_hl_n = []

    for fg_s in zip(fg_i_t, fg_n_t):
        flg = 0
        for x in zip(fg_i_t, fg_n_t):
            if (len(set(x[0])) > len(set(fg_s[0]))) and set(x[0]).issuperset(
                set(fg_s[0])
            ):
                # print ('Functional group', x[1], 'at', x[0], 'is a superset of functional group', fg_s[1], 'at', fg_s[0])
                pl_hl_i.append(x[0])
                pl_hl_n.append(x[1])
                flg = 1
        if flg == 0:
            fg_idx_list.append(fg_s[0])
            fg_name_list.append(fg_s[1])
        else:
            max_tup = len(max(pl_hl_i, key=len))
            for rfg in zip(pl_hl_i, pl_hl_n):
                if len(rfg[0]) == max_tup:
                    fg_idx_list.append(rfg[0])
                    fg_name_list.append(rfg[1])
                    break
    fg_dict = {}
    for fgs in zip(fg_idx_list, fg_name_list):
        if fgs[0] not in fg_dict.keys():
            fg_dict[fgs[0]] = fgs[1]

    return fg_dict
