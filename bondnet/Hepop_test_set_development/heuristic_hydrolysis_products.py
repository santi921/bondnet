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

# Add S6, S7, S10, S11, S16, S17, S18, S19, S20, S23


def get_hydrolysis_products(fg_dict, rdkit_mol):
    # Add S6, S11, S19 no reactions recorded but might be important

    hydrolysed_products = {}
    for fg_s in fg_dict.items():
        ##aliphatic halogen block

        # Scheme S1-3 aliphatic chloride
        if fg_s[1] == "aliphatic chloride":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                edi_mol = Chem.EditableMol(rdkit_mol)
                edi_mol.RemoveBond(bp[0], bp[1])
                combo_mol = Chem.CombineMols(edi_mol.GetMol(), Chem.MolFromSmiles("O"))
                if at_1 == "C":
                    ab_1 = bp[0]
                else:
                    ab_1 = bp[1]

                ab_2 = len(combo_mol.GetAtoms()) - 1
                edcombo = Chem.EditableMol(combo_mol)
                edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                mod_mol = Chem.RemoveHs(edcombo.GetMol())
                mod_mol_al_cl = mod_mol

            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "aliphatic chloride" in hydrolysed_products:
                hydrolysed_products["aliphatic chloride"].append(pmg_frags)
            else:
                hydrolysed_products["aliphatic chloride"] = pmg_frags

        # Scheme S1-3 aliphatic bromide
        if fg_s[1] == "aliphatic bromide":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                edi_mol = Chem.EditableMol(rdkit_mol)
                edi_mol.RemoveBond(bp[0], bp[1])
                combo_mol = Chem.CombineMols(edi_mol.GetMol(), Chem.MolFromSmiles("O"))
                if at_1 == "C":
                    ab_1 = bp[0]
                else:
                    ab_1 = bp[1]

                ab_2 = len(combo_mol.GetAtoms()) - 1
                edcombo = Chem.EditableMol(combo_mol)
                edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                mod_mol = Chem.RemoveHs(edcombo.GetMol())

            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "aliphatic bromide" in hydrolysed_products:
                hydrolysed_products["aliphatic bromide"].append(pmg_frags)
            else:
                hydrolysed_products["aliphatic bromide"] = pmg_frags

        # Scheme S1-3 aliphatic fluoride
        if fg_s[1] == "aliphatic fluoride":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                edi_mol = Chem.EditableMol(rdkit_mol)
                edi_mol.RemoveBond(bp[0], bp[1])
                combo_mol = Chem.CombineMols(edi_mol.GetMol(), Chem.MolFromSmiles("O"))
                if at_1 == "C":
                    ab_1 = bp[0]
                else:
                    ab_1 = bp[1]

                ab_2 = len(combo_mol.GetAtoms()) - 1
                edcombo = Chem.EditableMol(combo_mol)
                edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                mod_mol = Chem.RemoveHs(edcombo.GetMol())

            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "aliphatic fluoride" in hydrolysed_products:
                hydrolysed_products["aliphatic fluoride"].append(pmg_frags)
            else:
                hydrolysed_products["aliphatic fluoride"] = pmg_frags

        # Scheme S1-3 aliphatic iodide
        if fg_s[1] == "aliphatic iodide":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                edi_mol = Chem.EditableMol(rdkit_mol)
                edi_mol.RemoveBond(bp[0], bp[1])
                combo_mol = Chem.CombineMols(edi_mol.GetMol(), Chem.MolFromSmiles("O"))
                if at_1 == "C":
                    ab_1 = bp[0]
                else:
                    ab_1 = bp[1]

                ab_2 = len(combo_mol.GetAtoms()) - 1
                edcombo = Chem.EditableMol(combo_mol)
                edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                mod_mol = Chem.RemoveHs(edcombo.GetMol())

            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "aliphatic iodide" in hydrolysed_products:
                hydrolysed_products["aliphatic iodide"].append(pmg_frags)
            else:
                hydrolysed_products["aliphatic iodide"] = pmg_frags

        ##Epoxide Block

        # Scheme S5

        # Scheme S14 (Aliphatic Amide)
        if fg_s[1] == "epoxide":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "O")
                    and (at_2 == "C" or at_2 == "O")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_am = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "epoxide" in hydrolysed_products:
                hydrolysed_products["epoxide"].append(pmg_frags)
            else:
                hydrolysed_products["epoxide"] = pmg_frags
        #         if (fg_s[1] == 'epoxide'):
        #             pmg_frags = []
        #             mod_mol = Chem.ReplaceSubstructs(rdkit_mol, Chem.MolFromSmarts('C1OC1'),
        #                                              Chem.MolFromSmarts('C(O)C(O)'),
        #                                              replaceAll=True)
        #             if (type(mod_mol) == tuple):
        #                 hydro_mol = mod_mol[0]
        #             hydro_mol_sm = Chem.MolToSmiles(hydro_mol,True)
        #             pb_mol = pb.readstring("smi", hydro_mol_sm)
        #             pb_mol.addh()
        #             pb_mol.make3D()
        #             pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
        #             pmg_frags.append(pmg_mol)

        #             if 'epoxide' in hydrolysed_products:
        #                 hydrolysed_products['epoxide'].append(pmg_frags)
        #             else:
        #                 hydrolysed_products['epoxide'] = pmg_frags

        # #         if (fg_s[1] == 'epoxide'):
        # #             pmg_frags = []
        # #             poss_bond_pairs = [(a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1:]]
        # #             print(poss_bond_pairs)
        # #             for bp in poss_bond_pairs:
        # #                 at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
        # #                 at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
        # #                 if (rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None):
        # #                     b_12 = 0
        # #                 else:
        # #                     b_12 = rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]).GetBondTypeAsDouble()

        # #                 if (
        # #                     (at_1 == 'C' or at_1 == 'O') and
        # #                     (at_2 == 'C' or at_2 == 'O') and
        # #                     (b_12 == 1.0)
        # #                 ):
        # #                     edi_mol = Chem.EditableMol(rdkit_mol)
        # #                     edi_mol.RemoveBond(bp[0],bp[1])
        # #                     combo_mol = Chem.CombineMols(edi_mol.GetMol(),Chem.MolFromSmiles('O'))
        # #                     if (at_1 == 'C'):
        # #                         ab_1 = bp[0]
        # #                     else:
        # #                         ab_1 = bp[1]

        # #                     ab_2 = len(combo_mol.GetAtoms()) - 1
        # #                     edcombo = Chem.EditableMol(combo_mol)
        # #                     edcombo.AddBond(ab_1,ab_2,order=Chem.rdchem.BondType.SINGLE)
        # #                     mod_mol = Chem.RemoveHs(edcombo.GetMol())
        # #                     mod_mol_epo = mod_mol
        # #                     break
        # #             frags = Chem.GetMolFrags(mod_mol,asMols=True)
        # #             pmg_frags = []
        # #             for x in frags:
        # #                 a = Chem.MolToSmiles(x,True)
        # #                 pb_mol = pb.readstring("smi", a)
        # #                 pb_mol.addh()
        # #                 pb_mol.make3D()
        # #                 pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
        # #                 pmg_frags.append(pmg_mol)

        #             if 'epoxide' in hydrolysed_products:
        #                 hydrolysed_products['epoxide'].append(pmg_frags)
        #             else:
        #                 hydrolysed_products['epoxide'] = pmg_frags

        ##Organophosphorus Ester Hydrolysis Block

        # Scheme S7 (opo_oxygen)
        if fg_s[1] == "organophosphorus ester with oxygen":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "P" or at_1 == "O")
                    and (at_2 == "P" or at_2 == "O")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "P":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_opo = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "organophosphorus ester with oxygen" in hydrolysed_products:
                hydrolysed_products["op_es_o"].append(pmg_frags)
            else:
                hydrolysed_products["op_es_o"] = pmg_frags

        # Scheme S7 (opo_sulphur)
        if fg_s[1] == "organophosphorus ester with sulphur":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "P" or at_1 == "S")
                    and (at_2 == "P" or at_2 == "S")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "P":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_ops = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "organophosphorus ester with sulphur" in hydrolysed_products:
                hydrolysed_products["op_es_s"].append(pmg_frags)
            else:
                hydrolysed_products["op_es_s"] = pmg_frags

        ##Aliphatic and cyclic Carboxylic acid ester block

        # Scheme S8 (Aliphatic Carboxylic acid ester)
        if fg_s[1] == "carboxylic acid ester":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "O")
                    and (at_2 == "C" or at_2 == "O")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_es = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "carboxylic acid ester" in hydrolysed_products:
                hydrolysed_products["carboxylic acid ester"].append(pmg_frags)
            else:
                hydrolysed_products["carboxylic acid ester"] = pmg_frags

        # Scheme S9 (Cyclic Carboxylic acid ester/lactone)
        if fg_s[1] == "lactone":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "O")
                    and (at_2 == "C" or at_2 == "O")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_la = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "lactone" in hydrolysed_products:
                hydrolysed_products["lactone"].append(pmg_frags)
            else:
                hydrolysed_products["lactone"] = pmg_frags

        ##Aliphatic and cyclic carbonate block

        # Scheme S10 (Aliphatic Carbonate)
        if fg_s[1] == "carbonate":
            flag = 0
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "O")
                    and (at_2 == "C" or at_2 == "O")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif (
                    (at_1 == "C" or at_1 == "O")
                    and (at_2 == "C" or at_2 == "O")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif at_1 == "C" and at_2 == "O" and b_12 == 2.0:
                    ab_1 = bp[0]

                elif at_1 == "O" and at_2 == "C" and b_12 == 2.0:
                    ab_1 = bp[1]

            combo_mol = Chem.CombineMols(edi_mol.GetMol(), Chem.MolFromSmiles("O"))
            ab_2 = len(combo_mol.GetAtoms()) - 1
            edcombo = Chem.EditableMol(combo_mol)
            edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.DOUBLE)
            mod_mol = Chem.RemoveHs(edcombo.GetMol())
            mod_mol_car = mod_mol

            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "carbonate" in hydrolysed_products:
                hydrolysed_products["carbonate"].append(pmg_frags)
            else:
                hydrolysed_products["carbonate"] = pmg_frags

        # Scheme S11 (Cyclic Carbonate)
        if fg_s[1] == "cyclic carbonate":
            flag = 0
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "O")
                    and (at_2 == "C" or at_2 == "O")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif (
                    (at_1 == "C" or at_1 == "O")
                    and (at_2 == "C" or at_2 == "O")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif at_1 == "C" and at_2 == "O" and b_12 == 2.0:
                    ab_1 = bp[0]

                elif at_1 == "O" and at_2 == "C" and b_12 == 2.0:
                    ab_1 = bp[1]

            combo_mol = Chem.CombineMols(edi_mol.GetMol(), Chem.MolFromSmiles("O"))
            ab_2 = len(combo_mol.GetAtoms()) - 1
            edcombo = Chem.EditableMol(combo_mol)
            edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.DOUBLE)
            mod_mol = Chem.RemoveHs(edcombo.GetMol())
            mod_mol_car = mod_mol

            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "cyclic carbonate" in hydrolysed_products:
                hydrolysed_products["cyclic carbonate"].append(pmg_frags)
            else:
                hydrolysed_products["cyclic carbonate"] = pmg_frags

        ##Aliphatic and cyclic anhydride block

        # Scheme S12 (Aliphatic anhydride)
        if fg_s[1] == "anhydride":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "O")
                    and (at_2 == "C" or at_2 == "O")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_ah = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "anhydride" in hydrolysed_products:
                hydrolysed_products["anhydride"].append(pmg_frags)
            else:
                hydrolysed_products["anhydride"] = pmg_frags

        # Scheme S13 (Cyclic anhydride)
        if fg_s[1] == "cyclic anhydride":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "O")
                    and (at_2 == "C" or at_2 == "O")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_cah = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "cyclic anhydride" in hydrolysed_products:
                hydrolysed_products["cyclic anhydride"].append(pmg_frags)
            else:
                hydrolysed_products["cyclic anhydride"] = pmg_frags

        ##Aliphatic and cyclic amide block

        # Scheme S14 (Aliphatic Amide)
        if fg_s[1] == "amide":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "N")
                    and (at_2 == "C" or at_2 == "N")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_am = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "amide" in hydrolysed_products:
                hydrolysed_products["amide"].append(pmg_frags)
            else:
                hydrolysed_products["amide"] = pmg_frags

        # Scheme S15 (Cyclic Amide/Lactam)
        if fg_s[1] == "lactam":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "N")
                    and (at_2 == "C" or at_2 == "N")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_tam = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "lactam" in hydrolysed_products:
                hydrolysed_products["lactam"].append(pmg_frags)
            else:
                hydrolysed_products["lactam"] = pmg_frags

        ##Carbamate Block
        # Scheme S16
        if fg_s[1] == "carbamate":
            flag = 0
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "N")
                    and (at_2 == "C" or at_2 == "N")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif (
                    (at_1 == "C" or at_1 == "O")
                    and (at_2 == "C" or at_2 == "O")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif at_1 == "C" and at_2 == "O" and b_12 == 2.0:
                    ab_1 = bp[0]

                elif at_1 == "O" and at_2 == "C" and b_12 == 2.0:
                    ab_1 = bp[1]

            combo_mol = Chem.CombineMols(edi_mol.GetMol(), Chem.MolFromSmiles("O"))
            ab_2 = len(combo_mol.GetAtoms()) - 1
            edcombo = Chem.EditableMol(combo_mol)
            edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.DOUBLE)
            mod_mol = Chem.RemoveHs(edcombo.GetMol())
            mod_mol_cam = mod_mol

            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "carbamate" in hydrolysed_products:
                hydrolysed_products["carbamate"].append(pmg_frags)
            else:
                hydrolysed_products["carbamate"] = pmg_frags

        ##Thiocarbamate Block
        # Scheme S17
        if fg_s[1] == "thiocarbamate":
            flag = 0
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "N")
                    and (at_2 == "C" or at_2 == "N")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif (
                    (at_1 == "C" or at_1 == "S")
                    and (at_2 == "C" or at_2 == "S")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif at_1 == "C" and at_2 == "O" and b_12 == 2.0:
                    ab_1 = bp[0]

                elif at_1 == "O" and at_2 == "C" and b_12 == 2.0:
                    ab_1 = bp[1]

            combo_mol = Chem.CombineMols(edi_mol.GetMol(), Chem.MolFromSmiles("O"))
            ab_2 = len(combo_mol.GetAtoms()) - 1
            edcombo = Chem.EditableMol(combo_mol)
            edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.DOUBLE)
            mod_mol = Chem.RemoveHs(edcombo.GetMol())
            mod_mol_tcam = mod_mol

            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "thiocarbamate" in hydrolysed_products:
                hydrolysed_products["thiocarbamate"].append(pmg_frags)
            else:
                hydrolysed_products["thiocarbamate"] = pmg_frags

        ##Aliphatic and Cylic Urea Block
        # Scheme S18 (Aliphatic Urea)
        if fg_s[1] == "urea":
            flag = 0
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "N")
                    and (at_2 == "C" or at_2 == "N")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif (
                    (at_1 == "C" or at_1 == "N")
                    and (at_2 == "C" or at_2 == "N")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif at_1 == "C" and at_2 == "O" and b_12 == 2.0:
                    ab_1 = bp[0]

                elif at_1 == "O" and at_2 == "C" and b_12 == 2.0:
                    ab_1 = bp[1]

            combo_mol = Chem.CombineMols(edi_mol.GetMol(), Chem.MolFromSmiles("O"))
            ab_2 = len(combo_mol.GetAtoms()) - 1
            edcombo = Chem.EditableMol(combo_mol)
            edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.DOUBLE)
            mod_mol = Chem.RemoveHs(edcombo.GetMol())
            mod_mol_ur = mod_mol

            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "urea" in hydrolysed_products:
                hydrolysed_products["urea"].append(pmg_frags)
            else:
                hydrolysed_products["urea"] = pmg_frags

        ##Sulfonylurea Block
        # Scheme S20
        if fg_s[1] == "sulfonylurea":
            flag = 0
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "N")
                    and (at_2 == "C" or at_2 == "N")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif (
                    (at_1 == "C" or at_1 == "N")
                    and (at_2 == "C" or at_2 == "N")
                    and (b_12 == 1.0)
                ):
                    if flag == 0:
                        edi_mol = Chem.EditableMol(rdkit_mol)
                        edi_mol.RemoveBond(bp[0], bp[1])
                        inter_mol = edi_mol
                        flag = 1
                    else:
                        edi_mol = Chem.EditableMol(inter_mol.GetMol())
                        edi_mol.RemoveBond(bp[0], bp[1])

                elif at_1 == "C" and at_2 == "O" and b_12 == 2.0:
                    ab_1 = bp[0]

                elif at_1 == "O" and at_2 == "C" and b_12 == 2.0:
                    ab_1 = bp[1]

            combo_mol = Chem.CombineMols(edi_mol.GetMol(), Chem.MolFromSmiles("O"))
            ab_2 = len(combo_mol.GetAtoms()) - 1
            edcombo = Chem.EditableMol(combo_mol)
            edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.DOUBLE)
            mod_mol = Chem.RemoveHs(edcombo.GetMol())
            mod_mol_sur = mod_mol

            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "sulfonylurea" in hydrolysed_products:
                hydrolysed_products["sulfonylurea"].append(pmg_frags)
            else:
                hydrolysed_products["sulfonylurea"] = pmg_frags

        ##Nitrile Block

        # Scheme S21
        if fg_s[1] == "nitrile":
            pmg_frags = []
            mod_mol = Chem.ReplaceSubstructs(
                rdkit_mol,
                Chem.MolFromSmiles("C#N"),
                Chem.MolFromSmiles("C(=O)N"),
                replaceAll=True,
            )
            if type(mod_mol) == tuple:
                hydro_mol = Chem.RemoveHs(mod_mol[0])

            hydro_mol_sm = Chem.MolToSmiles(hydro_mol, True)
            pb_mol = pb.readstring("smi", hydro_mol_sm)
            pb_mol.addh()
            pb_mol.make3D()
            pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
            pmg_frags.append(pmg_mol)

            if "nitrile" in hydrolysed_products:
                hydrolysed_products["nitrile"].append(pmg_frags)
            else:
                hydrolysed_products["nitrile"] = pmg_frags

        ##N-S cleavage block

        # Scheme S22
        if fg_s[1] == "Nitrogen-Sulphur":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "N" or at_1 == "S")
                    and (at_2 == "N" or at_2 == "S")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "S":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_ns = mod_mol
                    break

            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "Nitrogen-Sulphur" in hydrolysed_products:
                hydrolysed_products["Nitrogen-Sulphur"].append(pmg_frags)
            else:
                hydrolysed_products["Nitrogen-Sulphur"] = pmg_frags

        # Scheme S23 (Imide)
        if fg_s[1] == "imide":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "N")
                    and (at_2 == "C" or at_2 == "N")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_im = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "imide" in hydrolysed_products:
                hydrolysed_products["imide"].append(pmg_frags)
            else:
                hydrolysed_products["imide"] = pmg_frags

        ##Acid Halide block

        # Scheme S24 acid chloride
        if fg_s[1] == "acid chloride":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "Cl")
                    and (at_2 == "C" or at_2 == "Cl")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_cl = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "acid chloride" in hydrolysed_products:
                hydrolysed_products["acid chloride"].append(pmg_frags)
            else:
                hydrolysed_products["acid chloride"] = pmg_frags

        # Scheme S24 acid bromide
        if fg_s[1] == "acid bromide":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "Br")
                    and (at_2 == "C" or at_2 == "Br")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_br = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "acid bromide" in hydrolysed_products:
                hydrolysed_products["acid bromide"].append(pmg_frags)
            else:
                hydrolysed_products["acid bromide"] = pmg_frags

        # Scheme S24 acid fluoride
        if fg_s[1] == "acid fluoride":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "F")
                    and (at_2 == "C" or at_2 == "F")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_fl = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "acid fluoride" in hydrolysed_products:
                hydrolysed_products["acid fluoride"].append(pmg_frags)
            else:
                hydrolysed_products["acid fluoride"] = pmg_frags

        # Scheme S24 acid fluoride
        if fg_s[1] == "acid iodide":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                if (
                    (at_1 == "C" or at_1 == "I")
                    and (at_2 == "C" or at_2 == "I")
                    and (b_12 == 1.0)
                ):
                    edi_mol = Chem.EditableMol(rdkit_mol)
                    edi_mol.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mol_mol_io = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "acid iodide" in hydrolysed_products:
                hydrolysed_products["acid iodide"].append(pmg_frags)
            else:
                hydrolysed_products["acid iodide"] = pmg_frags

        ##PDK depolymerization block

        if fg_s[1] == "PDK":
            fg_idx = list(fg_s[0])
            poss_bond_pairs = [
                (a, b) for idx, a in enumerate(fg_idx) for b in fg_idx[idx + 1 :]
            ]
            for bp in poss_bond_pairs:
                at_1 = rdkit_mol.GetAtomWithIdx(bp[0]).GetSymbol()
                at_2 = rdkit_mol.GetAtomWithIdx(bp[1]).GetSymbol()
                if rdkit_mol.GetBondBetweenAtoms(bp[0], bp[1]) == None:
                    b_12 = 0
                else:
                    b_12 = rdkit_mol.GetBondBetweenAtoms(
                        bp[0], bp[1]
                    ).GetBondTypeAsDouble()

                # if (
                #     (at_1 == 'C' or at_1 == 'C') and
                #     (at_2 == 'C' or at_2 == 'C') and
                #     (b_12 == 2.0)
                # ):
                #     edi_mol = Chem.EditableMol(rdkit_mol)
                #     edi_mol.RemoveBond(bp[0],bp[1])
                #     edi_mol.AddBond(bp[0],bp[1],order=Chem.rdchem.BondType.SINGLE)

                if (
                    (at_1 == "C" or at_1 == "N")
                    and (at_2 == "C" or at_2 == "N")
                    and (b_12 == 1.0)
                ):
                    # edi_mol_2 = Chem.EditableMol(edi_mol.GetMol())
                    edi_mol_2 = Chem.EditableMol(rdkit_mol)
                    edi_mol_2.RemoveBond(bp[0], bp[1])
                    combo_mol = Chem.CombineMols(
                        edi_mol_2.GetMol(), Chem.MolFromSmiles("O")
                    )
                    if at_1 == "C":
                        ab_1 = bp[0]
                    else:
                        ab_1 = bp[1]

                    ab_2 = len(combo_mol.GetAtoms()) - 1
                    edcombo = Chem.EditableMol(combo_mol)
                    edcombo.AddBond(ab_1, ab_2, order=Chem.rdchem.BondType.SINGLE)
                    mod_mol = Chem.RemoveHs(edcombo.GetMol())
                    mod_mol_pdk = mod_mol
                    break
            frags = Chem.GetMolFrags(mod_mol, asMols=True)
            pmg_frags = []
            for x in frags:
                a = Chem.MolToSmiles(x, True)
                pb_mol = pb.readstring("smi", a)
                pb_mol.addh()
                pb_mol.make3D()
                pmg_mol = BabelMolAdaptor(pb_mol.OBMol).pymatgen_mol
                pmg_frags.append(pmg_mol)

            if "PDK" in hydrolysed_products:
                hydrolysed_products["PDK"].append(pmg_frags)
            else:
                hydrolysed_products["PDK"] = pmg_frags

    return hydrolysed_products
