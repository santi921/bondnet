import pandas as pd 
import copy
import pickle
import itertools
import csv
import json

import numpy as np
import networkx as nx

from monty.serialization import loadfn, dumpfn

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from emmet.core.utils import get_graph_hash, get_molecule_id



def parse_molecule_graphs(rxn): 
    """
    Helper to parse MoleculeGraphs from json row
    """

    
    rct_graph_metal = MoleculeGraph.from_dict(rxn["reactant_molecule_graph"])
    spin = rxn["reactant_molecule_graph"]["molecule"]["spin_multiplicity"]
    charge = rxn["reactant_molecule_graph"]["molecule"]["charge"]
    rct_graph_metal.molecule.set_charge_and_spin(charge=charge, spin_multiplicity=spin)

    rct_graph_nometal = MoleculeGraph.from_dict(rxn["reactant_molecule_graph_nometal"])
    spin = rxn["reactant_molecule_graph_nometal"]["molecule"]["spin_multiplicity"]
    charge = rxn["reactant_molecule_graph_nometal"]["molecule"]["charge"]
    rct_graph_nometal.molecule.set_charge_and_spin(charge=charge, spin_multiplicity=spin)

    pro_graph_metal = MoleculeGraph.from_dict(rxn["product_molecule_graph"])
    spin = rxn["product_molecule_graph"]["molecule"]["spin_multiplicity"]
    charge = rxn["product_molecule_graph"]["molecule"]["charge"]
    pro_graph_metal.molecule.set_charge_and_spin(charge=charge, spin_multiplicity=spin)

    pro_graph_nometal = MoleculeGraph.from_dict(rxn["product_molecule_graph_nometal"])
    spin = rxn["product_molecule_graph_nometal"]["molecule"]["spin_multiplicity"]
    charge = rxn["product_molecule_graph_nometal"]["molecule"]["charge"]
    pro_graph_nometal.molecule.set_charge_and_spin(charge=charge, spin_multiplicity=spin)

    
    charge_rct = rxn["reactant_charges"]
    charge_pro = rxn["product_charges"]
    charge_total = rxn["charge"]

    return charge_rct, charge_pro, rct_graph_metal, rct_graph_nometal, pro_graph_metal, pro_graph_nometal, charge_total


def pull_ownership(subgs, metal_inds, metal_bonds):
    """
        Finds sgs which contain Metals. 
        Takes: 
            subgs - list of subgraph inds 
            li_inds - lithium atom indices
            metal_inds - list of lithium-containing bonds
        Returns: 
            list of sets s.t. each item is (li_ind, [sub_graph_ind]) i.e what sgs have a give li
    """

    metal_ownership = dict()
    ret_subgs = copy.deepcopy(subgs)
    for metal_ind in metal_inds: # list of Metals nodes
        metal_ownership[metal_ind] = set() # 
        for ii, subg in enumerate(ret_subgs): # list of sets of sg nodes
            if (any([(metal_ind, x) in metal_bonds for x in subg]) or any([(x, metal_ind) in metal_bonds for x in subg])):
                metal_ownership[metal_ind].add(ii)
    
    kvs = list(metal_ownership.items())
    #print("kvs: ", kvs)
    # if there is only one owner then there is no ownership issue
    for metal_ind, owners in kvs: 
        if len(owners) == 1:
            #print(list(owners)[0], metal_ind)
            ret_subgs[list(owners)[0]].add(metal_ind)
            del metal_ownership[metal_ind]
    
    return metal_ownership, ret_subgs


def filter_metal_bonds_from_graph(g, metal_list = ["Li", "Ca", "Mg", "Na"]):
    """
    Filters out all the metal bonds from the graph
    """

    metal_bonds = []
    metal_inds = []
    edges = list(g.edges())
    
    for metal in metal_list:
        metal_inds += [i for i, x in enumerate(g.nodes) if g.nodes[i]["specie"] == metal]
        metal_bonds_temp  = [e for e in edges if any([x in e for x in metal_inds])]
        metal_bonds += metal_bonds_temp

        for bond in metal_bonds_temp: 
            # check if the bond is in the graph
            if bond in g.edges():
                g.remove_edge(bond[0], bond[1])
            
    return g, metal_bonds, metal_inds





#if __name__ == "__main__":

train_loc = "./full_rapter_filtered_species.pkl"
train_df = pd.read_pickle(train_loc)

to_run = list()
to_run_id = list()
to_run_hash = list()

to_run_charge_exhaustive = list()
to_run_id_charge_exhaustive = list()
to_run_hash_charge_exhaustive = list()

to_run_full = list()
to_run_id_full = list()
to_run_hash_full = list()


mol_list = list()
mol_list_exhaustive = list()
mol_list_full = list()

atoms_not_allocated_count = 0
charge_none_count = 0
charge_not_equal_count = 0
count_single_sg = 0 
no_inf_sep_count = 0
backup_charge_set = [-2, -1, 0, 1, 2]

for name, rxn in train_df.iterrows():
    (
        charge_rct, 
        charge_pro, 
        rct_graph_metal, 
        rct_graph_nometal, 
        pro_graph_metal, 
        pro_graph_nometal,
        charge_total
    ) = parse_molecule_graphs(rxn)

    for (mtype, no_metal_mol, metal_mol, charge_list, charge_total) in [("rct", rct_graph_nometal, rct_graph_metal, charge_rct, charge_total), 
                                        ("pro", pro_graph_nometal, pro_graph_metal, charge_pro, charge_total)]:
        
        g = metal_mol.graph.to_undirected()
        g_nometal = no_metal_mol.graph.to_undirected()
        
        len_g = len(g)
        len_g_nometal = len(g_nometal)

        g, metal_bonds, metal_inds = filter_metal_bonds_from_graph(g)


        # this still adds metals that are disconnected
        subgs_prefilter = [x for x in nx.connected_components(g)]
        subgs = []
        for i in subgs_prefilter: 
            if len(i) > 1:
                subgs.append(i) 
            else: 
                if str(metal_mol.molecule.species[list(i)[0]]) not in ["Li", "Mg", "Ca", "Na"]:
                    subgs.append(i) 

        if len(subgs) == 1:
            molcopy = copy.deepcopy(metal_mol.molecule)
            molcopy_mee = metal_edge_extender(MoleculeGraph.with_local_env_strategy(molcopy, OpenBabelNN()))
            this_graph_hash = get_graph_hash(molcopy_mee.molecule, node_attr="specie") 
            this_mole_id = get_molecule_id(molcopy_mee.molecule, node_attr="specie")
            
            if this_graph_hash not in to_run_hash_full:
                match = False

            if not match:
                to_run_full.append(molcopy_mee)
                to_run_id_full.append(this_mole_id)
                to_run_hash_full.append(this_graph_hash)
                mol_list_full.append(molcopy_mee.molecule)

                to_run_charge_exhaustive.append(molcopy_mee)
                to_run_id_charge_exhaustive.append(this_mole_id)
                to_run_hash_charge_exhaustive.append(this_graph_hash)
                mol_list_exhaustive.append(molcopy_mee.molecule)
                
                count_single_sg += 1


        if metal_inds != []:
            metal_ownership, subgs = pull_ownership(subgs, metal_inds, metal_bonds)
            relevant_list_metals = [x for x in metal_inds if x in metal_ownership]
            ownership_distributions = list(itertools.product(*[metal_ownership[x] for x in relevant_list_metals]))
            if ownership_distributions==[]: ownership_distributions=[()]
            #print(ownership_distributions)
        else: 
            ownership_distributions = [()]
            relevant_list_metals = []

        some_valid_dist = False
        enter_dist = False

        for dist in ownership_distributions:
            enter_dist = True
            # create and distribute metals among this subgraph distribution
            sgs_copy = [copy.deepcopy(x) for x in subgs]
            
            # add metals to sgs for this distribution
            if dist != ():
                for ii, owner in enumerate(dist):
                    sgs_copy[owner].add(relevant_list_metals[ii]) 

            
            if sum([len(x) for x in sgs_copy]) != len(metal_mol):
                atoms_not_allocated_count += 1
                #print("PROBLEM: NOT ALL ATOMS ALLOCATED - {} {} {} {} {} {} {}".format(name, mtype, dist, sum([len(x) for x in sgs_copy]), len(no_metal_mol), len_g, len_g_nometal))
                continue

            charges = list()
            #print(charge_list)
            # check that there are no None values in charge_list
            if None in charge_list:
                #print("PROBLEM: CHARGE LIST CONTAINS NONE - {} {} {} {}".format(name, mtype, dist, charge_list))
                charge_none_count += 1
                num_subgraphs = len(sgs_copy)
                charge_comb_list_raw = list(itertools.product(backup_charge_set, repeat=num_subgraphs))
                charge_comb_list = []
                # generate all possible charge combinations for the subgraphs s.t. the sum is equal to the total charge
                for charge_comb in itertools.product(backup_charge_set, repeat=num_subgraphs):
                    if sum(charge_comb) == metal_mol.molecule.charge:
                        charge_comb_list.append(charge_comb)

                for charge_comb in charge_comb_list:
                    molcopy = copy.deepcopy(metal_mol.molecule)
                    for ii, sg in enumerate(sgs_copy):
                        molcopy.remove_sites([i for i in range(len(metal_mol)) if i not in sg])
                        molcopy.set_charge_and_spin(charge_comb[ii])
                    
                    this_mg = metal_edge_extender(MoleculeGraph.with_local_env_strategy(molcopy, OpenBabelNN()))
                    match = False
                    for mg in to_run_full:
                        if metal_mol.isomorphic_to(this_mg) and metal_mol.molecule.charge == charge_comb[ii]:
                            match = True
                            break

                    if not match:
                        this_graph_hash = get_graph_hash(this_mg.molecule, node_attr="specie") 
                        this_mole_id = get_molecule_id(this_mg.molecule, node_attr="specie")
                        to_run_full.append(this_mg)
                        to_run_id_full.append(this_mole_id)
                        to_run_hash_full.append(this_graph_hash)
                        mol_list_full.append(this_mg.molecule)

                        to_run_charge_exhaustive.append(this_mg)
                        to_run_id_charge_exhaustive.append(this_mole_id)
                        to_run_hash_charge_exhaustive.append(this_graph_hash)
                        mol_list_exhaustive.append(this_mg.molecule)
                

            else:
                for sg in sgs_copy:
                    sgcharge = 0
                    for atom in sg: 
                        sgcharge += charge_list[atom]
                    charges.append(int(round(sgcharge)))

                if sum(charges) != metal_mol.molecule.charge:
                #    #print("PROBLEM: DIST CHARGE {} NOT EQUAL TO MOL CHARGE {}".format(sum(charges), metal_mol.molecule.charge), name, mtype, dist)
                    charge_not_equal_count += 1
                    continue

                # If we get to this point, we're alright
                some_valid_dist = True
                
                # a dist for charge and atoms
                for ii, sg in enumerate(sgs_copy):
                    molcopy = copy.deepcopy(metal_mol.molecule)
                    # filter to only valid SG
                    molcopy.remove_sites([i for i in range(len(metal_mol)) if i not in sg])
                    # set charge and spin to determined values
                    molcopy.set_charge_and_spin(charges[ii])
                    # add bonds with metal as a guess
                    this_mg = metal_edge_extender(MoleculeGraph.with_local_env_strategy(molcopy, OpenBabelNN()))

                    match = False
                    for mg in to_run: # checks if it's already in the list
                        # basically if the sg is the same as the whole graph
                        if metal_mol.isomorphic_to(this_mg) and metal_mol.molecule.charge == charges[ii]:
                            match = True
                            break

                    if not match:
                        this_graph_hash = get_graph_hash(this_mg.molecule, node_attr="specie") 
                        this_mole_id = get_molecule_id(this_mg.molecule, node_attr="specie")

                        to_run.append(this_mg)
                        to_run_id.append(this_mole_id)
                        to_run_hash.append(this_graph_hash)
                        mol_list.append(this_mg.molecule)

                        to_run_full.append(this_mg)
                        to_run_id_full.append(this_mole_id)
                        to_run_hash_full.append(this_graph_hash)
                        mol_list_full.append(this_mg.molecule)

        if not some_valid_dist:
            no_inf_sep_count += 1
            #print("CANNOT AUTOMATICALLY DETERMINE INFSEP FOR {} {} {} {}".format(name, mtype, ownership_distributions, enter_dist))


print("to_run: ", len(to_run))
print("to_run_full: ", len(to_run_full))
print("to_run_charge_exhaustive: ", len(to_run_charge_exhaustive))

mp_str = [str(i) for i in to_run_id]
mp_str_charge_exhaustive = [str(i) for i in to_run_id_charge_exhaustive]
mp_str_full = [str(i) for i in to_run_id_full]

dict_results = {}
dict_results_charge_exhaustive = {}
dict_results_full = {}

for i in range(len(mp_str)):
    formula = to_run_id[i].split("-")[-3]
    spin = to_run_id[i].split("-")[-1]
    charge = to_run_id[i].split("-")[-2]
    xyz = to_run[i].molecule.cart_coords
    pmg_molecule_graph = to_run[i].as_dict()
    dict_results[str(to_run_id[i])] = {
        "xyz": xyz.tolist(), 
        "pmg_molecule_graph": pmg_molecule_graph, 
        "charge": charge, 
        "spin": int(spin), 
        "formula": formula
        }

for i in range(len(mp_str_charge_exhaustive)):
    formula = to_run_id_charge_exhaustive[i].split("-")[-3]
    spin = to_run_id_charge_exhaustive[i].split("-")[-1]
    charge = to_run_id_charge_exhaustive[i].split("-")[-2]
    xyz = to_run_charge_exhaustive[i].molecule.cart_coords
    pmg_molecule_graph = to_run_charge_exhaustive[i].as_dict()
    dict_results_charge_exhaustive[str(to_run_id_charge_exhaustive[i])] = {
        "xyz": xyz.tolist(), 
        "pmg_molecule_graph": pmg_molecule_graph, 
        "charge": charge, 
        "spin": int(spin), 
        "formula": formula
        }
    
for i in range(len(mp_str_full)):
    formula = to_run_id_full[i].split("-")[-3]
    spin = to_run_id_full[i].split("-")[-1]
    charge = to_run_id_full[i].split("-")[-2]
    xyz = to_run_full[i].molecule.cart_coords
    pmg_molecule_graph = to_run_full[i].as_dict()
    dict_results_full[str(to_run_id_full[i])] = {
        "xyz": xyz.tolist(), 
        "pmg_molecule_graph": pmg_molecule_graph, 
        "charge": charge, 
        "spin": int(spin), 
        "formula": formula
        }
    

# convert to pandas dataframe
print("atoms_not_allocated_count: ", atoms_not_allocated_count
    , "charge_none_count: ", charge_none_count
    #, "charge_not_equal_count: ", charge_not_equal_count
    , "no_inf_sep_count: ", no_inf_sep_count)

df = pd.DataFrame(dict_results).T
df.to_pickle("rapter_filtered_infsep_no_charge_sweep_on_none.pkl")
    
df_charge_exhaustive = pd.DataFrame(dict_results_charge_exhaustive).T
df_charge_exhaustive.to_pickle("rapter_filtered_infsep_charge_sweep_on_none.pkl")

df_full = pd.DataFrame(dict_results_full).T
df_full.to_pickle("rapter_filtered_infsep_charge_sweep_on_none_combined.pkl")