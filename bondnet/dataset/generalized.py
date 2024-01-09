from re import L
import time, copy, bson
import pandas as pd
import networkx as nx
import numpy as np
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from bondnet.core.reaction import Reaction
from bondnet.core.molwrapper import (
    rdkit_mol_to_wrapper_mol,
    create_wrapper_mol_from_atoms_and_bonds,
)
from bondnet.core.reaction import Reaction
from bondnet.core.reaction_collection import ReactionCollection
from bondnet.utils import int_atom, xyz2mol

from rdkit import Chem


# Chem.WrapLogs()
def parse_extra_global_feats(extra_feats):
    #print("global", extra_feats)
    ret_dict = {}
    extra_feature_keys = list(extra_feats.keys())
    # get key as everything after "product_" or "reactant_"
    extra_feature_keys_trimmed = [i.split("_")[1:] for i in extra_feature_keys]
    extra_feature_keys_trimmed = ["_".join(i) for i in extra_feature_keys_trimmed]
    ret_dict = {
        extra_feature_keys_trimmed[index]: extra_feats[i]
        for index, i in enumerate(extra_feature_keys)
    }
    #print("global ", ret_dict)
    return ret_dict


def parse_extra_electronic_feats_atom(extra_feats, inds):
    ret_dict = {}

    extra_feature_keys = list(extra_feats.keys())
    # get key as everything after "product_" or "reactant_"
    extra_feature_keys_trimmed = [i.split("_")[1:] for i in extra_feature_keys]
    extra_feature_keys_trimmed = ["_".join(i) for i in extra_feature_keys_trimmed]

    for index, i in enumerate(extra_feature_keys):
        ret_dict[extra_feature_keys_trimmed[index]] = extra_feats[i]

    if inds != None:
        for k in ret_dict.keys():
            try:
                ret_dict[k] = ret_dict[k][inds]
            except:
                pass
    return ret_dict


def parse_extra_electronic_feats_bond(extra_feats, bond_feat_mappings, dict_bonds_as_root_target_inds):
    #print("bond feats", extra_feats)
    #print("bond mappings", bond_feat_mappings)
    if bond_feat_mappings == None:
        return {}
    
    ret_dict_temp, ret_dict = {}, {}
    num_bonds = int(len(dict_bonds_as_root_target_inds.keys()))
    extra_feature_keys = list(extra_feats.keys())
    

    extra_feature_keys_trimmed = [i.split("_")[1:] for i in extra_feature_keys]
    extra_feature_keys_trimmed = ["_".join(i) for i in extra_feature_keys_trimmed]

    for index, i in enumerate(extra_feature_keys):
        # holds features prior to mapping to bonds
        ret_dict_temp[extra_feature_keys_trimmed[index]] = extra_feats[i]
        # final mat 
        ret_dict[extra_feature_keys_trimmed[index]] = []

    # value of first key is bond indices
    #print("feat mappings", bond_feat_mappings)
    extra_feat_bond_ind = list(bond_feat_mappings.values())[0]
    if type(extra_feat_bond_ind[0][0]) == list:
        extra_feat_bond_ind = extra_feat_bond_ind[0]
    
    extra_feat_bond_ind = [tuple(i) for i in extra_feat_bond_ind]

    if extra_feat_bond_ind == [] and num_bonds != 0:
        for k in ret_dict.keys():
            ret_dict[k] = [0] * num_bonds
        return ret_dict

    for k, v in dict_bonds_as_root_target_inds.items():

        if k in extra_feat_bond_ind:
            ind_in_extra = extra_feat_bond_ind.index(k)
            hit = True
        elif (k[-1], k[0]) in extra_feat_bond_ind:  # reverse order
            ind_in_extra = extra_feat_bond_ind.index((k[-1], k[0]))
            hit = True
        else:
            hit = False
        
        for k2, v2 in ret_dict_temp.items():
            if k2 not in ret_dict:
                ret_dict[k2] = []
            if hit: #and "indices" not in k2:
                val = v2[ind_in_extra]
                ret_dict[k2].append(val)
            else:
                ret_dict[k2].append(0)
    # print("parse bond feats")
    # print(ret_dict)
    return ret_dict


def split_and_map(
    elements,
    bonds,
    coords,
    atom_count,
    reaction_scaffold,
    id,
    bonds_nonmetal=None,
    functional_group=None,
    extra_feats_atom={},
    extra_feats_bond={},
    extra_feats_global={},
    bond_feat_mappings=None,
):
    """
    takes a list of nodes+bonds+reaction bonds and computes subgraphs/species
    also returns mappings

    takes:
        elements(list of strs): list of elements
        bonds(list of list/tuples): bond list
        coords(list of list): atomic position
        atom_count(int): number of nodes/atoms in the total reaction
        reaction_scaffold(list of list/tuples): total bonds in rxn
        id(str): unique id
        bonds_nonmetal(list of tuples/lists): list nonmetal bonds
        charge(int): charge for molecule
        functional_group(str): hydrolysed functional group in reactant, None entries in product
        extra_feats(dict): dictionary w/ extra features

    returns:
        species(list of molwrappers)
        atom_map(dict): maps atomic positions to reaction scaffold
        bond_mapp(dict): maps bonds in subgraphs to reaction template ind
    """
    # print("length of elements", len(elements))
    ret_list, bond_map = [], []
    id = str(id)
    G = nx.Graph()
    G.add_nodes_from([int(i) for i in range(atom_count)])

    for i in bonds:
        G.add_edge(i[0], i[1])
    sub_graphs = [G.subgraph(c) for c in nx.connected_components(G)]

    if len(sub_graphs) >= 2:
        mapping = []
        for ind_sg, sg in enumerate(sub_graphs):
            dict_bonds, dict_bonds_as_root_target_inds = {}, {}
            bond_reindex_list, species_sg, coords_sg = [], [], []
            nodes = list(sg.nodes())
            # finds bonds mapped to subgraphs
            for origin_bond_ind in bonds:
                # check if root to edge is in node list for subgraph
                check = any(item in origin_bond_ind for item in nodes)
                if check:  # if it is then map these to lowest values in nodes
                    bond_orig = nodes.index(origin_bond_ind[0])
                    bond_targ = nodes.index(origin_bond_ind[1])
                    bond_reindex_list.append([bond_orig, bond_targ])
                    # finds the index of these nodes in the reactant bonds
                    ordered_targ_obj = [
                        np.min([origin_bond_ind[0], origin_bond_ind[1]]),
                        np.max([origin_bond_ind[0], origin_bond_ind[1]]),
                    ]

                    original_bond_index = reaction_scaffold.index(ordered_targ_obj)
                    dict_bonds_as_root_target_inds[tuple(ordered_targ_obj)] = (
                        bond_orig,
                        bond_targ,
                    )
                    dict_bonds[len(bond_reindex_list) - 1] = original_bond_index

            bond_map.append(dict_bonds)

            for site in nodes:
                species_sg.append(elements[site])
                coords_sg.append(coords[site])

            mapping_temp = {i: ind for i, ind in enumerate(nodes)}
            mapping.append(mapping_temp)

            if extra_feats_atom != {}:
                atom_feats = parse_extra_electronic_feats_atom(extra_feats_atom, nodes)
            else:
                atom_feats = {}

            if extra_feats_bond != {}:
                # print("sg len > 1")
                # print(extra_feats_bond)
                # print(dict_bonds_as_root_target_inds)
                bond_feats = parse_extra_electronic_feats_bond(
                    extra_feats = extra_feats_bond, 
                    bond_feat_mappings=bond_feat_mappings, 
                    dict_bonds_as_root_target_inds=dict_bonds_as_root_target_inds
                )
            else:
                bond_feats = {}

            species_molwrapper = create_wrapper_mol_from_atoms_and_bonds(
                species=species_sg,
                coords=coords_sg,
                bonds=bond_reindex_list,
                functional_group=functional_group,
                atom_features=atom_feats,
                bond_features=bond_feats,
                global_features=parse_extra_global_feats(extra_feats_global),
                identifier=id + "_" + str(ind_sg),
            )

            if bonds_nonmetal == None:
                species_molwrapper.nonmetal_bonds = bond_reindex_list
            else:
                non_metal_filter = []
                for bond_non_metal in bonds_nonmetal:
                    if bond_non_metal[0] in nodes or bond_non_metal[1] in nodes:
                        non_metal_filter.append(bond_non_metal)
                species_molwrapper.nonmetal_bonds = bonds_nonmetal

            ret_list.append(species_molwrapper)

    else:
        bond_reindex_list = []
        dict_temp, dict_bonds, dict_bonds_as_root_target_inds = {}, {}, {}

        for origin_bond_ind in bonds:
            nodes = list(G.nodes())
            check = any(item in origin_bond_ind for item in nodes)
            if check:  # if it is then map these to lowest values in nodes
                bond_orig = nodes.index(origin_bond_ind[0])
                bond_targ = nodes.index(origin_bond_ind[1])
                bond_reindex_list.append([bond_orig, bond_targ])
                # finds the index of these nodes in the reactant bonds
                ordered_targ_obj = [
                    np.min([origin_bond_ind[0], origin_bond_ind[1]]),
                    np.max([origin_bond_ind[0], origin_bond_ind[1]]),
                ]
                original_bond_index = reaction_scaffold.index(ordered_targ_obj)
                dict_bonds[len(bond_reindex_list) - 1] = original_bond_index

                dict_bonds_as_root_target_inds[tuple(ordered_targ_obj)] = (
                    bond_orig,
                    bond_targ,
                )

        bond_map = [dict_bonds]

        if extra_feats_atom != {}:
            atom_feats = parse_extra_electronic_feats_atom(
                extra_feats_atom, list(G.nodes())
            )
            # print("atopmic feats")
            # print(atom_feats)
        else:
            atom_feats = {}

        if extra_feats_bond != {}:
            # print("extra features in bond")
            # print(extra_feats_bond)
            bond_feats = parse_extra_electronic_feats_bond(
                extra_feats=extra_feats_bond, 
                bond_feat_mappings=bond_feat_mappings,
                dict_bonds_as_root_target_inds=dict_bonds_as_root_target_inds
            )
            # print("bond feats after parsing")
            # print(bond_feats)
        else:
            # print("empty bond features!!!!!")
            bond_feats = {}

        species_molwrapper = create_wrapper_mol_from_atoms_and_bonds(
            species=elements,
            coords=coords,
            bonds=bonds,
            functional_group=functional_group,
            atom_features=atom_feats,
            bond_features=bond_feats,
            global_features=parse_extra_global_feats(extra_feats_global),
            identifier=id,
        )

        if bonds_nonmetal == None:
            species_molwrapper.nonmetal_bonds = bonds
        else:
            species_molwrapper.nonmetal_bonds = bonds_nonmetal

        # atom map
        for i in range(len(elements)):
            dict_temp[i] = i
        mapping = [dict_temp]
        ret_list.append(species_molwrapper)

    if len(ret_list) != len(mapping):
        print("ret list not equal to atom mapping list")
    if len(ret_list) != len(bond_map):
        print("ret list not equal to bond mapping list")

    return ret_list, mapping, bond_map


def process_species_graph(
    row,
    classifier=False,
    target="ts",
    reverse_rxn=False,
    verbose=False,
    filter_species=None,
    filter_outliers=False,
    filter_sparse_rxns=False,
    lower_bound=-99,
    upper_bound=100,
    feature_filter=False,
    categories=5,
    extra_keys=None,
    extra_info=None,
):
    """
    Takes a row and processes the products/reactants - entirely defined by graphs from row

    Args:
        row: the row (series) pandas object


    Returns:
        mol_list: a list of MolecularWrapper object(s) for product(s) or reactant
    """

    rxn = []
    #print("extra_keys", extra_keys)
    if filter_species == None:
        filter_prod = -99
        filter_reactant = -99
    else:
        filter_prod = filter_species[1]
        filter_reactant = filter_species[0]

    reactant_key = "reactant"
    product_key = "product"

    #charge = row["charge"]
    formed_len = len(row["bonds_formed"])
    broken_len = len(row["bonds_broken"])
    broken_bonds = [tuple(i) for i in row["bonds_broken"]]
    formed_bonds = [tuple(i) for i in row["bonds_formed"]]
    check_list_len = broken_len + formed_len

    if reverse_rxn:
        reactant_key = "product"
        product_key = "reactant"
        formed_len = len(row["bonds_broken"])
        broken_len = len(row["bonds_formed"])
        formed_bonds = row["bonds_broken"]
        broken_bonds = row["bonds_formed"]

    if broken_len == 0:
        temp_key = copy.deepcopy(reactant_key)
        reactant_key = product_key
        product_key = temp_key
        reverse_rxn = not reverse_rxn
        temp_broken = copy.deepcopy(broken_bonds)
        broken_bonds = formed_bonds
        formed_bonds = temp_broken

    bonds_reactant = row[reactant_key + "_bonds"]
    bonds_products = row[product_key + "_bonds"]
    bonds_nonmetal_product = row[product_key + "_bonds_nometal"]
    bonds_nonmetal_reactant = row[reactant_key + "_bonds_nometal"]
    
    if "functional_group_reacted" in row.index:
        reactant_functional_group = row["functional_group_reacted"]
    else:
        reactant_functional_group = None
    product_functional_group = None

    try:
        pymat_graph_reactants = row["combined_" + reactant_key + "s_graph"]["molecule"][
            "sites"
        ]
        pymat_graph_products = row["combined_" + product_key + "s_graph"]["molecule"][
            "sites"
        ]
    except:
        pymat_graph_reactants = row[reactant_key + "_molecule_graph"]["molecule"][
            "sites"
        ]
        pymat_graph_products = row[product_key + "_molecule_graph"]["molecule"]["sites"]

    species_reactant_full = [int_atom(i["name"]) for i in pymat_graph_reactants]
    species_products_full = [int_atom(i["name"]) for i in pymat_graph_products]
    coords_reactant = [i["xyz"] for i in pymat_graph_reactants]
    coords_products_full = [i["xyz"] for i in pymat_graph_products]

    # new
    if type(bonds_reactant[0][0]) == list:
        bonds_reactant = bonds_reactant[0]
        bonds_nonmetal_product = bonds_nonmetal_product[0]
        
        #bonds_reactant = [bonds_reactant]
    if type(bonds_products[0][0]) == list:
        bonds_products = bonds_products[0]
        bonds_nonmetal_reactant = bonds_nonmetal_reactant[0]
        #print("here!!" *10 )
        #
        
    #print("bonds reactant", type(bonds_reactant[0][0]))
    #print("bonds products", bonds_products[0][0])
    #print(bonds_reactant)

    total_bonds = [tuple(bond) for bond in bonds_reactant]
    [
        total_bonds.append((np.min(np.array(i)), np.max(np.array(i))))
        for i in bonds_products
    ]
    #print(total_bonds)
    total_bonds = list(set(total_bonds))
    total_bonds = [list(bond) for bond in total_bonds]

    num_nodes = 0
    for i in row["composition"].items():
        num_nodes += int(i[-1])
    
    extra_keys_full = {}
    # checks if there are other features to add to mol_wrapper object
    extra_keys_row = list(row.index)
    # translates from user input to actual keys in dataframe
    for k, v in extra_keys.items():
        for i in extra_keys_row:
            if k != "mappings":
                for j in v:
                    if k in i and j in i:
                        if k not in extra_keys_full:
                            extra_keys_full[k] = []
                        extra_keys_full[k].append(i)    
            else: 
                if v[0] in i: # there should only be one mapping key
                    if k not in extra_keys_full:
                        extra_keys_full[k] = []
                    extra_keys_full[k].append(i)

    #print("extra keys post filter: ", extra_keys_full)
    #print("full extra keys", extra_keys_full)
    # check all pandas columns that start with "extra_feat_atom" or "extra_feat_bond"
    extra_atom_feats_dict_prod, extra_atom_feats_dict_react = {}, {}
    extra_bond_feats_dict_prod, extra_bond_feats_dict_react = {}, {}
    extra_global_feats_dict_prod, extra_global_feats_dict_react = {}, {}
    extra_bond_feats_mappings_prod, extra_bond_feats_mappings_react = {}, {}

    #print("extra keys", extra_keys)
    #print("extra keys full", extra_keys_full)
    
    if "atom" in extra_keys_full.keys() and extra_keys_full["atom"] != [] and extra_keys_full["atom"] != None:
        for key in extra_keys_full["atom"]:
            #print("checking atom ", key)
            prod = False
            # check if next underscore is product or reactant
            # add key to dict if reactant equivalent is also present
            if key.split("_")[3] == "product":
                opposite_key = key.replace("product", "reactant")
                prod = True
            if key.split("_")[3] == "reactant":
                opposite_key = key.replace("reactant", "product")
                prod = False
            if opposite_key in extra_keys_full["atom"]:
                final_key = key.replace("extra_feat_atom_", "")
                if prod:
                    extra_atom_feats_dict_prod[final_key] = row[key]
                else:
                    extra_atom_feats_dict_react[final_key] = row[key]
    
    if "bond" in extra_keys_full.keys() and extra_keys_full["bond"] != [] and extra_keys_full["bond"] != None :
        for key in extra_keys_full["bond"] :
            prod = False
            #print("checking bond", key)
            # check if next underscore is product or reactant
            # add key to dict if reactant equivalent is also present
            if key.split("_")[3] == "product":
                opposite_key = key.replace("product", "reactant")
                prod = True

            if key.split("_")[3] == "reactant":
                opposite_key = key.replace("reactant", "product")
                prod = False
            #print(key, opposite_key)
            if opposite_key in extra_keys_full["bond"]:
                # remove the "extra_feat_bond" from the key
                final_key = key.replace("extra_feat_bond_", "")
                #if "indices" in key:
                #    row_data = row[key]
                #else:
                row_data = row[key][0]

                if prod:
                    extra_bond_feats_dict_prod[final_key] = row_data
                else:
                    extra_bond_feats_dict_react[final_key] = row_data

    if "mappings" in extra_keys_full.keys() and extra_keys_full["mappings"] != [] and extra_keys_full["mappings"] != None:
        for key in extra_keys_full["mappings"]:
            prod = False
            #print("checking mapping", key)
            if key.split("_")[3] == "product":
                opposite_key = key.replace("product", "reactant")
                prod = True                

            if key.split("_")[3] == "reactant":
                opposite_key = key.replace("reactant", "product")
                prod = False
            
            if opposite_key in extra_keys_full["mappings"]:
                # remove the "extra_feat_bond" from the key
                final_key = key.replace("extra_feat_bond_", "")
                row_data = row[key]#[0]

                if prod:
                    extra_bond_feats_mappings_prod[final_key] = row_data
                else:
                    extra_bond_feats_mappings_react[final_key] = row_data

    if "global" in extra_keys_full.keys() and extra_keys_full["global"] != [] and extra_keys_full["global"] != None:
        for key in extra_keys_full["global"]:
            #print("checking global", key)
            prod = False
            # check if next underscore is product or reactant
            # add key to dict if reactant equivalent is also present
            if key.split("_")[3] == "product":
                opposite_key = key.replace("product", "reactant")
                prod = True

            if key.split("_")[3] == "reactant":
                opposite_key = key.replace("reactant", "product")
                prod = False
            #print(key, opposite_key)
            if opposite_key in extra_keys_full["global"]:
                # remove the "extra_feat_bond" from the key
                final_key = key.replace("extra_feat_global_", "")

                if prod:
                    extra_global_feats_dict_prod[final_key] = row[key]
                else:
                    extra_global_feats_dict_react[final_key] = row[key]

    
    #print("extra global feats dict prod", extra_global_feats_dict_prod)
    #print("extra global feats dict react", extra_global_feats_dict_react)
    #print("extra atom feats dict prod", extra_atom_feats_dict_prod)
    #print("extra atom feats dict react", extra_atom_feats_dict_react)
    #print("extra bond feats dict prod", extra_bond_feats_dict_prod)
    #print("extra bond feats dict react", extra_bond_feats_dict_react)
    #print("extra bond feats mappings prod", extra_bond_feats_mappings_prod)
    #print("extra bond feats mappings react", extra_bond_feats_mappings_react)

    if feature_filter:  # filter out reactions without complete features
        filter_rxn = False
        for dict in [
            extra_atom_feats_dict_prod,
            extra_atom_feats_dict_react,
            extra_bond_feats_dict_prod,
            extra_bond_feats_dict_react,
            
        ]:
            for k, v in dict.items():
                if v == [] or v == None or v == [[]]:
                    filter_rxn = True
        if filter_rxn:
            if verbose:
                print("filter rxn bc of missing features")
            return []
    #print("extra atom feats dict prod", extra_atom_feats_dict_prod)
    #print("extra atom feats dict react", extra_atom_feats_dict_react)
    #print("extra bond feats dict prod", extra_bond_feats_dict_prod)
    #print("extra bond feats dict react", extra_bond_feats_dict_react)
        # new
    if type(bonds_reactant[0][0]) == list:
        bonds_reactant = row[reactant_key + "_bonds"]
        bonds_nonmetal_product = bonds_nonmetal_product[0]
        
        #bonds_reactant = [bonds_reactant]
    if type(bonds_products[0][0]) == list:
        bonds_products = row[product_key + "_bonds"]
        bonds_nonmetal_reactant = bonds_nonmetal_reactant[0]

    products, atoms_products, mapping_products = split_and_map(
        elements=species_products_full,
        coords=coords_products_full,
        bonds=bonds_products,
        atom_count=num_nodes,
        reaction_scaffold=total_bonds,
        id=str(row[product_key + "_id"]),
        bonds_nonmetal=bonds_nonmetal_product,
        functional_group=product_functional_group,
        extra_feats_atom=extra_atom_feats_dict_prod,
        extra_feats_bond=extra_bond_feats_dict_prod,
        extra_feats_global=extra_global_feats_dict_prod,
        bond_feat_mappings = extra_bond_feats_mappings_prod
    )

    reactants, atoms_reactants, mapping_reactants = split_and_map(
        elements=species_reactant_full,
        coords=coords_reactant,
        bonds=bonds_reactant,
        atom_count=num_nodes,
        reaction_scaffold=total_bonds,
        id=str(row[reactant_key + "_id"]),
        bonds_nonmetal=bonds_nonmetal_reactant,
        functional_group=reactant_functional_group,
        extra_feats_atom=extra_atom_feats_dict_react,
        extra_feats_bond=extra_bond_feats_dict_react,
        extra_feats_global=extra_global_feats_dict_react,
        bond_feat_mappings = extra_bond_feats_mappings_react 
    )
    # print("reactant bond features")
    # for i in reactants:
    #    # print(i.atom_features)
    #    # print(i.bond_features)

    total_atoms = list(
        set(list(np.concatenate([list(i.values()) for i in atoms_reactants]).flat))
    )
    check = False
    if check:
        total_atoms_check = list(
            set(list(np.concatenate([list(i.values()) for i in atoms_products]).flat))
        )
        assert (
            total_atoms == total_atoms_check
        ), "atoms in reactant and products are not equal"

    if products != [] and reactants != []:
        rxn_type = []

        if filter_prod != -99:
            if len(products) > filter_prod:
                if verbose:
                    print("too many products")
                return []
        if filter_reactant != -99:
            if len(reactants) > filter_reactant:
                if verbose:
                    print("too many reactants")
                return []

        try:
            id = [i for i in row["reaction_id"].split("-")]
            id = str(id[0] + id[1] + id[2])
        except:
            id = str(row["reactant_id"])
            if type(row["product_id"]) == list:
                for i in row["product_id"]:
                    id += str(i)
            else:
                id += str(row["product_id"])
            id = str(id)

        if target == "ts":
            value = row["transition_state_free_energy"] - row[reactant_key + "_free_energy"]
            reverse_energy = (
                row["transition_state_free_energy"] - row[product_key + "_free_energy"]
            )
            if reverse_energy < 0.0:
                reverse_energy = 0.0
            if value < 0.0:
                value = 0.0
        elif target == "dG_sp":
            value = row["dG_sp"]
            reverse_energy = -value
        elif target == "diff":
            value = row[product_key + "_free_energy"] - row[reactant_key + "_free_energy"]
            reverse_energy = (
                row[reactant_key + "_free_energy"] - row[product_key + "_free_energy"]
            )
        else:
            value = row[target]
            reverse_energy = -value

        if classifier:
            if categories == 3:
                if value <= 0.1:
                    value = 0
                elif value < 0.7 and value > 0.1:
                    value = 1
                else:
                    value = 2

                if reverse_energy <= 0.1:
                    reverse_energy = 0
                elif reverse_energy < 0.7 and reverse_energy > 0.1:
                    reverse_energy = 1
                else:
                    reverse_energy = 2

            else:
                if value <= 0.04:
                    value = 0
                elif value < 0.3 and value > 0.04:
                    value = 1
                elif value < 0.7 and value > 0.3:
                    value = 2
                elif value < 1.5 and value > 0.7:
                    value = 3
                else:
                    value = 4

                if reverse_energy <= 0.04:
                    reverse_energy = 0
                elif reverse_energy < 0.3 and reverse_energy > 0.04:
                    reverse_energy = 1
                elif reverse_energy < 0.7 and reverse_energy > 0.3:
                    reverse_energy = 2
                elif reverse_energy < 1.5 and reverse_energy > 0.7:
                    reverse_energy = 3
                else:
                    reverse_energy = 4

        if len(broken_bonds) > 0:
            for i in broken_bonds:
                key = "broken_"
                index = i

                try:
                    atom_1 = row["combined_" + reactant_key + "s_graph"]["molecule"][
                        "sites"
                    ][index[0]]["name"]
                    atom_2 = row["combined_" + reactant_key + "s_graph"]["molecule"][
                        "sites"
                    ][index[1]]["name"]
                except:
                    atom_1 = row[reactant_key + "_molecule_graph"]["molecule"]["sites"][
                        index[0]
                    ]["name"]
                    atom_2 = row[reactant_key + "_molecule_graph"]["molecule"]["sites"][
                        index[1]
                    ]["name"]

                atoms = [atom_1, atom_2]
                atoms.sort()
                key += atoms[0] + "_" + atoms[1]
                rxn_type.append(key)

        if len(formed_bonds) > 0:
            for i in formed_bonds:
                key = "formed_"
                index = i
                try:
                    atom_1 = row["combined_" + reactant_key + "s_graph"]["molecule"][
                        "sites"
                    ][index[0]]["name"]
                    atom_2 = row["combined_" + reactant_key + "s_graph"]["molecule"][
                        "sites"
                    ][index[1]]["name"]
                except:
                    atom_1 = row[reactant_key + "_molecule_graph"]["molecule"]["sites"][
                        index[0]
                    ]["name"]
                    atom_2 = row[reactant_key + "_molecule_graph"]["molecule"]["sites"][
                        index[1]
                    ]["name"]

                atoms = [atom_1, atom_2]
                atoms.sort()
                key += atoms[0] + "_" + atoms[1]
                rxn_type.append(key)

        if filter_sparse_rxns:
            filter_rxn_list = [
                "broken_C_C",
                "broken_C_Cl",
                "broken_C_F",
                "broken_C_H",
                "broken_C_Li",
                "broken_C_N",
                "broken_C_O",
                "broken_H_Li",
                "broken_F_H",
                "broken_H_N",
                "broken_H_O",
                "broken_Li_O",
                "formed_C_C",
                "formed_C_Cl",
                "formed_C_F",
                "formed_C_H",
                "formed_C_Li",
                "formed_C_N",
                "formed_C_O",
                "formed_H_Li",
                "formed_F_H",
                "formed_H_H",
                "formed_H_O",
                "formed_Li_O",
            ]

            check = any(item in rxn_type for item in filter_rxn_list)
            if check == False:
                print("filtering rxn")
                return []

        extra_info_dict = {}
        if extra_info != None:
            for key in extra_info:
                if key in row.keys():
                    extra_info_dict[key] = row[key]

        rxn = Reaction(
            reactants=reactants,
            products=products,
            free_energy=value,
            broken_bond=broken_bonds,
            formed_bond=formed_bonds,
            total_bonds=total_bonds,
            total_atoms=total_atoms,
            reverse_energy_target=reverse_energy,
            identifier=id,
            reaction_type=rxn_type,
            extra_info=extra_info_dict,
        )
        atom_mapping_check = []
        for i in atoms_reactants:
            for key in i.keys():
                atom_mapping_check.append(i[key])
        atom_mapping_check = list(set(atom_mapping_check))
        if atom_mapping_check != total_atoms:
            print(atom_mapping_check, total_atoms)

        rxn.set_atom_mapping([atoms_reactants, atoms_products])
        rxn._bond_mapping_by_int_index = [mapping_reactants, mapping_products]

        outlier_condition = lower_bound > value or upper_bound < value
        if outlier_condition and filter_outliers:
            return []

    return rxn


def process_species_rdkit(row, classifier=False):
    """
    Takes a row and processes the products/reactants - entirely defined by rdkit definitions

    Args:
        row: the row (series) pandas object

    Returns:
        mol_list: a list of MolecularWrapper object(s) for product(s) or reactant
    """
    fail = 0
    rxn, reactant_list, product_list, bond_map = [], [], [], []
    reactant_key = "reactant"
    product_key = "product"

    reverse_rxn = False
    if row["bonds_broken"] == [] and row["bonds_formed"] != []:
        reverse_rxn = True
        reactant_key = "product"
        product_key = "reactant"

    if row["bonds_broken"] == [] and row["bonds_formed"] == []:
        return rxn

    species_reactant_full = [
        int_atom(i["name"])
        for i in row[reactant_key + "_molecule_graph"]["molecule"]["sites"]
    ]
    species_products_full = [
        int_atom(i["name"])
        for i in row[product_key + "_molecule_graph"]["molecule"]["sites"]
    ]
    coords_reactant_full = [
        i["xyz"] for i in row[reactant_key + "_molecule_graph"]["molecule"]["sites"]
    ]
    coords_products_full = [
        i["xyz"] for i in row[product_key + "_molecule_graph"]["molecule"]["sites"]
    ]

    charge = row["charge"]
    id = str(row[reactant_key + "_id"])
    free_energy = row[product_key + "_free_energy"]

    reactant_mol = xyz2mol(
        atoms=species_reactant_full,
        coordinates=coords_reactant_full,
        charge=charge,
    )
    reactant_wrapper = rdkit_mol_to_wrapper_mol(
        reactant_mol[0], charge=charge, free_energy=free_energy, identifier=id
    )
    reactant_list.append(reactant_wrapper)

    # handle products
    # check subgraphs first
    num_nodes = 0
    for i in row["composition"].items():
        num_nodes += int(i[-1])
    G = nx.Graph()
    G.add_nodes_from([int(i) for i in range(num_nodes)])
    for i in row[product_key + "_bonds"]:
        G.add_edge(i[0], i[1])  # rdkit bonds are a subset of user-defined bonds
    sub_graphs = [G.subgraph(c) for c in nx.connected_components(G)]
    id = str(row[product_key + "_id"])

    # still no handling for rxns A --> B + C +....
    if len(sub_graphs) > 2:
        pass  # print("cannot handle three or more products")
    # handle A --> B + C
    elif len(sub_graphs) == 2:
        mapping, mol_prod = [], []
        for sg in sub_graphs:
            coords_products, species_products, bond_reindex_list = [], [], []
            nodes = list(sg.nodes())
            bonds = list(sg.edges())

            # finds bonds mapped to subgraphs
            for origin_bond_ind in row[product_key + "_bonds"]:
                # check if root to edge is in node list for subgraph
                check = any(item in origin_bond_ind for item in nodes)
                if check:
                    bond_orig = nodes.index(origin_bond_ind[0])
                    bond_targ = nodes.index(origin_bond_ind[1])
                    bond_reindex_list.append([bond_orig, bond_targ])

            for site in nodes:
                species_products.append(
                    int_atom(
                        row[product_key + "_molecule_graph"]["molecule"]["sites"][site][
                            "name"
                        ]
                    )
                )
                coords_products.append(
                    row[product_key + "_molecule_graph"]["molecule"]["sites"][site][
                        "xyz"
                    ]
                )

            mapping_temp = {i: ind for i, ind in enumerate(nodes)}
            mapping.append(mapping_temp)

            mol = xyz2mol(
                atoms=species_products,
                coordinates=coords_products,
                charge=charge,
            )[0]
            product = rdkit_mol_to_wrapper_mol(
                mol, charge=charge, free_energy=free_energy, identifier=id
            )
            product_list.append(product)
    else:
        mol_prod = xyz2mol(
            atoms=species_products_full,
            coordinates=coords_products_full,
            charge=charge,
        )[0]
        product = rdkit_mol_to_wrapper_mol(
            mol_prod, charge=charge, free_energy=free_energy, identifier=id
        )
        product_list.append(product)

        # atom mapping - order is preserved
        dict_temp = {}
        for i in range(len(species_products_full)):
            dict_temp[i] = i
        mapping = [dict_temp]

    if fail == 0 and product_list != [] and reactant_list != []:
        id = [i for i in row["reaction_id"].split("-")]
        id = int(id[0] + id[1] + id[2])
        broken_bond = None

        if row["bonds_broken"] != []:
            broken_bond = row["bonds_broken"][0]
        if reverse_rxn:
            broken_bond = row["bonds_formed"][0]

        if reverse_rxn:
            value = row["transition_state_energy"] - row["product_energy"]
        else:
            value = row["dE_barrier"]

        if classifier:
            if value <= 0.04:
                value = 0
            elif value < 0.3 and value > 0.04:
                value = 1
            elif value < 0.7 and value > 0.3:
                value = 2
            elif value < 1.5 and value > 0.7:
                value = 3
            else:
                value = 4

        rxn = Reaction(
            reactants=reactant_list,
            products=product_list,
            free_energy=value,
            broken_bond=broken_bond,
            identifier=id,
        )
        rxn.set_atom_mapping(mapping)
    return rxn


def task_done(future):
    try:
        result = future.result()  # blocks until results are ready
    except TimeoutError as error:
        print("Function took longer than %d seconds" % error.args[1])
    except Exception as error:
        print("Function raised %s" % error)
        print(error.traceback)  # traceback of the function


def create_reaction_network_files_and_valid_rows(
    filename,
    bond_map_filter=False,
    target="ts",
    classifier=False,
    debug=False,
    filter_species=False,
    filter_outliers=True,
    filter_sparse_rxn=False,
    feature_filter=False,
    categories=5,
    extra_keys=None,
    extra_info=None,
    return_reactions=False,
):
    """
    Processes json file or bson from emmet to use in training bondnet

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for trianing
        bond_map_filter: true uses filter with sdf
        target (str): target for regression either 'ts' or 'diff'
        classifier(bool): whether to create a classification or reg. task
        debug(bool): use smaller dataset or not
    """

    # path_mg_data = "../../../dataset/mg_dataset/20220613_reaction_data.json"

    print("reading file from: {}".format(filename))
    if filename.endswith(".json"):
        path_json = filename
        mg_df = pd.read_json(path_json)
    elif filename.endswith(".pkl"):
        path_pkl = filename
        mg_df = pd.read_pickle(path_pkl)

    else:
        path_bson = filename
        with open(path_bson, "rb") as f:
            data = bson.decode_all(f.read())
        mg_df = pd.DataFrame(data)

    start_time = time.perf_counter()
    reactions, ind_val, rxn_raw, ind_final = [], [], [], []
    lower_bound, upper_bound = 0, 0

    if debug:
        mg_df = mg_df.head(100)

    if filter_outliers:
        if target == "ts":
            energy_dist = mg_df["transition_state_free_energy"] - mg_df["reactant_free_energy"]

        elif target == "dG_sp":
            energy_dist = mg_df["dG_sp"]

        elif target == "diff":
            energy_dist = mg_df["product_free_energy"] - mg_df["reactant_free_energy"]

        else:
            energy_dist = mg_df[target]
        q1, q3, med = (
            np.quantile(energy_dist, 0.25),
            np.quantile(energy_dist, 0.75),
            np.median(energy_dist),
        )
        # finding the iqr region
        iqr = q3 - q1
        # finding upper and lower whiskers
        upper_bound = q3 + (2.0 * iqr)
        lower_bound = q1 - (2.0 * iqr)
    #print("extra keys: ", extra_keys)
    with ProcessPool(max_workers=12, max_tasks=10) as pool:
        for ind, row in mg_df.iterrows():
            future = pool.schedule(
                process_species_graph,
                args=[row],
                kwargs={
                    "classifier": classifier,
                    "target": target,
                    "reverse_rxn": False,
                    "verbose": False,
                    "categories": categories,
                    "filter_species": filter_species,
                    "filter_outliers": filter_outliers,
                    "upper_bound": upper_bound,
                    "lower_bound": lower_bound,
                    "filter_sparse_rxns": filter_sparse_rxn,
                    "feature_filter": feature_filter,
                    "extra_keys": extra_keys,
                    "extra_info": extra_info,
                },
                timeout=30,
            )
            future.add_done_callback(task_done)
            try:
                rxn_raw.append(future.result())
                ind_val.append(ind)
            except:
                pass

    finish_time = time.perf_counter()
    print("rxn raw len: {}".format(int(len(rxn_raw))))
    print(f"Program finished in {finish_time-start_time} seconds")
    fail_default, fail_count, fail_sdf_map, fail_prod_len = 0, 0, 0, 0

    for ind, rxn_temp in enumerate(rxn_raw):
        if not isinstance(rxn_temp, list):
            try:
                if bond_map_filter:
                    try:
                        bond_map = rxn_temp.bond_mapping_by_int_index()
                        rxn_temp._bond_mapping_by_int_index = bond_map
                        reactant_bond_count = int(
                            len(rxn_temp.reactants[0].rdkit_mol.GetBonds())
                        )  # here we're using rdkit still
                        prod_bond_count = 0
                        for i in rxn_temp.products:
                            prod_bond_count += int(len(i.rdkit_mol.GetBonds()))
                        if reactant_bond_count < prod_bond_count:
                            fail_prod_len += 1
                        else:
                            reactions.append(rxn_temp)
                            ind_final.append(ind_val[ind])
                    except:
                        fail_sdf_map += 1
                else:
                    bond_map = rxn_temp.bond_mapping_by_int_index()

                    if len(rxn_temp.reactants) != len(bond_map[0]) or len(
                        rxn_temp.products
                    ) != len(bond_map[1]):
                        print("mappings invalid")
                        fail_count += 1
                    else:
                        rxn_temp._bond_mapping_by_int_index = bond_map
                        reactions.append(rxn_temp)
                        ind_final.append(ind_val[ind])
            except:
                fail_count += 1
        else:
            fail_default += 1

    print(".............failures.............")
    print("reactions len: {}".format(int(len(reactions))))
    print("valid ind len: {}".format(int(len(ind_final))))
    print("bond break fail count: \t\t{}".format(fail_count))
    print("default fail count: \t\t{}".format(fail_default))
    print("sdf map fail count: \t\t{}".format(fail_sdf_map))
    print("product bond fail count: \t{}".format(fail_prod_len))
    print("about to group and organize")
    extra_info = False
    if extra_info is not None:
        extra_info_tf = True

    extractor = ReactionCollection(
        reactions
    )  # this needs to be gutted and modified entirely
    # un comment to show # of products and reactants
    # [
    #    print(
    #        "# reactants: {}, # products: {}".format(len(i.reactants), len(i.products))
    #    )
    #    for i in reactions
    # ]
    (
        all_mols,
        all_labels,
        features,
    ) = extractor.create_struct_label_dataset_reaction_based_regression_general(
        group_mode="all",
        sdf_mapping=False,
        extra_info=extra_info_tf,
    )
    if return_reactions:
        return all_mols, all_labels, features, reactions
    return all_mols, all_labels, features
