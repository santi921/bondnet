from re import L
import time
import pandas as pd
import networkx as nx
from tqdm import tqdm
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

Chem.WrapLogs()


def process_species_graph(row, classifier=False):
    """
    Takes a row and processes the products/reactants - entirely defined by graphs from row

    Args:
        row: the row (series) pandas object

    Returns:
        mol_list: a list of MolecularWrapper object(s) for product(s) or reactant
    """
    fail = 0
    rxn, reactant_list, product_list, bond_map = [], [], [], []
    reactant_key = 'reactant'
    product_key = 'product'
    
    reverse_rxn = False
    check_list = row['bonds_formed'] + row['bonds_broken']
    if(len(check_list) == 0 or len(check_list) > 1): 
        return rxn 
    else:
        if(row['bonds_broken'] == [] and row['bonds_formed'] != []):
            if(len(row['bonds_formed']) > 1 or len(row['bonds_formed']) == 0): 
                return rxn
            else:
                reverse_rxn = True
                reactant_key = 'product'
                product_key = 'reactant'
    

    species_reactant = [
        int_atom(i["name"]) for i in row[reactant_key+"_molecule_graph"]["molecule"]["sites"]
    ]
    species_products_full = [
        int_atom(i["name"]) for i in row[product_key+"_molecule_graph"]["molecule"]["sites"]
    ]
    coords_reactant = [
        i["xyz"] for i in row[reactant_key+"_molecule_graph"]["molecule"]["sites"]
    ]
    coords_products_full = [
        i["xyz"] for i in row[product_key+"_molecule_graph"]["molecule"]["sites"]
    ]

    charge = row["charge"]
    id = str(row[reactant_key+"_id"])
    free_energy = row[product_key+"_free_energy"]
    bonds_reactant = row[reactant_key+"_bonds"]
    bonds_products = row[product_key+"_bonds"]

    reactant = create_wrapper_mol_from_atoms_and_bonds(
        species_reactant,
        coords_reactant,
        bonds_reactant,
        charge=charge,
        free_energy=free_energy,
        identifier=id,
    )
    reactant.nonmetal_bonds = row[reactant_key+'_bonds_nometal']
    reactant_list.append(reactant)

    # handle products
    # check subgraphs first
    num_nodes = 0
    for i in row["composition"].items():
        num_nodes += int(i[-1])
    G = nx.Graph()
    G.add_nodes_from([int(i) for i in range(num_nodes)])
    for i in row[product_key+"_bonds"]:
        G.add_edge(i[0], i[1])  # rdkit bonds are a subset of user-defined bonds
    sub_graphs = [G.subgraph(c) for c in nx.connected_components(G)]
    id = str(row[product_key+"_id"])

    # still no handling for rxns A --> B + C +....
    if len(sub_graphs) > 2:
        pass #print("cannot handle three or more products")
    # handle A --> B + C
    elif len(sub_graphs) == 2:
        mapping, mol_prod = [], []
        for ind_sg, sg in enumerate(sub_graphs):
            dict_prod = {}
            coords_products, species_products, bond_reindex_list = [], [], []
            nodes = list(sg.nodes())
            bonds = list(sg.edges())

            # finds bonds mapped to subgraphs
            for origin_bond_ind in row[product_key+"_bonds"]:
                # check if root to edge is in node list for subgraph
                check = any(item in origin_bond_ind for item in nodes)
                if check:
                    bond_orig = nodes.index(origin_bond_ind[0])
                    bond_targ = nodes.index(origin_bond_ind[1])
                    bond_reindex_list.append([bond_orig, bond_targ])
                    # finds the index of these nodes in the reactant bonds
                    try:
                        original_bond_index = row[reactant_key+"_bonds"].index(
                            [origin_bond_ind[0], origin_bond_ind[1]]
                        )
                        dict_prod[len(bond_reindex_list) - 1] = original_bond_index
                    except:
                        print("detected bond in prod. not in react.")
            bond_map.append(dict_prod)

            for site in nodes:
                species_products.append(
                    int_atom(
                        row[product_key+"_molecule_graph"]["molecule"]["sites"][site]["name"]
                    )
                )
                coords_products.append(
                    row[product_key+"_molecule_graph"]["molecule"]["sites"][site]["xyz"]
                )

            mapping_temp = {i: ind for i, ind in enumerate(nodes)}
            mapping.append(mapping_temp)

            product = create_wrapper_mol_from_atoms_and_bonds(
                species_products,
                coords_products,
                bond_reindex_list,
                charge=0,
                free_energy=free_energy,
                identifier=id+"_" + str(ind_sg),
            )
            
            non_metal_products = row[product_key+'_bonds_nometal']
            non_metal_filter = []
            for bond_non_metal in non_metal_products:
                if(bond_non_metal[0] in nodes or bond_non_metal[1] in nodes):
                    non_metal_filter.append(bond_non_metal)
            product.nonmetal_bonds = non_metal_filter
            product_list.append(product)

    else:
        dict_prod = {}

        product = create_wrapper_mol_from_atoms_and_bonds(
            species_products_full,
            coords_products_full,
            bonds_products,
            charge=charge,
            free_energy=free_energy,
            identifier=id,
        )

        product.nonmetal_bonds = row[product_key+'_bonds_nometal']
            
        # set bond mapping for one product w/only one product
        for ind, i in enumerate(list(product.bonds.keys())):
            try:
                key_index = list(reactant_list[0].bonds.keys()).index(i)
                dict_prod[ind] = key_index
            except:
                pass
        bond_map = [dict_prod]

        # atom mapping - order is preserved
        dict_temp = {}
        for i in range(len(species_products_full)):
            dict_temp[i] = i
        mapping = [dict_temp]
        product_list.append(product)

    if fail == 0 and product_list != [] and reactant_list != 0:
        id = [i for i in row["reaction_id"].split("-")]
        id = int(id[0] + id[1] + id[2])
        broken_bond = None

        if(reverse_rxn):
            if row["bonds_formed"] != []:
                broken_bond = row["bonds_formed"][0]
        else:
            if row["bonds_broken"] != []:
                broken_bond = row["bonds_broken"][0]
        if(reverse_rxn):
            value = row['transition_state_energy'] - row['product_energy']
            #value = row['reactant_energy'] - row['product_energy']

        else: 
            #value = row['product_energy'] - row['reactant_energy']
            value = row["dE_barrier"]

        rxn = Reaction(
            reactants=reactant_list,
            products=product_list,
            free_energy=value,
            broken_bond=broken_bond,
            identifier=id,
        )

        rxn.set_atom_mapping(mapping)
        rxn._bond_mapping_by_int_index = bond_map

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
    reactant_key = 'reactant'
    product_key = 'product'
    
    reverse_rxn = False
    if(row['bonds_broken'] == [] and row['bonds_formed'] != []):
        reverse_rxn = True
        reactant_key = 'product'
        product_key = 'reactant'
    
    if(row['bonds_broken'] == [] and row['bonds_formed'] == []):
        return rxn

    species_reactant = [
        int_atom(i["name"]) for i in row[reactant_key+"_molecule_graph"]["molecule"]["sites"]
    ]
    species_products_full = [
        int_atom(i["name"]) for i in row[product_key+"_molecule_graph"]["molecule"]["sites"]
    ]
    coords_reactant = [
        i["xyz"] for i in row[reactant_key+"_molecule_graph"]["molecule"]["sites"]
    ]
    coords_products_full = [
        i["xyz"] for i in row[product_key+"_molecule_graph"]["molecule"]["sites"]
    ]

    charge = row["charge"]
    id = str(row[reactant_key+"_id"])
    free_energy = row[product_key+"_free_energy"]

    reactant_mol = xyz2mol(
        atoms=species_reactant,
        coordinates=coords_reactant,
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
    for i in row[product_key+"_bonds"]:
        G.add_edge(i[0], i[1])  # rdkit bonds are a subset of user-defined bonds
    sub_graphs = [G.subgraph(c) for c in nx.connected_components(G)]
    id = str(row[product_key+"_id"])

    # still no handling for rxns A --> B + C +....
    if len(sub_graphs) > 2:
        pass #print("cannot handle three or more products")
    # handle A --> B + C
    elif len(sub_graphs) == 2:
        mapping, mol_prod = [], []
        for sg in sub_graphs:

            coords_products, species_products, bond_reindex_list = [], [], []
            nodes = list(sg.nodes())
            bonds = list(sg.edges())

            # finds bonds mapped to subgraphs
            for origin_bond_ind in row[product_key+"_bonds"]:
                # check if root to edge is in node list for subgraph
                check = any(item in origin_bond_ind for item in nodes)
                if check:
                    bond_orig = nodes.index(origin_bond_ind[0])
                    bond_targ = nodes.index(origin_bond_ind[1])
                    bond_reindex_list.append([bond_orig, bond_targ])

            for site in nodes:
                species_products.append(
                    int_atom(
                        row[product_key+"_molecule_graph"]["molecule"]["sites"][site]["name"]
                    )
                )
                coords_products.append(
                    row[product_key+"_molecule_graph"]["molecule"]["sites"][site]["xyz"]
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
        if(reverse_rxn):
            broken_bond = row["bonds_formed"][0]

        if(reverse_rxn):
            value = row['transition_state_energy'] - row['product_energy']
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


def create_struct_label_dataset_bond_based_regression(filename, out_file):
    """
    Processes json file from emmet to use in training bondnet

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for training
    """
    path_mg_data = "../../home/santiagovargas/Documents/Dataset/mg_dataset/"
    path_json = path_mg_data + "20220613_reaction_data.json"
    mg_df = pd.read_json(path_json)
    reactions = []

    start_time = time.perf_counter()

    # pebble implementation
    rxn_raw = []
    with ProcessPool(max_workers=12, max_tasks=10) as pool:
        for _, row in mg_df.head(50).iterrows():
            future = pool.schedule(process_species, args=[row], timeout=30)
            rxn_raw.append(future.result())
    print(rxn_raw)
    finish_time = time.perf_counter()
    # pool.close()
    # pool.join()

    print(f"Program finished in {finish_time-start_time} seconds")

    for rxn_temp in rxn_raw:
        if not isinstance(rxn_temp, list):
            # if(reactions.)
            reactions.append(rxn_temp)

    print("number of rxns counted: " + str(len(reactions)))
    extractor = ReactionCollection(reactions)

    # works
    extractor.create_struct_label_dataset_bond_based_regression(
        struct_file=path_mg_data + "mg_struct_bond_rgrn.sdf",
        label_file=path_mg_data + "mg_label_bond_rgrn.yaml",
        feature_file=path_mg_data + "mg_feature_bond_rgrn.yaml",
        group_mode="charge_0",
    )


def create_struct_label_dataset_reaction_network(filename, out_file):
    """
    Processes json file from emmet to use in training bondnet

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for trianing
    """
    path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"
    path_json = path_mg_data + "20220613_reaction_data.json"
    mg_df = pd.read_json(path_json)
    reactions = []

    start_time = time.perf_counter()
    # pebble implementation
    rxn_raw = []
    with ProcessPool(max_workers=12, max_tasks=10) as pool:
        for _, row in mg_df.head(50).iterrows():
            process_species = process_species_rdkit
            future = pool.schedule(process_species, args=[row], timeout=30)
            future.add_done_callback(task_done)
            try:
                rxn_raw.append(future.result())
            except:
                pass
    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
    for rxn_temp in rxn_raw:
        if not isinstance(rxn_temp, list):
            reactions.append(rxn_temp)

    print("number of rxns counted: " + str(len(reactions)))
    # molecules = get_molecules_from_reactions(reactions)
    extractor = ReactionCollection(reactions)

    # works
    extractor.create_struct_label_dataset_reaction_based_regression(
        struct_file=path_mg_data + "mg_struct_bond_rgrn.sdf",
        label_file=path_mg_data + "mg_label_bond_rgrn.yaml",
        feature_file=path_mg_data + "mg_feature_bond_rgrn.yaml",
        group_mode="charge_0",
    )


def create_reaction_network_files(filename, out_file, classifier=False):
    """
    Processes json file from emmet to use in training bondnet

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for trianing
    """
    path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"
    path_json = path_mg_data + "20220613_reaction_data.json"
    mg_df = pd.read_json(path_json)
    reactions = []

    start_time = time.perf_counter()

    rxn_raw = []
    with ProcessPool(max_workers=12, max_tasks=10) as pool:
        for _, row in mg_df.head(100).iterrows():
            # process_species = process_species_rdkit
            future = pool.schedule(process_species_rdkit, args=[row, True], timeout=30)
            future.add_done_callback(task_done)
            try:
                rxn_raw.append(future.result())
            except:
                pass
    finish_time = time.perf_counter()
    print("rxn raw len: {}".format(int(len(rxn_raw))))
    print(f"Program finished in {finish_time-start_time} seconds")
    fail_default, fail_count, fail_sdf_map, fail_prod_len = 0, 0, 0, 0
    for rxn_temp in rxn_raw:
        if not isinstance(rxn_temp, list):  # bunch of stuff is being filtered here
            try:
                rxn_temp.get_broken_bond()
                try:
                    bond_map = rxn_temp.bond_mapping_by_sdf_int_index()
                    rxn_temp._bond_mapping_by_int_index = bond_map

                    reactant_bond_count = int(
                        len(rxn_temp.reactants[0].rdkit_mol.GetBonds())
                    )
                    prod_bond_count = 0
                    for i in rxn_temp.products:
                        prod_bond_count += int(len(i.rdkit_mol.GetBonds()))
                    if reactant_bond_count < prod_bond_count:
                        fail_prod_len += 1
                    else:
                        reactions.append(rxn_temp)
                except:
                    fail_sdf_map += 1
            except:
                fail_count += 1
        else:
            fail_default += 1

    print(".............failures.............")
    print("reactions len: {}".format(int(len(reactions))))
    print("bond break fail count: \t\t{}".format(fail_count))
    print("default fail count: \t\t{}".format(fail_default))
    print("sdf map fail count: \t\t{}".format(fail_sdf_map))
    print("product bond fail count: \t{}".format(fail_prod_len))

    extractor = ReactionCollection(reactions)
    # works
    if classifier:
        (
            all_mols,
            all_labels,
            features,
        ) = extractor.create_struct_label_dataset_reaction_based_regression_alt(
            struct_file=path_mg_data + "mg_struct_bond_rgrn_classify.sdf",
            label_file=path_mg_data + "mg_label_bond_rgrn_classify.yaml",
            feature_file=path_mg_data + "mg_feature_bond_rgrn_classify.yaml",
            group_mode="charge_0",
        )

    else:
        (
            all_mols,
            all_labels,
            features,
        ) = extractor.create_struct_label_dataset_reaction_based_regression_alt(
            struct_file=path_mg_data + "mg_struct_bond_rgrn.sdf",
            label_file=path_mg_data + "mg_label_bond_rgrn.yaml",
            feature_file=path_mg_data + "mg_feature_bond_rgrn.yaml",
            group_mode="charge_0",
        )
    
    return all_mols, all_labels, features


def create_reaction_network_files_and_valid_rows(filename, out_file, bond_map_filter = True):
    """
    Processes json file from emmet to use in training bondnet

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for trianing
        bond_map_filter: true uses filter with sdf
    """
    path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"
    path_json = path_mg_data + "20220613_reaction_data.json"
    mg_df = pd.read_json(path_json) 

    start_time = time.perf_counter()
    reactions, ind_val, rxn_raw, ind_final = [], [], [], []
    with ProcessPool(max_workers=12, max_tasks=10) as pool:
        #for ind, row in mg_df.head(100).iterrows():
        for ind, row in mg_df.iterrows(): 
            # process_species = process_species_rdkit
            future = pool.schedule(process_species_graph, args=[row, True], timeout=30)
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
                rxn_temp.get_broken_bond()
                if(bond_map_filter):
                    try:
                        bond_map = rxn_temp.bond_mapping_by_int_index()

                        rxn_temp._bond_mapping_by_int_index = bond_map

                        reactant_bond_count = int(
                            len(rxn_temp.reactants[0].rdkit_mol.GetBonds())
                        ) # here we're using rdkit still
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
                        rxn_temp._bond_mapping_by_int_index = bond_map
                        reactant_bond_count = int(
                            len(rxn_temp.reactants[0].bonds)
                        ) # here we're using rdkit still
                        prod_bond_count = 0
                        for i in rxn_temp.products:
                            prod_bond_count += int(len(i.bonds))
                        if reactant_bond_count < prod_bond_count:
                            fail_prod_len += 1
                        else:
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

    extractor = ReactionCollection(reactions)
    (
        all_mols,
        all_labels,
        features,
    
    ) = extractor.create_struct_label_dataset_reaction_based_regression_alt(
        struct_file=path_mg_data + "mg_struct_bond_rgrn_classify.sdf",
        label_file=path_mg_data + "mg_label_bond_rgrn_classify.yaml",
        feature_file=path_mg_data + "mg_feature_bond_rgrn_classify.yaml",
        group_mode="charge_0",
        sdf_mapping=False
    )

    return all_mols, all_labels, features, mg_df.iloc[ind_final]


def process_data():
    # create_struct_label_dataset_reaction_network(filename='', out_file='./')
    # create_struct_label_dataset_bond_based_regression(filename='', out_file='./')
    all_mols, all_labels, features = create_reaction_network_files(
        filename="", out_file="./"
    )
    return all_mols, all_labels, features


if __name__ == "__main__":
    process_data()
