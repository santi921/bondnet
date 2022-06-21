import logging
import multiprocessing
import networkx as nx 
import pandas as pd
from io import StringIO
import sys
from rdkit import Chem
Chem.WrapLogs()
from tqdm import tqdm
from bondnet.core.molwrapper import create_wrapper_mol_from_atoms_and_bonds
from bondnet.core.reaction import Reaction
from bondnet.core.rdmol import smiles_to_rdkit_mol, RdkitMolCreationError
from bondnet.core.molwrapper import rdkit_mol_to_wrapper_mol
from bondnet.core.reaction import Reaction
from bondnet.core.reaction_collection import (
    ReactionCollection,
    get_molecules_from_reactions,
    get_atom_bond_mapping,
)
from bondnet.utils import to_path, int_atom, xyz2mol

def process_species(row):
    '''
    Takes a row and processes the products/reactants 

    Args: 
        row: the row (series) pandas object
        
    Returns: 
        mol_list: a list of MolecularWrapper object(s) for product(s) or reactant
    '''
    fail = 0 
    rxn, reactant_list, product_list = [], [], []
    species_reactant = [int_atom(i['name']) for i in row['reactant_molecule_graph']["molecule"]["sites"]]
    species_products = [int_atom(i['name']) for i in row['product_molecule_graph']["molecule"]["sites"]]
    coords_reactant = [i["xyz"] for i in row['reactant_molecule_graph']["molecule"]["sites"]]
    coords_products = [i["xyz"] for i in row['product_molecule_graph']["molecule"]["sites"]]
    charge = row['charge']
    id = str(row["reactant_id"])
    free_energy = row['product_free_energy']

    #try:
    reactant_mol = xyz2mol(atoms = species_reactant, 
        coordinates = coords_reactant, 
        charge = charge,
        )
    reactant_wrapper = rdkit_mol_to_wrapper_mol(reactant_mol[0],
        charge = charge,
        free_energy=free_energy, 
        identifier = id
        )
    reactant_list.append(reactant_wrapper)
    #except: fail+=1

    # handle products
    # check subgraphs first
    num_nodes = 0
    for i in row["composition"].items():
        num_nodes += int(i[-1])
    G = nx.Graph()
    G.add_nodes_from([int(i) for i in range(num_nodes)])
    for i in row["product_bonds"]: G.add_edge(i[0], i[1])
    sub_graphs = [G.subgraph(c) for c in nx.connected_components(G)]

    # still no handling for rxns A --> B + C +....
    if(len(sub_graphs) > 2): pass
    # handle A --> B + C
    elif(len(sub_graphs) == 2):
        for sg in sub_graphs:  
            nodes = list(sg.nodes())
            bonds = list(sg.edges())
            bond_reindex_list = []
            for origin_bond_ind in row["product_bonds"]:
                check = any(item in origin_bond_ind for item in nodes)
                if(check): 
                    bond_orig = nodes.index(origin_bond_ind[0])
                    bond_targ = nodes.index(origin_bond_ind[1])
                    bond_reindex_list.append([bond_orig, bond_targ])
            species_products = []
            coords_products = []

            for site in nodes:
                species_products.append(int_atom(row['product_molecule_graph']["molecule"]["sites"][site]["name"]))
                coords_products.append(row['product_molecule_graph']["molecule"]["sites"][site]["xyz"])
                #species_products.append(int_atom(i['name'])) for i in 
                #[i["xyz"] for i in row['product_molecule_graph']["molecule"]["sites"]]
            #species_products = [int_atom(i['name']) for i in row['product_molecule_graph']["molecule"]["sites"]]
            #coords_products = [i["xyz"] for i in row['product_molecule_graph']["molecule"]["sites"]]
            mol = xyz2mol(atoms = species_products, 
                coordinates = coords_products, 
                charge = charge,
                )[0]

            product = rdkit_mol_to_wrapper_mol(mol,
                charge = charge,
                free_energy=free_energy, 
                identifier = id
            )
            product_list.append(product)


    else: 
        mol = xyz2mol(atoms = species_products, 
            coordinates = coords_products, 
            charge = charge,
            )[0]
        product = rdkit_mol_to_wrapper_mol(mol,
            charge = charge,
            free_energy=free_energy, 
            identifier = id
        )
        product_list.append(product)

    if(fail == 0 and product_list != [] and reactant_list != 0):
        id = [i for i in row["reaction_id"].split("-")]
        id = int(id[0] + id[1] + id[2])
            
        rxn = Reaction(
            reactants = reactant_list, 
            products=product_list, 
            free_energy= row["dE_barrier"],
            broken_bond = None,
            identifier = id
            )
    print(rxn)
    return rxn

def create_struct_label_dataset_bond_based_regression(filename, out_file):
    '''
    Processes json file from emmet to use in training bondnet 

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for trianing
    '''
    path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"
    path_json = path_mg_data + "20220613_reaction_data.json"
    mg_df = pd.read_json(path_json)

    error = 0 
    rdkit_failure = 0 
    single_product_count = 0 
    two_product_count = 0
    three_prouct_count = 0
    no_bond_delta = 0 
    failed_wrapper_reactant = 0
    reactions = []
    # reaction class can handle no id -- try that??
    for index, row in tqdm(mg_df.head(100).iterrows()):
        rxn = process_species(row)
        if(not isinstance(rxn, list)):
            reactions.append(rxn)
        
    print("number of rxns counted: "+str(len(reactions)))
    molecules = get_molecules_from_reactions(reactions)
    print(len(molecules))
    extractor = ReactionCollection(reactions)

    #atom_mp, bond_mp = get_atom_bond_mapping(extractor)
    #failed, succ = 0, 0
    #for i in reactions: 
    #    try:
    #        i.get_broken_bond()
    #        succ+=1
    #    except:
    #        failed += 1
    #print("failed: \t" + str(failed))
    #print("success: \t" + str(succ))

    extractor.create_regression_dataset_reaction_network_simple(
        struct_file=path_mg_data + "mg_struct_bond_rgrn.sdf",
        label_file=path_mg_data + "mg_label_bond_rgrn.yaml",
        feature_file=path_mg_data + "mg_feature_bond_rgrn.yaml",
    )
    #extractor.create_struct_label_dataset_bond_based_regression(
    #    struct_file=path_mg_data + "mg_struct_bond_rgrn.sdf",
    #    label_file=path_mg_data + "mg_label_bond_rgrn.yaml",
    #    feature_file=path_mg_data + "mg_feature_bond_rgrn.yaml",
    #)

        
if __name__ == "__main__":
    # plot_zinc_mols()
    #mg_create_struct_label_dataset_reaction_network_based_regression()
    create_struct_label_dataset_bond_based_regression(filename='', out_file='./')
