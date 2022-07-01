import time 
import pandas as pd
import networkx as nx
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from bondnet.core.reaction import Reaction
from bondnet.core.molwrapper import rdkit_mol_to_wrapper_mol
from bondnet.core.reaction import Reaction
from bondnet.core.reaction_collection import ReactionCollection
from bondnet.utils import int_atom, xyz2mol

from rdkit import Chem
Chem.WrapLogs()

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
    id = str(row["product_id"])

    # still no handling for rxns A --> B + C +....
    if(len(sub_graphs) > 2): pass
    # handle A --> B + C
    elif(len(sub_graphs) == 2):
        mapping = []
        mol_prod = []
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
            
            mapping_temp = {i:ind for i,ind in enumerate(nodes)}
            mapping.append(mapping_temp)

            mol = xyz2mol(atoms = species_products, 
                coordinates = coords_products, 
                charge = charge,
                )[0]
            mol_prod.append(mol)
            product = rdkit_mol_to_wrapper_mol(mol,charge = charge, free_energy=free_energy, identifier = id)
            product_list.append(product)


    else: 
        mol_prod = xyz2mol(atoms = species_products, 
            coordinates = coords_products, 
            charge = charge,
            )[0]
        product = rdkit_mol_to_wrapper_mol(mol_prod,
            charge = charge,
            free_energy=free_energy, 
            identifier = id
        )
        mapping = [{i:i} for i in range(len(product.species))]
        product_list.append(product)

    if(fail == 0 and product_list != [] and reactant_list != 0):
        id = [i for i in row["reaction_id"].split("-")]
        id = int(id[0] + id[1] + id[2])
        broken_bond = None

        if(row["bonds_broken"] != []):
            broken_bond = row['bonds_broken'][0]
            
        #if(row["bonds_formed"] != [] and broken_bond == None):
        #    broken_bond = row['bonds_formed'][0]
        
        rxn = Reaction(
            reactants = reactant_list, 
            products=product_list, 
            free_energy= row["dE_barrier"],
            broken_bond = broken_bond,
            identifier = id
            )
        rxn.set_atom_mapping(mapping)
        # failing
        # rxn.set_bond_mapping_by_sdf_int_index(mapping_sdf)

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
    '''
    Processes json file from emmet to use in training bondnet 

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for trianing
    '''
    path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"
    path_json = path_mg_data + "20220613_reaction_data.json"
    mg_df = pd.read_json(path_json)
    reactions = []
    #for index, row in tqdm(mg_df.iterrows()):
    #for index, row in tqdm(mg_df.head(1000).iterrows()):
    #    rxn = process_species(row)
    #    if(not isinstance(rxn, list)):
    #        reactions.append(rxn)
    #mg_df.head(1000).iterrows()
    
    #with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    #    rxn_raw = p.map(process_species, mg_df.head(100).iterrows())
    start_time = time.perf_counter()
    # pebble implementation 
    rxn_raw = []
    with ProcessPool(max_workers = 12 , max_tasks =10) as pool:
        for _, row in mg_df.head(50).iterrows():
            future = pool.schedule(process_species, args = [row], timeout = 30)
            #future.add_done_callback(task_done)
            rxn_raw.append(future.result())
    print(rxn_raw)
    finish_time = time.perf_counter()
    #pool.close()
    #pool.join()

    print(f"Program finished in {finish_time-start_time} seconds")
    

    for rxn_temp in rxn_raw:
        if(not isinstance(rxn_temp, list)):
            reactions.append(rxn_temp)
    
    """    mappings = []
    for i in reactions: 
        try:
            mappings.append(get_atom_bond_mapping(i))
            except: pass"""
            
    #print("reactions that pass the mapping test: {}/{}".format(len(mappings), len(reactions)))
    print("number of rxns counted: "+str(len(reactions)))
    #molecules = get_molecules_from_reactions(reactions)
    extractor = ReactionCollection(reactions)

    # works
    extractor.create_struct_label_dataset_bond_based_regression(
        struct_file=path_mg_data + "mg_struct_bond_rgrn.sdf",
        label_file=path_mg_data + "mg_label_bond_rgrn.yaml",
        feature_file=path_mg_data + "mg_feature_bond_rgrn.yaml",
        group_mode='charge_0'
    )
    
def create_struct_label_dataset_reaction_network(filename, out_file):
    '''
    Processes json file from emmet to use in training bondnet 

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for trianing
    '''
    path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"
    path_json = path_mg_data + "20220613_reaction_data.json"
    mg_df = pd.read_json(path_json)
    reactions = []

    start_time = time.perf_counter()
    # pebble implementation 
    rxn_raw = []
    with ProcessPool(max_workers=12 , max_tasks=10) as pool:
        for _, row in mg_df.head(50).iterrows():
            future = pool.schedule(process_species, args = [row], timeout = 30)
            future.add_done_callback(task_done)
            try:
                rxn_raw.append(future.result())
            except:pass
    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
    for rxn_temp in rxn_raw:
        if(not isinstance(rxn_temp, list)):
            reactions.append(rxn_temp)
    
    print("number of rxns counted: "+str(len(reactions)))
    #molecules = get_molecules_from_reactions(reactions)
    extractor = ReactionCollection(reactions)

    # works
    extractor.create_struct_label_dataset_reaction_based_regression(
        struct_file=path_mg_data + "mg_struct_bond_rgrn.sdf",
        label_file=path_mg_data + "mg_label_bond_rgrn.yaml",
        feature_file=path_mg_data + "mg_feature_bond_rgrn.yaml",
        group_mode='charge_0'
    )

def create_reaction_network_files(filename, out_file):
    '''
    Processes json file from emmet to use in training bondnet 

    Args:
        filename: name of the json file to be used to gather data
        out_file: name of folder where to store the three files for trianing
    '''
    path_mg_data = "/home/santiagovargas/Documents/Dataset/mg_dataset/"
    path_json = path_mg_data + "20220613_reaction_data.json"
    mg_df = pd.read_json(path_json)
    reactions = []

    start_time = time.perf_counter()
     
    rxn_raw = []
    with ProcessPool(max_workers=12 , max_tasks=10) as pool:
        for _, row in mg_df.iterrows():
            future = pool.schedule(process_species, args = [row], timeout = 30)
            future.add_done_callback(task_done)t
            try:
                rxn_raw.append(future.result())
            except:pass
    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
    for rxn_temp in rxn_raw:
        if(not isinstance(rxn_temp, list)):
            reactions.append(rxn_temp)
    
    print("number of rxns counted: "+str(len(reactions)))
    extractor = ReactionCollection(reactions)
    # works
    extractor.create_struct_label_dataset_reaction_based_regression_alt(
        struct_file=path_mg_data + "mg_struct_bond_rgrn.sdf",
        label_file=path_mg_data + "mg_label_bond_rgrn.yaml",
        feature_file=path_mg_data + "mg_feature_bond_rgrn.yaml",
        group_mode='charge_0'
    )
        
if __name__ == "__main__":
    #create_struct_label_dataset_reaction_network(filename='', out_file='./')
    #create_struct_label_dataset_bond_based_regression(filename='', out_file='./')
    create_reaction_network_files(filename='', out_file='./')