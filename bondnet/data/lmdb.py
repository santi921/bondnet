import os
import dgl
import lmdb
import pickle
import glob
import tempfile
import numpy as np 
from copy import deepcopy
from tqdm import tqdm
#from pathlib import Path
#import math
#import multiprocessing as mp
#from torch.utils.data import random_split
#from bondnet.dataset.utils import divide_to_list
from bondnet.data.dataset import LmdbBaseDataset, LmdbMoleculeDataset, LmdbReactionDataset

from bondnet.dataset.utils import (
    clean,
    clean_op
)
from bondnet.data.utils import find_rings, create_rxn_graph, construct_rxn_graph_empty


scalar = 1 


def TransformMol(data_object):
    serialized_graph = data_object['molecule_graph']
    # check if serialized_graph is DGL graph or if it is chunk 
    if isinstance(serialized_graph, dgl.DGLGraph):
        return data_object
    elif isinstance(serialized_graph, dgl.DGLHeteroGraph):
        return data_object

    dgl_graph = load_dgl_graph_from_serialized(serialized_graph)
    data_object["molecule_graph"] = dgl_graph
    return data_object



def TransformReaction(data_object):
    serialized_graph = data_object['reaction_graph']
    
    if isinstance(serialized_graph, dgl.DGLGraph):
        return data_object
    elif isinstance(serialized_graph, dgl.DGLHeteroGraph):
        return data_object

    dgl_graph = load_dgl_graph_from_serialized(serialized_graph)
    data_object["reaction_graph"] = dgl_graph
    return data_object



def construct_lmdb_and_save_reaction_dataset(dataset, lmdb_dir, workers=8):
    # List of Molecules
    dgl_graphs = []
    dgl_graphs_serialized = []
    pmg_objects = []
    molecule_ind_list = []
    charge_set = set()
    ring_size_set = set()
    element_set = set()
    feature_size = dataset._feature_size
    feature_name = dataset._feature_name
    feature_scaler_mean = dataset._feature_scaler_mean
    feature_scaler_std = dataset._feature_scaler_std
    label_scaler_mean = dataset._label_scaler_mean
    label_scaler_std = dataset._label_scaler_std
    dtype = dataset.dtype


    for ind, molecule_in_rxn_network in enumerate(
        dataset.molecules
    ):

        formula = molecule_in_rxn_network.pymatgen_mol.composition.formula.split()
        elements = [clean(x) for x in formula]
        atom_num = np.sum(np.array([int(clean_op(x)) for x in formula]))
        
        charge = molecule_in_rxn_network.pymatgen_mol.charge
        
        bond_list = [
            [i[0], i[1]] for i in molecule_in_rxn_network.mol_graph.graph.edges
        ]
        cycles = find_rings(atom_num, bond_list, edges=False)
        ring_len_list = [len(i) for i in cycles]
        graph_mol_in_rxn_network = dataset.graphs[ind]        
        
        element_set.update(elements)
        ring_size_set.update(ring_len_list)
        charge_set.add(charge)
        pmg_objects.append(molecule_in_rxn_network.pymatgen_mol)
        molecule_ind_list.append(ind)
        dgl_graphs_serialized.append(serialize_dgl_graph(graph_mol_in_rxn_network))
        dgl_graphs.append(graph_mol_in_rxn_network)



    batched_graph = dgl.batch(dgl_graphs)
    feats = batched_graph.ndata["ft"]
    for nt, ft in feats.items():
        batched_graph.nodes[nt].data.update({"ft": ft})
    graphs = dgl.unbatch(batched_graph)

    extra_info = []
    reaction_molecule_info = []
    label_list = []
    reverse_list = []
    has_bonds_list = []
    mappings_list = []
    empty_reaction_graphs = []
    empty_reaction_fts = []
    reaction_indicies = []

    global_dict = {
        "feature_size": feature_size,
        "mean": label_scaler_mean,
        "std": label_scaler_std,
        "feature_name": feature_name,
        "feature_scaler_mean": feature_scaler_mean,
        "feature_scaler_std": feature_scaler_std,
        "dtype": dtype
    }

        
    for ind, rxn in enumerate(dataset.reactions):
        rxn_copy = deepcopy(rxn)
        
        reactants = [dgl_graphs[i] for i in rxn_copy.reactants]
        products = [dgl_graphs[i] for i in rxn_copy.products]
        
        mappings = {
            "bond_map": rxn_copy.bond_mapping,
            "atom_map": rxn_copy.atom_mapping,
            "total_bonds": rxn_copy.total_bonds,
            "total_atoms": rxn_copy.total_atoms,
            "num_bonds_total": rxn_copy.num_bonds_total,
            "num_atoms_total": rxn_copy.num_atoms_total,
        }
        has_bonds = {
            "reactants": [
                True if len(mp) > 0 else False for mp in rxn_copy.bond_mapping[0]
            ],
            "products": [
                True if len(mp) > 0 else False for mp in rxn_copy.bond_mapping[1]
            ],
        }


        if len(has_bonds["reactants"]) != len(reactants) or len(
            has_bonds["products"]
        ) != len(products):
            print("unequal mapping & graph len")


        molecule_info_temp = {
            "reactants": {
                #"reactants": rxn.reactants,
                "init_reactants": rxn_copy.init_reactants,
                "has_bonds": has_bonds["reactants"]
            },
            "products": {
                #"products": rxn.products,
                "init_products": rxn_copy.init_products,
                "has_bonds": has_bonds["products"]
            },
            #"has_bonds": has_bonds,
            "mappings": mappings,
        }

        empty_graph, empty_fts = construct_rxn_graph_empty(
            mappings=mappings,
            self_loop=True, 
            ret_feats=True
        )

        extra_info.append([])
        # extra_info.append(reaction_in_rxn_network.extra_info)
        label_list.append(dataset.labels[ind]["value"])
        reverse_list.append(dataset.labels[ind]["value_rev"])
        reaction_molecule_info.append(molecule_info_temp)
        has_bonds_list.append(has_bonds) # don't need to save
        reaction_indicies.append(str(rxn.id[0])) # need 
        empty_reaction_graphs.append(empty_graph) # need 
        empty_reaction_fts.append(empty_fts) # potentially source of bugginess
        mappings_list.append(mappings)


    print("...> writing molecules to lmdb")
    print("number of molecules to write: ", len(molecule_ind_list))
    write_molecule_lmdb(
        indices=molecule_ind_list,
        graphs=dgl_graphs_serialized,
        pmgs=pmg_objects,
        charges=charge_set,
        ring_sizes=ring_size_set,
        elements=element_set,
        feature_info={
            "feature_size": feature_size,
            "feature_scaler_mean": feature_scaler_mean,
            "feature_scaler_std": feature_scaler_std,
        },
        #num_workers=1,
        lmdb_dir=lmdb_dir,
        lmdb_name="/molecule.lmdb",
    )
    print("...> writing reactions to lmdb")

    
    write_reaction_lmdb(
        indices=reaction_indicies,
        empty_reaction_graphs=empty_reaction_graphs,
        empty_reaction_fts=empty_reaction_fts,
        reaction_molecule_info=reaction_molecule_info,
        labels=label_list,
        reverse_labels=reverse_list,
        extra_info=extra_info,
        lmdb_dir=lmdb_dir,
        lmdb_name="/reaction.lmdb",
        mappings=mappings_list, 
        has_bonds=has_bonds_list, 
        global_values=global_dict,
    )
    

def serialize_dgl_graph(dgl_graph):
    # import pdb
    # pdb.set_trace()
    # Create a temporary file
    with tempfile.NamedTemporaryFile() as tmpfile:
        # Save the graph to the temporary file

        dgl.save_graphs(tmpfile.name, [dgl_graph])

        # Read the content of the file
        tmpfile.seek(0)
        serialized_data = tmpfile.read()

    return serialized_data


def load_dgl_graph_from_serialized(serialized_graph):
    with tempfile.NamedTemporaryFile(mode='wb', delete=True) as tmpfile:
        tmpfile.write(serialized_graph)
        tmpfile.flush()  # Ensure all data is written

        # Rewind the file to the beginning before reading
        tmpfile.seek(0)

        # Load the graph using the file handle
        graphs, _ = dgl.load_graphs(tmpfile.name)

    return graphs[0]  # Assuming there's only one graph


def write2moleculelmdb(mp_args
    ):
    """
    write molecule lmdb in parallel.
    in species filter, there should be only one thread. no need parallelizations.
    """
    db_path, samples, global_keys, pid = mp_args
    #Samples: [mol_indices, dgl_graph, pmg]
    #Global_keys: [charge, ring_sizes, elements.]
    #Pid: i_th process

    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    pbar = tqdm(
        total=len(samples),
        position=pid,
        desc=f"Worker {pid}: Writing LMDBs",
    )
    #write samples
    for sample in samples:
        sample_index = sample["molecule_index"]
        txn = db.begin(write=True)
        txn.put(
            #let index of molecule identical to index of sample
            f"{sample_index}".encode("ascii"),
            pickle.dumps(sample, protocol=-1),
        )
        pbar.update(1)
        txn.commit()

    #write properties.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(samples), protocol=-1))
    txn.commit()

    for key, value in global_keys.items():
        txn = db.begin(write=True)
        txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
        txn.commit()

    db.sync()
    db.close()


def merge_lmdbs(db_paths, out_path, output_file):
    env_out = lmdb.open(
        os.path.join(out_path, output_file),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    
    
    idx = 0
    for db_path in db_paths:
        env_in = lmdb.open(
            str(db_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        
        #should set indexes so that properties do not writtent down as well.
        with env_out.begin(write=True) as txn_out, env_in.begin(write=False) as txn_in:
            cursor = txn_in.cursor()
            for key, value in cursor:
                #write indexed samples
                #print(key)
                try:
                    int(key.decode("ascii"))
                    txn_out.put(
                    f"{idx}".encode("ascii"),
                    value,
                    )
                    idx+=1
                    #print(idx)
                #write properties
                except ValueError:
                    txn_out.put(
                        key,
                        value
                    )
        env_in.close()
    
    #update length
    txn_out=env_out.begin(write=True)
    txn_out.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn_out.commit()
        
    env_out.sync()
    env_out.close()


def cleanup_lmdb_files(directory, pattern):
    file_list = glob.glob(os.path.join(directory, pattern))

    for file_path in file_list:
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except OSError as e:
            print(f"Error deleting file: {file_path}. {str(e)}")


def write_molecule_lmdb(
    indices,
    graphs,
    pmgs,
    charges,
    ring_sizes,
    elements,
    feature_info,
    #num_workers,
    lmdb_dir,
    lmdb_name,
):
    os.makedirs(lmdb_dir, exist_ok=True)

    #db_paths = [
    #    os.path.join(lmdb_dir, "_tmp_molecule_data.%04d.lmdb" % i)
    #    for i in range(num_workers)
    #]#

    key_template = ["molecule_index", "molecule_graph", "molecule_wrapper"]
    dataset = [
        {k: v for k, v in zip(key_template, values)}
        for values in zip(indices, graphs, pmgs)
    ]

    #dataset_chunked = random_split(dataset, divide_to_list(len(indices), num_workers))

    global_keys = {
        "charges": charges,
        "ring_sizes": ring_sizes,
        "elements": elements,
        "feature_info": feature_info,
    }

    db = lmdb.open(
        lmdb_dir + lmdb_name,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    #write samples
    for sample in dataset:
        sample_index = sample["molecule_index"]
        txn = db.begin(write=True)
        txn.put(
            #let index of molecule identical to index of sample
            f"{sample_index}".encode("ascii"),
            pickle.dumps(sample, protocol=-1),
        )
        txn.commit()

    #write properties.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(dataset), protocol=-1))
    txn.commit()

    for key, value in global_keys.items():
        txn = db.begin(write=True)
        txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
        txn.commit()

    db.sync()
    db.close()


def write_reaction_lmdb(
    indices,
    empty_reaction_graphs,
    empty_reaction_fts,
    reaction_molecule_info,
    labels,
    reverse_labels,
    extra_info,
    lmdb_dir,
    lmdb_name,
    has_bonds, 
    mappings, 
    global_values
):
    os.makedirs(lmdb_dir, exist_ok=True)


    key_template = [
        "reaction_index",
        "reaction_graph",
        "reaction_feature",
        "reaction_molecule_info",
        "label",
        "reverse_label",
        "extra_info",
        "mappings",
        "has_bonds",
    ]


    dataset = [
        {k: v for k, v in zip(key_template, values)}
        for values in zip(
            indices,
            empty_reaction_graphs,
            empty_reaction_fts,
            reaction_molecule_info,
            labels,
            reverse_labels,
            extra_info,
            mappings,
            has_bonds, 
        )
    ]

    
    db_path = lmdb_dir + lmdb_name
    print("len dataset to write: {}".format(len(dataset)))
    #Samples

    # Samples: [mol_indices, dgl_graph, pmg]
    # Global_keys: [charge, ring_sizes, elements.]
    # Pid: i_th process
    print("opening lmdb")
    db = lmdb.open(
        db_path,
        map_size=int(1099511627776 * scalar),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # write  samples.
    idx = 0
    for sample in dataset:
        txn = db.begin(write=True)
        #print(idx)
        txn.put(
            f"{idx}".encode("ascii"),
            pickle.dumps(sample, protocol=-1),
        )
        idx += 1
        txn.commit()

    # write properties.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(dataset), protocol=-1))
    txn.commit()

    if len(global_values) != 0:
        for key, value in global_values.items():
            
            txn = db.begin(write=True)
            txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
            txn.commit()

    db.sync()
    db.close()

