import os
import math
import dgl
import lmdb
import pickle
import glob
import tempfile
import numpy as np 

from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from torch.utils.data import random_split

from bondnet.dataset.utils import divide_to_list
from bondnet.data.dataset import LmdbBaseDataset, LmdbMoleculeDataset, LmdbReactionDataset


scalar = 1 / 1024


def TransformMol(data_object):
    serialized_graph = data_object['molecule_graph']
    dgl_graph = load_dgl_graph_from_serialized(serialized_graph)
    data_object["molecule_graph"] = dgl_graph
    return data_object

def TransformReaction(data_object):
    serialized_graph = data_object['reaction_graph']
    dgl_graph = load_dgl_graph_from_serialized(serialized_graph)
    data_object["reaction_graph"] = dgl_graph
    return data_object


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

    #mp_args = [
    #    (db_paths[i], dataset_chunked[i], global_keys, i) for i in range(num_workers)
    #]

    #pool = mp.Pool(num_workers)
    #pool.map(write2moleculelmdb, mp_args)
    #pool.close()
    
    #merge_lmdbs(db_paths, lmdb_dir, lmdb_name, type_lmdb="molecule")
    #merge_lmdbs(db_paths, lmdb_dir, lmdb_name)
    #cleanup_lmdb_files(lmdb_dir, "_tmp_molecule_data*")
    #db_path, samples, global_keys = mp_args
    #Samples: [mol_indices, dgl_graph, pmg]
    #Global_keys: [charge, ring_sizes, elements.]
    #Pid: i_th process

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
        #"reaction_graph",
        #"reaction_feature",
        "reaction_molecule_info",
        "label",
        "reverse_label",
        "extra_info",
        #"has_bonds", 
        "mappings"
    ]


    dataset = [
        {k: v for k, v in zip(key_template, values)}
        for values in zip(
            indices,
            #empty_reaction_graphs,
            #empty_reaction_fts,
            reaction_molecule_info,
            labels,
            reverse_labels,
            extra_info,
            #has_bonds, 
            mappings
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

