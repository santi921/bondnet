import lmdb
import pickle
from tqdm import tqdm
import glob
import os
import multiprocessing as mp
from torch.utils.data import random_split

from bondnet.dataset.utils import divide_to_list

scalar = 1 / 1024


def write2reactionlmdb(mp_args):
    db_path, samples, global_keys, pid = mp_args
    # Samples: [mol_indices, dgl_graph, pmg]
    # Global_keys: [charge, ring_sizes, elements.]
    # Pid: i_th process
    # print("openining lmdb")
    db = lmdb.open(
        db_path,
        map_size=int(1099511627776 * scalar),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    pbar = tqdm(
        total=len(samples),
        position=pid,
        desc=f"Worker {pid}: Writing LMDBs",
    )

    # write  samples.
    idx = 0
    for sample in samples:
        txn = db.begin(write=True)
        txn.put(
            f"{idx}".encode("ascii"),
            pickle.dumps(sample, protocol=-1),
        )
        idx += 1
        pbar.update(1)
        txn.commit()

    # write properties.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(samples), protocol=-1))
    txn.commit()

    if len(global_keys) != 0:
        for key, value in global_keys.items():
            txn = db.begin(write=True)
            txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
            txn.commit()

    db.sync()
    db.close()


def write2moleculelmdb(mp_args):
    """
    Write samples to lmdb file.
    Takes:
        mp_args: tuple of (db_path, samples, global_keys, pid)
    """

    db_path, samples, global_keys, pid = mp_args
    # Samples: [mol_indices, dgl_graph, pmg]
    # Global_keys: [charge, ring_sizes, elements.]
    # Pid: i_th process
    # print("openining lmdb")
    db = lmdb.open(
        db_path,
        map_size=int(1099511627776 * scalar),
        subdir=False,
        meminit=False,
        map_async=True,
    )
    pbar = tqdm(
        total=len(samples),
        position=pid,
        desc=f"Worker {pid}: Writing LMDBs",
    )
    # print("writing lmdb")
    # write samples
    for sample in samples:
        sample_index = sample["molecule_index"]
        txn = db.begin(write=True)
        txn.put(
            # let index of molecule identical to index of sample
            f"{sample_index}".encode("ascii"),
            pickle.dumps(sample, protocol=-1),
        )
        pbar.update(1)
        txn.commit()
    # print("writing properties")
    # write properties.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(samples), protocol=-1))
    txn.commit()

    for key, value in global_keys.items():
        txn = db.begin(write=True)
        txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
        txn.commit()

    db.sync()
    db.close()


def merge_lmdbs(db_paths, out_path, output_file, type_lmdb):
    """_summary_

    Args:
        db_paths (_type_): _description_
        out_path (_type_): _description_
        output_file (_type_): name of new lmdb file
        type_lmdb (_type_): _description_

    Raises:
        KeyError: _description_
    """

    env_out = lmdb.open(
        os.path.join(out_path, output_file),
        map_size=int(1099511627776 * scalar),
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

        #! Idx are indices of molecule.  Idx are reordered for reactions.
        # should set indexes so that properties do not written down as well.
        with env_out.begin(write=True) as txn_out, env_in.begin(write=False) as txn_in:
            length = txn_in.get("length".encode("ascii"))
            cursor = txn_in.cursor()
            for key, value in cursor:
                if type_lmdb == "molecule":
                    try:
                        int_key = key.decode("ascii")
                        int(key.decode("ascii"))
                        txn_out.put(
                            f"{int_key}".encode("ascii"),
                            value,
                        )
                        idx += 1  # global doesn't account for idx
                    except ValueError:
                        # write properties
                        txn_out.put(key, value)

                # write indexed samples
                elif type_lmdb == "reaction":
                    try:
                        int(key.decode("ascii"))
                        txn_out.put(
                            f"{idx}".encode("ascii"),
                            value,
                        )
                        idx += 1  # global doesn't account for idx
                    except ValueError:
                        # write properties
                        txn_out.put(key, value)
                else:
                    raise KeyError(f"This type of {type_lmdb} is not supported")
        env_in.close()

    # update length
    txn_out = env_out.begin(write=True)
    txn_out.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn_out.commit()

    env_out.sync()
    env_out.close()


def cleanup_lmdb_files(directory, pattern):
    """
    Cleans up files matching the given pattern in the specified directory.
    """
    file_list = glob.glob(os.path.join(directory, pattern))

    for file_path in file_list:
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except OSError as e:
            print(f"Error deleting file: {file_path}. {str(e)}")


def parallel2moleculelmdb(
    indices,
    graphs,
    pmgs,
    charges,
    ring_sizes,
    elements,
    feature_info,
    num_workers,
    lmdb_dir,
    lmdb_name,
):
    os.makedirs(lmdb_dir, exist_ok=True)

    db_paths = [
        os.path.join(lmdb_dir, "_tmp_molecule_data.%04d.lmdb" % i)
        for i in range(num_workers)
    ]

    key_tempalte = ["molecule_index", "molecule_graph", "molecule_wrapper"]
    dataset = [
        {k: v for k, v in zip(key_tempalte, values)}
        for values in zip(indices, graphs, pmgs)
    ]

    dataset_chunked = random_split(dataset, divide_to_list(len(indices), num_workers))

    global_keys = {
        "charges": charges,
        "ring_sizes": ring_sizes,
        "elements": elements,
        "feature_info": feature_info,
    }

    mp_args = [
        (db_paths[i], dataset_chunked[i], global_keys, i) for i in range(num_workers)
    ]

    pool = mp.Pool(num_workers)
    pool.map(write2moleculelmdb, mp_args)
    pool.close()
    merge_lmdbs(db_paths, lmdb_dir, lmdb_name, type_lmdb="molecule")
    cleanup_lmdb_files(lmdb_dir, "_tmp_molecule_data*")


def parallel2reactionlmdb(
    indices,
    empty_reaction_graphs,
    empty_reaction_fts,
    reaction_molecule_info,
    labels,
    reverse_labels,
    extra_info,
    num_workers,
    lmdb_dir,
    lmdb_name,
):
    os.makedirs(lmdb_dir, exist_ok=True)

    db_paths = [
        os.path.join(lmdb_dir, "_tmp_reaction_data.%04d.lmdb" % i)
        for i in range(num_workers)
    ]

    key_tempalte = [
        "reaction_index",
        "reaction_graph",
        "reaction_feature",
        "reaction_molecule_info",
        "label",
        "reverse_label",
        "extra_info",
    ]
    # TODO: add global keys for scaling info to dataset
    """global_keys = {
        "label_scaler_mean": ,
        "label_scaler_std": ,
    }"""

    dataset = [
        {k: v for k, v in zip(key_tempalte, values)}
        for values in zip(
            indices,
            empty_reaction_graphs,
            empty_reaction_fts,
            reaction_molecule_info,
            labels,
            reverse_labels,
            extra_info,
        )
    ]

    dataset_chunked = random_split(dataset, divide_to_list(len(indices), num_workers))
    global_keys = {}
    mp_args = [
        (db_paths[i], dataset_chunked[i], global_keys, i) for i in range(num_workers)
    ]

    pool = mp.Pool(num_workers)
    pool.map(write2reactionlmdb, mp_args)
    pool.close()
    merge_lmdbs(db_paths, lmdb_dir, lmdb_name, type_lmdb="reaction")
    cleanup_lmdb_files(lmdb_dir, "_tmp_reaction_data*")
