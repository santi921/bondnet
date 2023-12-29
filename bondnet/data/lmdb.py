import lmdb
import pickle
from tqdm import tqdm
import glob
import os
from pathlib import Path
import multiprocessing as mp
from torch.utils.data import random_split
from torch.utils.data import Dataset
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
    """
    merge lmdb files and reordering indexes.
    """
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
    #merge_lmdbs(db_paths, lmdb_dir, lmdb_name, type_lmdb="molecule")
    merge_lmdbs(db_paths, lmdb_dir, lmdb_name)
    
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
    #merge_lmdbs(db_paths, lmdb_dir, lmdb_name, type_lmdb="reaction")
    merge_lmdbs(db_paths, lmdb_dir, lmdb_name)
    cleanup_lmdb_files(lmdb_dir, "_tmp_reaction_data*")


class LmdbBaseDataset(Dataset):

    """
    Dataset class to
    1. write Reaction networks objecs to lmdb
    2. load lmdb files
    """

    def __init__(self, config, transform=None):
        super(LmdbBaseDataset, self).__init__()

        self.config = config
        self.path = Path(self.config["src"])

        # Get metadata in case
        # self.metadata_path = self.path.parent / "metadata.npz"
        self.env = self.connect_db(self.path)

        # If "length" encoded as ascii is present, use that
        # If there are additional properties, there must be length.
        length_entry = self.env.begin().get("length".encode("ascii"))
        if length_entry is not None:
            num_entries = pickle.loads(length_entry)
        else:
            # Get the number of stores data from the number of entries
            # in the LMDB
            num_entries = self.env.stat()["entries"]

        self._keys = list(range(num_entries))
        self.num_samples = num_entries

        # Get portion of total dataset
        self.sharded = False
        if "shard" in self.config and "total_shards" in self.config:
            self.sharded = True
            self.indices = range(self.num_samples)
            # split all available indices into 'total_shards' bins
            self.shards = np.array_split(
                self.indices, self.config.get("total_shards", 1)
            )
            # limit each process to see a subset of data based off defined shard
            self.available_indices = self.shards[self.config.get("shard", 0)]
            self.num_samples = len(self.available_indices)

        # TODO
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # if sharding, remap idx to appropriate idx of the sharded set
        if self.sharded:
            idx = self.available_indices[idx]

        #!CHECK, _keys should be less then total numbers of keys as there are more properties.
        datapoint_pickled = self.env.begin().get(f"{self._keys[idx]}".encode("ascii"))

        data_object = pickle.loads(datapoint_pickled)

        # TODO
        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=False,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()

    def get_metadata(self, num_samples=100):
        pass

class LmdbMoleculeDataset(LmdbBaseDataset):
    def __init__(self, config, transform=None):
        super(LmdbMoleculeDataset, self).__init__(config=config, transform=transform)

    @property
    def charges(self):
        charges = self.env.begin().get("charges".encode("ascii"))
        return pickle.loads(charges)

    @property
    def ring_sizes(self):
        ring_sizes = self.env.begin().get("ring_sizes".encode("ascii"))
        return pickle.loads(ring_sizes)

    @property
    def elements(self):
        elements = self.env.begin().get("elements".encode("ascii"))
        return pickle.loads(elements)

    @property
    def feature_info(self):
        feature_info = self.env.begin().get("feature_info".encode("ascii"))
        return pickle.loads(feature_info)

class LmdbReactionDataset(LmdbBaseDataset):
    def __init__(self, config, transform=None):
        super(LmdbReactionDataset, self).__init__(config=config, transform=transform)

    @property
    def dtype(self):
        dtype = self.env.begin().get("dtype".encode("ascii"))
        return  pickle.loads(dtype)
            
    @property
    def feature_size(self):
        feature_size = self.env.begin().get("feature_size".encode("ascii"))
        return pickle.loads(feature_size)

    @property
    def feature_name(self):
        feature_name = self.env.begin().get("feature_name".encode("ascii"))
        return pickle.loads(feature_name)
    
    @property
    def mean(self):
        mean = self.env.begin().get("mean".encode("ascii"))
        return pickle.loads(mean)
    
    @property
    def std(self):
        std = self.env.begin().get("std".encode("ascii"))
        return pickle.loads(std)
