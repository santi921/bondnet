#import torch, itertools
#import pandas as pd
import numpy as np
#import os
from pathlib import Path
#from collections import defaultdict, OrderedDict
from concurrent.futures import TimeoutError
from tqdm import tqdm
from rdkit import Chem, RDLogger
import pickle
import lmdb
from copy import deepcopy

from torch.utils.data import Dataset
import torch

from bondnet.dataset.generalized import create_reaction_network_files_and_valid_rows
from bondnet.data.reaction_network import ReactionInNetwork, ReactionNetwork
from bondnet.data.transformers import HeteroGraphFeatureStandardScaler, StandardScaler
#from bondnet.data.utils import get_dataset_species, get_hydro_data_functional_groups
from bondnet.utils import to_path, yaml_load, list_split_by_size
#from bondnet.data.utils import create_rxn_graph
#from bondnet.data.lmdb import TransformMol

logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)


def task_done(future):
    try:
        result = future.result()  # blocks until results are ready
    except TimeoutError as error:
        print("Function took longer than %d seconds" % error.args[1])
    except Exception as error:
        print("Function raised %s" % error)
        print(error.traceback)  # traceback of the function


class BaseDataset:
    """
     Base dataset class.

    Args:
     grapher (BaseGraph): grapher object that build different types of graphs:
         `hetero`, `homo_bidirected` and `homo_complete`.
         For hetero graph, atom, bond, and global state are all represented as
         graph nodes. For homo graph, atoms are represented as node and bond are
         represented as graph edges.
     molecules (list or str): rdkit molecules. If a string, it should be the path
         to the sdf file of the molecules.
     labels (list or str): each element is a dict representing the label for a bond,
         molecule or reaction. If a string, it should be the path to the label file.
     extra_features (list or str or None): each element is a dict representing extra
         features provided to the molecules. If a string, it should be the path to the
         feature file. If `None`, features will be calculated only using rdkit.
     feature_transformer (bool): If `True`, standardize the features by subtracting the
         means and then dividing the standard deviations.
     label_transformer (bool): If `True`, standardize the label by subtracting the
         means and then dividing the standard deviations. More explicitly,
         labels are standardized by y' = (y - mean(y))/std(y), the model will be
         trained on this scaled value. However for metric measure (e.g. MAE) we need
         to convert y' back to y, i.e. y = y' * std(y) + mean(y), the model
         prediction is then y^ = y'^ *std(y) + mean(y), where ^ means predictions.
         Then MAE is |y^-y| = |y'^ - y'| *std(y), i.e. we just need to multiple
         standard deviation to get back to the original scale. Similar analysis
         applies to RMSE.
     state_dict_filename (str or None): If `None`, feature mean and std (if
         feature_transformer is True) and label mean and std (if label_transformer is True)
         are computed from the dataset; otherwise, they are read from the file.
    """

    def __init__(
        self,
        grapher,
        molecules,
        labels,
        extra_features=None,
        feature_transformer=True,
        label_transformer=True,
        dtype="float32",
        state_dict_filename=None,
    ):
        if dtype not in ["float32", "float64"]:
            raise ValueError(f"`dtype {dtype}` should be `float32` or `float64`.")

        self.grapher = grapher
        self.molecules = (
            to_path(molecules) if isinstance(molecules, (str, Path)) else molecules
        )
        try:
            self.molecules = [mol.rdkit_mol() for mol in self.molecules]
        except:
            print("molecules already some rdkit object")

        self.raw_labels = to_path(labels) if isinstance(labels, (str, Path)) else labels
        self.extra_features = (
            to_path(extra_features)
            if isinstance(extra_features, (str, Path))
            else extra_features
        )
        self.feature_transformer = feature_transformer
        self.label_transformer = label_transformer
        self.dtype = dtype
        self.state_dict_filename = state_dict_filename

        self.graphs = None
        self.labels = None
        self._feature_size = None
        self._feature_name = None
        self._feature_scaler_mean = None
        self._feature_scaler_std = None
        self._label_scaler_mean = None
        self._label_scaler_std = None
        self._species = None
        self._failed = None

        self._load()

    @property
    def feature_size(self):
        """
        Returns a dict of feature size with node type as the key.
        """
        return self._feature_size

    @property
    def feature_name(self):
        """
        Returns a dict of feature name with node type as the key.
        """
        return self._feature_name

    def get_feature_size(self, ntypes):
        """
        Get feature sizes.

        Args:
              ntypes (list of str): types of nodes.

        Returns:
             list: sizes of features corresponding to note types in `ntypes`.
        """
        size = []
        for nt in ntypes:
            for k in self.feature_size:
                if nt in k:
                    size.append(self.feature_size[k])
        # TODO more checks needed e.g. one node get more than one size
        msg = f"cannot get feature size for nodes: {ntypes}"
        assert len(ntypes) == len(size), msg

        return size

    @property
    def failed(self):
        """
        Whether an entry (molecule, reaction) fails upon converting using rdkit.

        Returns:
            list of bool: each element indicates whether a entry fails. The size of
                this list is the same as the labels, each one corresponds a label in
                the same order.
            None: is this info is not set
        """
        return self._failed

    def state_dict(self):
        d = {
            "feature_size": self._feature_size,
            "feature_name": self._feature_name,
            "feature_scaler_mean": self._feature_scaler_mean,
            "feature_scaler_std": self._feature_scaler_std,
            "label_scaler_mean": self._label_scaler_mean,
            "label_scaler_std": self._label_scaler_std,
            "species": self._species,
        }

        return d

    def load_state_dict(self, d):
        self._feature_size = d["feature_size"]
        self._feature_name = d["feature_name"]
        self._feature_scaler_mean = d["feature_scaler_mean"]
        self._feature_scaler_std = d["feature_scaler_std"]
        self._label_scaler_mean = d["label_scaler_mean"]
        self._label_scaler_std = d["label_scaler_std"]
        self._species = d["species"]

    def _load(self):
        """Read data from files and then featurize."""
        raise NotImplementedError

    @staticmethod
    def get_molecules(molecules):
        if isinstance(molecules, Path):
            path = str(molecules)
            supp = Chem.SDMolSupplier(path, sanitize=True, removeHs=False)
            molecules = [m for m in supp]
        return molecules

    @staticmethod
    def build_graphs(grapher, molecules, features, species):
        """
        Build DGL graphs using grapher for the molecules.

        Args:
            grapher (Grapher): grapher object to create DGL graphs
            molecules (list): rdkit molecules
            features (list): each element is a dict of extra features for a molecule
            species (list): chemical species (str) in all molecules

        Returns:
            list: DGL graphs
        """
        graphs = []
        """
        with ProcessPool(max_workers=12, max_tasks=10) as pool:
            for i, (m, feats) in enumerate(zip(molecules, features)):
                if m is not None:
                    future = pool.schedule(grapher.build_graph_and_featurize, 
                                            args=[m], timeout=30,
                                            kwargs={"extra_feats_info":feats, 
                                                    "dataset_species":species}
                                            )
                    future.add_done_callback(task_done)
                    try:
                        g = future.result()
                        g.graph_id = i
                        graphs.append(g)
                    except:
                        pass
                else: graphs.append(None)

        """
        for i, (m, feats) in tqdm(enumerate(zip(molecules, features))):
            if m is not None:
                g = grapher.build_graph_and_featurize(
                    m, extra_feats_info=feats, element_set=species
                )
                # add this for check purpose; some entries in the sdf file may fail
                g.graph_id = i

            else:
                g = None
            graphs.append(g)

        return graphs

    def __getitem__(self, item):
        """Get data point with index

        Args:
            item (int): data point index

        Returns:
            g (DGLGraph or DGLHeteroGraph): graph ith data point
            lb (dict): Labels of the data point
        """
        (
            g,
            lb,
        ) = (
            self.graphs[item],
            self.labels[item],
        )
        return g, lb

    def __len__(self):
        """
        Returns:
            int: length of dataset
        """
        return len(self.graphs)

    def __repr__(self):
        rst = "Dataset " + self.__class__.__name__ + "\n"
        rst += "Length: {}\n".format(len(self))
        for ft, sz in self.feature_size.items():
            rst += "Feature: {}, size: {}\n".format(ft, sz)
        for ft, nm in self.feature_name.items():
            rst += "Feature: {}, name: {}\n".format(ft, nm)
        return rst


class ReactionDatasetGraphs(BaseDataset):
    def __init__(
        self,
        grapher,
        file,
        feature_transformer=True,
        label_transformer=True,
        dtype="float32",
        target="ts",
        filter_species=[2, 3],
        filter_outliers=True,
        filter_sparse_rxns=False,
        feature_filter=False,
        classifier=False,
        debug=False,
        classif_categories=None,
        extra_keys=None,
        dataset_atom_types=None,
        extra_info=None,
        species=["C", "F", "H", "N", "O", "Mg", "Li", "S", "Cl", "P", "O", "Br"],
    ):
        if dtype not in ["float32", "float64"]:
            raise ValueError(f"`dtype {dtype}` should be `float32` or `float64`.")
        #print("extra keys: " + str(extra_keys))
        self.grapher = grapher
        (
            all_mols,
            all_labels,
            features,
        ) = create_reaction_network_files_and_valid_rows(
            file,
            bond_map_filter=False,
            target=target,
            filter_species=filter_species,
            classifier=classifier,
            debug=debug,
            filter_outliers=filter_outliers,
            categories=classif_categories,
            filter_sparse_rxn=filter_sparse_rxns,
            feature_filter=feature_filter,
            extra_keys=extra_keys,
            extra_info=extra_info,
        )

        self.molecules = all_mols
        self.raw_labels = all_labels
        self.extra_features = features
        self.feature_transformer = feature_transformer
        self.label_transformer = label_transformer
        self.dtype = dtype
        self.state_dict_filename = None
        self.graphs = None
        self.labels = None
        self.target = target
        self.extra_keys = extra_keys
        self._feature_size = None
        self._feature_name = None
        self._feature_scaler_mean = None
        self._feature_scaler_std = None
        self._label_scaler_mean = None
        self._label_scaler_std = None
        self._species = species
        self._elements = dataset_atom_types
        self._failed = None
        self.classifier = classifier
        self.classif_categories = classif_categories
        self._load()

    def _load(self):
        logger.info("Start loading dataset")

        # get molecules, labels, and extra features
        molecules = self.get_molecules(self.molecules)
        raw_labels = self.get_labels(self.raw_labels)
        
        #if self.extra_features is not None:
        #    extra_features = self.get_features(self.extra_features)
        #else:
        #    extra_features = [None] * len(molecules)

        # get state info
        if self.state_dict_filename is not None:
            logger.info(f"Load dataset state dict from: {self.state_dict_filename}")
            state_dict = torch.load(str(self.state_dict_filename))
            self.load_state_dict(state_dict)

        # get species
        # species = get_dataset_species_from_json(self.pandas_df)
        if self._species is None:
            system_species = set()
            for mol in self.molecules:
                species = list(set(mol.species))
                system_species.update(species)

            self._species = sorted(system_species)
        # self._species

        # create dgl graphs
        print("constructing graphs & features....")

        graphs = self.build_graphs(
            self.grapher, self.molecules, self._species
        )
        graphs_not_none_indices = [i for i, g in enumerate(graphs) if g is not None]
        print("number of graphs valid: " + str(len(graphs_not_none_indices)))
        print("number of graphs: " + str(len(graphs)))
        assert len(graphs_not_none_indices) == len(
            graphs
        ), "Some graphs are invalid in construction, this should not happen"
        # store feature name and size
        self._feature_name = self.grapher.feature_name
        self._feature_size = self.grapher.feature_size
        logger.info("Feature name: {}".format(self.feature_name))
        logger.info("Feature size: {}".format(self.feature_size))

        # feature transformers
        if self.feature_transformer:
            if self.state_dict_filename is None:
                feature_scaler = HeteroGraphFeatureStandardScaler(mean=None, std=None)
            else:
                assert (
                    self._feature_scaler_mean is not None
                ), "Corrupted state_dict file, `feature_scaler_mean` not found"
                assert (
                    self._feature_scaler_std is not None
                ), "Corrupted state_dict file, `feature_scaler_std` not found"

                feature_scaler = HeteroGraphFeatureStandardScaler(
                    mean=self._feature_scaler_mean, std=self._feature_scaler_std
                )

            graphs_not_none = [graphs[i] for i in graphs_not_none_indices]
            graphs_not_none = feature_scaler(graphs_not_none)
            molecules_ordered = [self.molecules[i] for i in graphs_not_none_indices]
            molecules_final = [0 for i in graphs_not_none_indices]
            # update graphs
            for i, g in zip(graphs_not_none_indices, graphs_not_none):
                molecules_final[i] = molecules_ordered[i]
                graphs[i] = g
            self.molecules_ordered = molecules_final

            if self.state_dict_filename is None:
                self._feature_scaler_mean = feature_scaler.mean
                self._feature_scaler_std = feature_scaler.std

            logger.info(f"Feature scaler mean: {self._feature_scaler_mean}")
            logger.info(f"Feature scaler std: {self._feature_scaler_std}")

        # create reaction
        reactions = []
        self.labels = []
        self._failed = []
        for i, lb in enumerate(raw_labels):
            mol_ids = lb["reactants"] + lb["products"]
            for d in mol_ids:
                # ignore reaction whose reactants or products molecule is None
                if d not in graphs_not_none_indices:
                    self._failed.append(True)
                    break
            else:
                rxn = ReactionInNetwork(
                    reactants=lb["reactants"],
                    products=lb["products"],
                    atom_mapping=lb["atom_mapping"],
                    bond_mapping=lb["bond_mapping"],
                    total_bonds=lb["total_bonds"],
                    total_atoms=lb["total_atoms"],
                    id=lb["id"],
                    extra_info=lb["extra_info"],
                )

                reactions.append(rxn)

                if "environment" in lb:
                    environemnt = lb["environment"]
                else:
                    environemnt = None

                if self.classifier:
                    lab_temp = torch.zeros(self.classif_categories)
                    lab_temp[int(lb["value"][0])] = 1

                    if lb["value_rev"] != None:
                        lab_temp_rev = torch.zeros(self.classif_categories)
                        lab_temp[int(lb["value_rev"][0])] = 1
                    else:
                        lab_temp_rev = None

                    label = {
                        "value": lab_temp,
                        "value_rev": lab_temp_rev,
                        "id": lb["id"],
                        "environment": environemnt,
                        "atom_map": lb["atom_mapping"],
                        "bond_map": lb["bond_mapping"],
                        "total_bonds": lb["total_bonds"],
                        "total_atoms": lb["total_atoms"],
                        "reaction_type": lb["reaction_type"],
                        "extra_info": lb["extra_info"],
                    }
                    self.labels.append(label)
                else:
                    label = {
                        "value": torch.tensor(
                            lb["value"], dtype=getattr(torch, self.dtype)
                        ),
                        "value_rev": torch.tensor(
                            lb["value_rev"], dtype=getattr(torch, self.dtype)
                        ),
                        "id": lb["id"],
                        "environment": environemnt,
                        "atom_map": lb["atom_mapping"],
                        "bond_map": lb["bond_mapping"],
                        "total_bonds": lb["total_bonds"],
                        "total_atoms": lb["total_atoms"],
                        "reaction_type": lb["reaction_type"],
                        "extra_info": lb["extra_info"],
                    }
                    self.labels.append(label)

                self._failed.append(False)

        self.reaction_ids = list(range(len(reactions)))

        # create reaction network - this needs to be axed
        #self.reaction_network = ReactionNetwork(
        #    molecules=graphs, reactions=reactions, wrappers=molecules_final
        #)
        self.reactions = reactions
        self.graphs = graphs

        # feature transformers
        if self.label_transformer:
            # normalization
            values = torch.stack([lb["value"] for lb in self.labels])  # 1D tensor
            values_rev = torch.stack([lb["value_rev"] for lb in self.labels])

            if self.state_dict_filename is None:
                mean = torch.mean(values)
                std = torch.std(values)
                self._label_scaler_mean = mean
                self._label_scaler_std = std
            else:
                assert (
                    self._label_scaler_mean is not None
                ), "Corrupted state_dict file, `label_scaler_mean` not found"
                assert (
                    self._label_scaler_std is not None
                ), "Corrupted state_dict file, `label_scaler_std` not found"
                mean = self._label_scaler_mean
                std = self._label_scaler_std

            values = (values - mean) / std
            value_rev_scaled = (values_rev - mean) / std

            # update label
            for i, lb in enumerate(values):
                self.labels[i]["value"] = lb
                self.labels[i]["scaler_mean"] = mean
                self.labels[i]["scaler_stdev"] = std
                self.labels[i]["value_rev"] = value_rev_scaled[i]

            logger.info(f"Label scaler mean: {mean}")
            logger.info(f"Label scaler std: {std}")

        logger.info(f"Finish loading {len(self.labels)} reactions...")

    @staticmethod
    def build_graphs(grapher, molecules, species):
        """
        Build DGL graphs using grapher for the molecules.

        Args:
            grapher (Grapher): grapher object to create DGL graphs
            molecules (list): rdkit molecules
            features (list): each element is a dict of extra features for a molecule
            species (list): chemical species (str) in all molecules

        Returns:
            list: DGL graphs
        """

        count = 0
        graphs = []
        # use tqdm to show progress bar
        
        #for ind, mol in enumerate(molecules):
        ind = 0 
        for mol in tqdm(molecules, desc="mol graphs"):
            #feats = features[count]
            if mol is not None:
                g = grapher.build_graph_and_featurize(
                    mol, element_set=species
                )

                # add this for check purpose; some entries in the sdf file may fail
                g.graph_id = ind
            else:
                g = None
            graphs.append(g)
            count += 1
            ind += 1
        return graphs

    @staticmethod
    def get_labels(labels):
        if isinstance(labels, Path):
            labels = yaml_load(labels)
        return labels
    

    def __getitem__(self, item):
        
        # reactions subset
        sub_reactions = self.reactions[item]
        label = self.labels[item]

        # subset ids and map between global molecule index and subset molecule index
        ids = set()
        ids.update(sub_reactions.init_reactants + sub_reactions.init_products)
        ids = sorted(ids)
        
        
        ret_labels = {
            "value": label["value"], 
            "value_rev": label["value_rev"],
            "id": label["id"], 
            "scaler_mean": label["scaler_mean"],
            "scaler_stdev": label["scaler_stdev"], 
            "ids": ids, 
            "extra_info": label["extra_info"],
        }

        #return sub_reactions, ids, self.graphs, self.labels[item]
        return sub_reactions, ret_labels

    def __len__(self):
        return len(self.reaction_ids)


class ReactionDatasetLMDBDataset(BaseDataset):
    def __init__(self, reaction_lmdb):
        self.reaction_lmdb = reaction_lmdb
        self.reactions = self.reaction_lmdb.reactions
        self.molecules = self.reaction_lmdb.molecules
        self.graphs = [i["molecule_graph"] for i in self.molecules]
        #self.graphs = self.reaction_lmdb.molecules.graphs
        self.elements = self.reaction_lmdb.molecules.elements
        self.rings = self.reaction_lmdb.molecules.ring_sizes
        self.charges = self.reaction_lmdb.molecules.charges
        self.feature_info = self.reaction_lmdb.molecules.feature_info
        self.reaction_ids = [
            i["reaction_index"] for i in self.reactions
        ]  # here we can either use reaction index or the specific id
        self.reaction_ids = [int(i) for i in range(len(self.reaction_ids))]
        self._feature_name = self.reaction_lmdb.reactions.feature_name
        self._feature_size = self.reaction_lmdb.reactions.feature_size
        self.dtype = self.reaction_lmdb.reactions.dtype

    
    def __getitem__(self, item):
        
        # reaction from reactions
        reaction = deepcopy(self.reactions[item])


        # subset ids and map between global molecule index and subset molecule index
        ids = set()
        ids.update(
            reaction["reaction_molecule_info"]["reactants"]["init_reactants"]+
            reaction["reaction_molecule_info"]["products"]["init_products"]
        )
        #sort_order = np.argsort(ids)
        ids = sorted(ids)
    
 
        
        label_mean = self.reactions.mean
        label_std = self.reactions.std

        ret_labels = {
            "value": reaction["label"],
            "value_rev":reaction["reverse_label"],
            "id": reaction["reaction_index"], 
            "scaler_mean": label_mean,
            "scaler_stdev": label_std, 
            "ids": ids
        }
        return reaction, ret_labels
        
    def __len__(self):
        return len(self.reaction_lmdb.reactions)


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

        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
            #self.metadata_path = self.path / "metadata.npz"

            self._keys = []
            self.envs = []
            for db_path in db_paths:
                cur_env = self.connect_db(db_path)
                self.envs.append(cur_env)

                # If "length" encoded as ascii is present, use that
                length_entry = cur_env.begin().get("length".encode("ascii"))
                if length_entry is not None:
                    num_entries = pickle.loads(length_entry)
                else:
                    # Get the number of stores data from the number of entries in the LMDB
                    num_entries = cur_env.stat()["entries"]

                # Append the keys (0->num_entries) as a list
                self._keys.append(list(range(num_entries)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
            
        
        else:
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
        
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)
            #data_object.id = f"{db_idx}_{el_idx}"
    
        else:
            #!CHECK, _keys should be less then total numbers of keys as there are more properties.
            datapoint_pickled = self.env.begin().get(f"{self._keys[idx]}".encode("ascii"))

            data_object = pickle.loads(datapoint_pickled)

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
        if not self.path.is_file():
            self.env_ = self.envs[0]
            raise("Not Implemented Yet")
                
        else:
            self.env_ = self.env

        
    @property
    def charges(self):
        charges = self.env_.begin().get("charges".encode("ascii"))
        return pickle.loads(charges)

    @property
    def ring_sizes(self):
        ring_sizes = self.env_.begin().get("ring_sizes".encode("ascii"))
        return pickle.loads(ring_sizes)

    @property
    def elements(self):
        elements = self.env_.begin().get("elements".encode("ascii"))
        return pickle.loads(elements)

    @property
    def feature_info(self):
        feature_info = self.env_.begin().get("feature_info".encode("ascii"))
        return pickle.loads(feature_info)


class LmdbReactionDataset(LmdbBaseDataset):
    def __init__(self, config, transform=None):
        super(LmdbReactionDataset, self).__init__(config=config, transform=transform)

        if not self.path.is_file():
            self.env_ = self.envs[0]
            #get keys
            for i in range(1, len(self.envs)):
                for key in ["feature_size", "dtype", "feature_name"]: #, "mean", "std"]:
                    assert self.envs[i].begin().get(key.encode("ascii")) == self.envs[0].begin().get(key.encode("ascii"))
                    #! mean and std are not equal across different dataset at this time.
            #get mean and std
            mean_list = [pickle.loads(self.envs[i].begin().get("mean".encode("ascii"))) for i in range(0, len(self.envs))]
            std_list = [pickle.loads(self.envs[i].begin().get("std".encode("ascii"))) for i in range(0, len(self.envs))]
            count_list = [pickle.loads(self.envs[i].begin().get("length".encode("ascii"))) for i in range(0, len(self.envs))]
            self._mean, self._std = combined_mean_std(mean_list, std_list, count_list)
                    
        else:
            self.env_ = self.env
            self._mean = pickle.loads(self.env_.begin().get("mean".encode("ascii")))
            self._std  = pickle.loads(self.env_.begin().get("std".encode("ascii")))
        
    @property
    def dtype(self):
        dtype = self.env_.begin().get("dtype".encode("ascii"))
        return  pickle.loads(dtype)

    @property
    def feature_size(self):
        feature_size = self.env_.begin().get("feature_size".encode("ascii"))
        return pickle.loads(feature_size)

    @property
    def feature_name(self):
        feature_name = self.env_.begin().get("feature_name".encode("ascii"))
        return pickle.loads(feature_name)

    @property
    def mean(self):
        mean = self.env.begin().get("mean".encode("ascii"))
        return pickle.loads(mean)
    
    @property
    def std(self):
        std = self.env.begin().get("std".encode("ascii"))
        return pickle.loads(std)
    

class Subset(BaseDataset):
    def __init__(self, dataset, indices):
        self.dtype = dataset.dtype
        self.dataset = dataset
        self.indices = indices

        self._feature_size = dataset._feature_size
        self._feature_name = dataset._feature_name
        self.molecules = dataset.graphs

        self._feature_scaler_std = dataset._feature_scaler_std
        self._feature_scaler_mean = dataset._feature_scaler_mean
        self._label_scaler_mean = dataset._label_scaler_mean
        self._label_scaler_std = dataset._label_scaler_std
        
        self.reaction_network = dataset.reaction_network
        # iterate through indices to get the graphs, labels 
        #self.graphs = [dataset.graphs[i] for i in indices]
        #self.labels = [dataset.labels[i] for i in indices]

    @property
    def feature_size(self):
        return self.dataset._feature_size

    @property
    def feature_name(self):
        return self.dataset._feature_name

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class SubsetLMDB(BaseDataset):

    def __init__(self, dataset, indices):
        self.dtype = dataset.dtype
        self.dataset = dataset

        self.indices = indices
        
        self._feature_scaler_std = dataset._feature_scaler_std
        self._feature_scaler_mean = dataset._feature_scaler_mean
        self._label_scaler_mean = dataset._label_scaler_mean
        self._label_scaler_std = dataset._label_scaler_std
        self._feature_size = dataset._feature_size
        self._feature_name = dataset._feature_name
        
        self.molecules = dataset.molecules
        self.graphs = dataset.graphs
        self.labels = dataset.labels
        self.reactions = dataset.reactions

        
    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def train_validation_test_split(dataset, validation=0.1, test=0.1, random_seed=None, lmdb=True):
    """
    Split a dataset into training, validation, and test set.

    The training set will be automatically determined based on `validation` and `test`,
    i.e. train = 1 - validation - test.

    Args:
        dataset: the dataset
        validation (float, optional): The amount of data (fraction) to be assigned to
            validation set. Defaults to 0.1.
        test (float, optional): The amount of data (fraction) to be assigned to test
            set. Defaults to 0.1.
        random_seed (int, optional): random seed that determines the permutation of the
            dataset. Defaults to 35.

    Returns:
        [train set, validation set, test_set]
    """
    assert validation + test <= 1.0, "validation + test >= 1"
    size = len(dataset)
    num_val = int(size * validation)
    num_test = int(size * test)
    num_train = size - num_val - num_test

    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.permutation(size)
    train_idx = idx[:num_train]
    val_idx = idx[num_train : num_train + num_val]
    test_idx = idx[num_train + num_val :]

    
    
    if bool(lmdb):
        if test==0.0:
            return [
                    SubsetLMDB(dataset, train_idx),
                    SubsetLMDB(dataset, val_idx),
            ]
        else:
            return [
                    SubsetLMDB(dataset, train_idx),
                    SubsetLMDB(dataset, val_idx),
                    SubsetLMDB(dataset, test_idx),
            ]

    else:
        if test==0.0:
            return [
                    Subset(dataset, train_idx),
                    Subset(dataset, val_idx),
            ]
        else:
            return [
                    Subset(dataset, train_idx),
                    Subset(dataset, val_idx),
                    Subset(dataset, test_idx),
            ]
    
