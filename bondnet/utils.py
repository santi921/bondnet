
try:
    from rdkit.Chem import rdEHTTools  # requires RDKit 2019.9.1 or later
except ImportError:
    rdEHTTools = None
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, Get3DDistanceMatrix

import os
import time
import pickle
import yaml
import random
import torch
import dgl
import logging
import warnings
import sys
import shutil
import itertools
import copy
from pathlib import Path
import numpy as np
from typing import List, Any
from collections import defaultdict
import networkx as nx




global __ATOM_LIST__
__ATOM_LIST__ = [
    "h",
    "he",
    "li",
    "be",
    "b",
    "c",
    "n",
    "o",
    "f",
    "ne",
    "na",
    "mg",
    "al",
    "si",
    "p",
    "s",
    "cl",
    "ar",
    "k",
    "ca",
    "sc",
    "ti",
    "v ",
    "cr",
    "mn",
    "fe",
    "co",
    "ni",
    "cu",
    "zn",
    "ga",
    "ge",
    "as",
    "se",
    "br",
    "kr",
    "rb",
    "sr",
    "y",
    "zr",
    "nb",
    "mo",
    "tc",
    "ru",
    "rh",
    "pd",
    "ag",
    "cd",
    "in",
    "sn",
    "sb",
    "te",
    "i",
    "xe",
    "cs",
    "ba",
    "la",
    "ce",
    "pr",
    "nd",
    "pm",
    "sm",
    "eu",
    "gd",
    "tb",
    "dy",
    "ho",
    "er",
    "tm",
    "yb",
    "lu",
    "hf",
    "ta",
    "w",
    "re",
    "os",
    "ir",
    "pt",
    "au",
    "hg",
    "tl",
    "pb",
    "bi",
    "po",
    "at",
    "rn",
    "fr",
    "ra",
    "ac",
    "th",
    "pa",
    "u",
    "np",
    "pu",
]


global atomic_valence
global atomic_valence_electrons

atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[3] = [1, 2, 3, 4, 5, 6]
atomic_valence[5] = [3, 4]
atomic_valence[6] = [4]
atomic_valence[7] = [3, 4]
atomic_valence[8] = [2, 1, 3]
atomic_valence[9] = [1]
atomic_valence[12] = [3, 5, 6]
atomic_valence[14] = [4]
atomic_valence[15] = [5, 3]  # [5,4,3]
atomic_valence[16] = [6, 3, 2]  # [6,4,2]
atomic_valence[17] = [1]
atomic_valence[32] = [4]
atomic_valence[35] = [1]
atomic_valence[53] = [1]

atomic_valence_electrons = {}
atomic_valence_electrons[1] = 1
atomic_valence_electrons[3] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[12] = 2
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[32] = 4
atomic_valence_electrons[35] = 7
atomic_valence_electrons[53] = 7
logger = logging.getLogger(__name__)


def np_split_by_size(array, sizes, axis=0):
    """
    Split an array into `len(sizes)` chunks with size of each chunk in dim
    according to `sizes`.

    This is a convenient function over np.split(), where one need to give the indices at
    which to split the array (not easy to work with).

    Args:
        array:
        sizes (list): size of each chunk.
        axis (int): the axis along which to split the data

    Returns:
        list: a list of array.

    Example:
        >>> np_split_by_size([0,1,2,3,4,5], [1,2,3])
        >>>[[0], [1,2], [3,4,5]]
    """
    array = np.asarray(array)
    assert array.shape[axis] == sum(sizes), "array.shape[axis] not equal to sum(sizes)"

    indices = list(itertools.accumulate(sizes))
    indices = indices[:-1]

    return np.split(array, indices, axis=axis)


def list_split_by_size(data: List[Any], sizes: List[int]) -> List[List[Any]]:
    """
    Split a list into `len(sizes)` chunks with the size of each chunk given by `sizes`.

    This is a similar to `np_split_by_size` for a list. We cannot use
    `np_split_by_size` for a list of graphs, because DGL errors out if we convert a
    list of graphs to an array of graphs.

    Args:
        data: the list of data to split
        sizes: size of each chunk.

    Returns:
        a list of list, where the size of each inner list is given by `sizes`.

    Example:
        >>> list_split_by_size([0,1,2,3,4,5], [1,2,3])
        >>>[[0], [1,2], [3,4,5]]
    """
    assert len(data) == sum(
        sizes
    ), f"Expect len(array) be equal to sum(sizes); got {len(data)} and {sum(sizes)}"

    indices = list(itertools.accumulate(sizes))

    new_data = []
    a = []
    for i, x in enumerate(data):
        a.append(x)
        if i + 1 in indices:
            new_data.append(a)
            a = []

    return new_data


def to_path(path):
    return Path(path).expanduser().resolve()


def check_exists(path, is_file=True):
    p = to_path(path)
    if is_file:
        if not p.is_file():
            raise ValueError(f"File does not exist: {path}")
    else:
        if not p.is_dir():
            raise ValueError(f"File does not exist: {path}")


def create_directory(path, path_is_directory=False):
    p = to_path(path)
    if not path_is_directory:
        dirname = p.parent
    else:
        dirname = p
    if not dirname.exists():
        os.makedirs(dirname)


def pickle_dump(obj, filename):
    create_directory(filename)
    with open(to_path(filename), "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    with open(to_path(filename), "rb") as f:
        obj = pickle.load(f)
    return obj


def yaml_dump(obj, filename):
    create_directory(filename)
    with open(to_path(filename), "w") as f:
        yaml.dump(obj, f, default_flow_style=False)


def yaml_load(filename):
    with open(to_path(filename), "r") as f:
        obj = yaml.safe_load(f)
    return obj


def stat_cuda(msg):
    print("-" * 10, "cuda status:", msg, "-" * 10)
    print(
        "allocated: {}M, max allocated: {}M, cached: {}M, max cached: {}M".format(
            torch.cuda.memory_allocated() / 1024 / 1024,
            torch.cuda.max_memory_allocated() / 1024 / 1024,
            torch.cuda.memory_cached() / 1024 / 1024,
            torch.cuda.max_memory_cached() / 1024 / 1024,
        )
    )


def seed_torch(seed=35, cudnn_benchmark=False, cudnn_deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    dgl.random.seed(seed)


def save_checkpoints(
    state_dict_objects, misc_objects, is_best, msg=None, filename="checkpoint.pkl"
):
    """
    Save checkpoints for all objects for later recovery.

    Args:
        state_dict_objects (dict): A dictionary of objects to save. The object should
            have state_dict() (e.g. model, optimizer, ...)
        misc_objects (dict): plain python object to save
        filename (str): filename for the checkpoint

    """
    objects = copy.copy(misc_objects)
    for k, obj in state_dict_objects.items():
        objects[k] = obj.state_dict()
    torch.save(objects, filename)
    if is_best:
        shutil.copyfile(filename, "best_checkpoint.pkl")
        if msg is not None:
            logger.info(msg)


def load_checkpoints(state_dict_objects, map_location=None, filename="checkpoint.pkl"):
    """
    Load checkpoints for all objects for later recovery.

    Args:
        state_dict_objects (dict): A dictionary of objects to save. The object should
            have state_dict() (e.g. model, optimizer, ...)
    """
    checkpoints = torch.load(str(filename), map_location)
    for k, obj in state_dict_objects.items():
        state_dict = checkpoints.pop(k)
        obj.load_state_dict(state_dict)
    return checkpoints


class Timer:
    def __init__(self):
        self.first = None
        self.latest = None

    def step(self, msg=None):
        if self.first is None:
            self.first = self.latest = time.time()
        current = time.time()
        if msg is None:
            m = ""
        else:
            m = " User message: {}.".format(msg)
        print(
            "{:.2f} | {:.2f}. Time (s) since last called and since first called.{}".format(
                current - self.latest, current - self.first, m
            )
        )
        self.latest = current


def warn_stdout(message, category, filename, lineno, file=None, line=None):
    """
    Redirect warning message to stdout instead of stderr.

    To use this:
    >>> warnings.showwarning = warn_stdout
    >>> warnings.warn("some warning message")

    see: https://stackoverflow.com/questions/858916/how-to-redirect-python-warnings-to-a-custom-stream
    """
    sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))


def parse_settings(file="settings.txt"):
    """
    Processes a text file into a dictionary for training
    
    Args:
        file (file): file with settings 
    Returns: 
        dict_ret(dictionary): dictionary with settings 
    """
    # some default values that get written over if in the file
    test = False
    filter_species = [2, 3]
    restore = False
    freeze = True
    on_gpu = False
    augment = False
    distributed = False
    classifier = False
    featurizer_xyz = True
    save_hyper_params = "./hyper.pkl"
    dataset_state_dict_filename = "./dataset_state_dict.pkl"
    dataset_loc = "../../../dataset/mg_dataset/20220613_reaction_data.json"
    model_path = "./"
    epochs = 100
    start_epoch = 0
    embedding_size = 24
    batch_size = 256
    lr = 0.00001
    num_gpu = 1
    categories = 5
    category_weights = [1.0, 2.0, 1.0, 1.5, 2.0]
    weight_decay = 0.0001
    
    early_stop = True
    scheduler = False
    filter_outliers = True
    transfer_epochs = 100
    transfer = True
    loss = 'mse'

    fc_hidden_size = [128, 64]
    fc_layers = -1
    fc_activation = "ReLU"
    fc_batch_norm = 0
    fc_dropout = 0.0

    gated_hidden_size = [64, 64, 64]
    gated_batch_norm = 0
    gated_graph_norm = 0
    gated_dropout = 0.0
    gated_activation = "ReLU"
    gated_num_fc_layers = 2
    gated_num_layers = 4
    gated_residual = False

    num_lstm_layers = 3
    num_lstm_iters = 5

    with open(file) as f:
        lines = f.readlines()

        for i in lines:
            if len(i.split()) > 1:
                if i.split()[0] == "dataset_loc": 
                    dataset_loc = i.split()[1]
                if i.split()[0] == "restore":
                    restore = "True" == i.split()[1]
                if i.split()[0] == "on_gpu":
                    on_gpu = "True" == i.split()[1]
                if i.split()[0] == "test":
                    test = "True" == i.split()[1]
                if i.split()[0] == 'filter_species':
                    filter_species = [int(j) for j in i.split()[1:]]
                if i.split()[0] == 'freeze':
                    freeze = "True" == i.split()[1]                  
                if i.split()[0] == "distributed":
                    distributed = "True" == i.split()[1]
                if i.split()[0] == "save_hyper_params":
                    save_hyper_params = i.split()[1]
                if i.split()[0] == "dataset_state_dict_filename":
                    dataset_state_dict_filename = i.split()[1]
                if i.split()[0] == "model_path":
                    model_path = i.split()[1]
                if i.split()[0] == "augment":
                    augment = "True" == i.split()[1]
                if i.split()[0] == "num_gpu":
                    num_gpu = int(i.split()[1])
                if i.split()[0] == 'xyz_featurizer':
                    featurizer_xyz = "True" == i.split()[1]
                if i.split()[0] == "early_stop":
                    early_stop = "True" == i.split()[1]
                if i.split()[0] == "scheduler":
                    scheduler = "True" == i.split()[1]
                if i.split()[0] == "transfer_epochs":
                    transfer_epochs = int(i.split()[1])
                if i.split()[0] == "transfer":
                    transfer = "True" == i.split()[1]
                if i.split()[0] == "loss":
                    loss = i.split()[1]
                if i.split()[0] == 'classifier':
                    classifier = "True" == i.split()[1]
                if i.split()[0] == "categories":
                    categories = int(i.split()[1])
                if i.split()[0] == "batch_size":
                    batch_size = int(i.split()[1])
                if i.split()[0] == "epochs":
                    epochs = int(i.split()[1])
                if i.split()[0] == "start_epoch":
                    start_epoch = int(i.split()[1])
                if i.split()[0] == "embedding_size":
                    embedding_size = int(i.split()[1])
                if i.split()[0] == "lr":
                    lr = float(i.split()[1])
                if i.split()[0] == "weight_decay":
                    weight_decay = float(i.split()[1])
                if i.split()[0] == "gated_hidden_size":
                    gated_hidden_size = [int(j) for j in i.split()[1:]]              
                if i.split()[0] == "category_weights":
                    category_weights = [float(j) for j in i.split()[1:]]

                if i.split()[0] == "gated_dropout":
                    gated_dropout = float(i.split()[1])
                if i.split()[0] == "gated_graph_norm":
                    gated_graph_norm = "True" == i.split()[1]
                if i.split()[0] == "gated_batch_norm":
                    gated_batch_norm = "True" == i.split()[1]
                if i.split()[0] == "gated_activation":
                    gated_activation = str(i.split()[1])
                if i.split()[0] == "gated_num_fc_layers":
                    gated_num_fc_layers = int(i.split()[1])
                if i.split()[0] == "gated_residual":
                    gated_residual = "True" == i.split()[1]
                if i.split()[0] == "gated_num_layers":
                    gated_num_layers = int(i.split()[1])

                if i.split()[0] == "fc_hidden_size":
                    fc_hidden_size = [int(j) for j in i.split()[1:]]
                if i.split()[0] == "fc_layers":
                    fc_layers = int(i.split()[1])
                if i.split()[0] == "fc_activation":
                    fc_activation = str(i.split()[1])
                if i.split()[0] == "fc_batch_norm":
                    fc_batch_norm = "True" == i.split()[1]
                if i.split()[0] == "fc_dropout":
                    fc_dropout = float(i.split()[1])
                if i.split()[0] == 'filter_outliers':
                    filter_outliers = "True" == i.split()[1]
                
                if i.split()[0] == "num_lstm_iters":
                    num_lstm_iters = int(i.split()[1])
                if i.split()[0] == "num_lstm_layers":
                    num_lstm_layers = int(i.split()[1])

        if gated_num_layers == -1:
            gated_num_layers = len(gated_hidden_size)
        if fc_layers == -1:
            fc_layers = len(fc_hidden_size)

        print("using the following settings:")
        print("--" * 20)
        print("dataset loc:                     {}".format(dataset_loc))
        print("Small Dataset?:                  {}".format(str(test)))
        print("augment:                         {}".format(augment))
        print("restore:                         {}".format(str(restore)))
        print("freeze graph layers w/transfer:  {}".format(freeze))
        print("distributed:                     {}".format(str(distributed)))
        print("on gpu:                          {}".format(str(on_gpu)))
        print("filter species?                  {}".format(filter_species))
        print("filter outliers?                 {}".format(filter_outliers))
        print("num gpu:                         {}".format(str(num_gpu)))
        print("xyz featurizer:                 {}".format(featurizer_xyz))
        print("hyperparam save file:            {}".format(save_hyper_params))
        print("dataset state dict:              {}".format(dataset_state_dict_filename))
        print("model dir                        {}".format(model_path))
        print("classifier                       {}".format(classifier))
        print("category weights:                {}".format(str(category_weights)))
        print("batch size:                      {}".format(batch_size))
        print("epochs:                          {:1d}".format(epochs))
        print("lr:                              {:7f}".format(lr))
        print("weight decay:                    {:.3f}".format(weight_decay))
        print("early_stop:                      {}".format(str(early_stop)))
        print("scheduler:                       {}".format(str(scheduler)))
        print("transfer_epochs:                 {}".format(str(transfer_epochs)))
        print("transfer:                        {}".format(str(transfer)))
        print("loss:                            {}".format(str(loss)))
        print("categories:                      {}".format(str(categories)))
        print("embedding size:                  {:1d}".format(embedding_size))
        print("fc layers:                       {:1d}".format(fc_layers))
        print("fc hidden layer:                 {}".format(str(fc_hidden_size)))
        print("gated layers:                    {:1d}".format(gated_num_layers))
        print("gated hidden layers:             {}".format(str(gated_hidden_size)))
        print("num lstm iters:                  {}".format(str(num_lstm_iters)))
        print("num lstm layer:                  {}".format(str(num_lstm_layers)))
        print("gated fc layers:                 {}".format(str(gated_num_fc_layers)))
        print("fc activation:                   {}".format(str(fc_activation)))
        print("fc batch norm:                   {}".format(str(fc_batch_norm)))
        print("fc dropout:                      {:.2f}".format(fc_dropout))
        print("gated activation:                {}".format(str(gated_activation)))
        print("gated dropout:                   {:.2f}".format(gated_dropout))
        print("gated batch norm:                {}".format(str(gated_batch_norm)))
        print("gated graph norm:                {}".format(str(gated_graph_norm)))
        print("gated resid:                     {}".format(str(gated_residual)))
        print("--" * 20)

        dict_ret = {}
        dict_ret["classifier"] = classifier
        dict_ret["categories"] = categories
        dict_ret["category_weights"] = category_weights
        dict_ret["dataset_loc"] = dataset_loc
        dict_ret["augment"] = augment
        dict_ret["debug"] = test
        dict_ret["on_gpu"] = on_gpu
        dict_ret["filter_species"] = filter_species
        dict_ret["num_gpu"] = num_gpu
        dict_ret["epochs"] = epochs
        dict_ret["distributed"] = distributed
        dict_ret["save_hyper_params"] = save_hyper_params
        dict_ret["dataset_state_dict_filename"] = Path(dataset_state_dict_filename)
        dict_ret["model_path"] = Path(model_path)
        dict_ret["featurizer_xyz"] = featurizer_xyz
        dict_ret['early_stop'] = early_stop 
        dict_ret['scheduler'] = scheduler 
        dict_ret['transfer_epochs'] = transfer_epochs 
        dict_ret['transfer'] = transfer 
        dict_ret['loss'] = loss 
        dict_ret['filter_outliers'] = filter_outliers
        dict_ret["freeze"] = freeze  

        dict_ret["start_epoch"] = start_epoch
        dict_ret["embedding_size"] = embedding_size
        dict_ret["batch_size"] = batch_size
        dict_ret["learning_rate"] = lr
        dict_ret["weight_decay"] = weight_decay
        dict_ret["restore"] = restore

        dict_ret["fc_hidden_size"] = fc_hidden_size
        dict_ret["fc_num_layers"] = fc_layers
        dict_ret["fc_dropout"] = fc_dropout
        dict_ret["fc_batch_norm"] = fc_batch_norm
        dict_ret["fc_activation"] = fc_activation
        dict_ret["gated_hidden_size"] = gated_hidden_size
        dict_ret["gated_activation"] = gated_activation
        dict_ret["gated_graph_norm"] = gated_graph_norm
        dict_ret["gated_batch_norm"] = gated_batch_norm
        dict_ret["gated_dropout"] = gated_dropout
        dict_ret["gated_num_fc_layers"] = gated_num_fc_layers
        dict_ret["gated_num_layers"] = gated_num_layers
        dict_ret["gated_residual"] = gated_residual
        dict_ret["num_lstm_iters"] = num_lstm_iters
        dict_ret["num_lstm_layers"] = num_lstm_layers

        return dict_ret


def str_atom(atom):
    """
    convert integer atom to string atom
    """
    global __ATOM_LIST__
    atom = __ATOM_LIST__[atom - 1]
    return atom


def int_atom(atom):
    """
    convert str atom to integer atom
    """
    global __ATOM_LIST__
    # print(atom)
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1


def get_UA(maxValence_list, valence_list):
    """ """
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if not maxValence - valence > 0:
            continue
        UA.append(i)
        DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, use_graph=True):
    """ """
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = get_UA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA, AC, use_graph=use_graph)[0]

    return BO


def valences_not_too_large(BO, valences):
    """ """
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False

    return True


def charge_is_OK(
    BO,
    AC,
    charge,
    DU,
    atomic_valence_electrons,
    atoms,
    valences,
    allow_charged_fragments=True,
):
    # total charge
    Q = 0

    # charge fragment list
    q_list = []

    if allow_charged_fragments:

        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atoms):
            q = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1

            if q != 0:
                q_list.append(q)

    return charge == Q


def BO_is_OK(
    BO,
    AC,
    charge,
    DU,
    atomic_valence_electrons,
    atoms,
    valences,
    allow_charged_fragments=True,
):
    """
    Sanity of bond-orders
    args:
        BO -
        AC -
        charge -
        DU -
    optional
        allow_charges_fragments -
    returns:
        boolean - true of molecule is OK, false if not
    """

    if not valences_not_too_large(BO, valences):
        return False

    check_sum = (BO - AC).sum() == sum(DU)
    check_charge = charge_is_OK(
        BO,
        AC,
        charge,
        DU,
        atomic_valence_electrons,
        atoms,
        valences,
        allow_charged_fragments,
    )

    if check_charge and check_sum:
        return True

    return False


def get_atomic_charge(atom, atomic_valence_electrons, BO_valence):
    """ """

    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif atom == 15 and BO_valence == 5:
        charge = 0
    elif atom == 16 and BO_valence == 6:
        charge = 0
    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge


def clean_charges(mol):
    """
    This hack should not be needed anymore, but is kept just in case
    """

    Chem.SanitizeMol(mol)
    # rxn_smarts = ['[N+:1]=[*:2]-[C-:3]>>[N+0:1]-[*:2]=[C-0:3]',
    #              '[N+:1]=[*:2]-[O-:3]>>[N+0:1]-[*:2]=[O-0:3]',
    #              '[N+:1]=[*:2]-[*:3]=[*:4]-[O-:5]>>[N+0:1]-[*:2]=[*:3]-[*:4]=[O-0:5]',
    #              '[#8:1]=[#6:2]([!-:6])[*:3]=[*:4][#6-:5]>>[*-:1][*:2]([*:6])=[*:3][*:4]=[*+0:5]',
    #              '[O:1]=[c:2][c-:3]>>[*-:1][*:2][*+0:3]',
    #              '[O:1]=[C:2][C-:3]>>[*-:1][*:2]=[*+0:3]']

    rxn_smarts = [
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][CX3-,NX3-:5][#6,#7:6]1=[#6,#7:7]>>"
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][-0,-0:5]=[#6,#7:6]1[#6-,#7-:7]",
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3](=[#6,#7:4])[#6,#7:5]=[#6,#7:6][CX3-,NX3-:7]1>>"
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3]([#6-,#7-:4])=[#6,#7:5][#6,#7:6]=[-0,-0:7]1",
    ]

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

    for i, fragment in enumerate(fragments):
        for smarts in rxn_smarts:
            patt = Chem.MolFromSmarts(smarts.split(">>")[0])
            while fragment.HasSubstructMatch(patt):
                rxn = AllChem.ReactionFromSmarts(smarts)
                ps = rxn.RunReactants((fragment,))
                fragment = ps[0][0]
                Chem.SanitizeMol(fragment)
        if i == 0:
            mol = fragment
        else:
            mol = Chem.CombineMols(mol, fragment)

    return mol


def BO2mol(
    mol,
    BO_matrix,
    atoms,
    atomic_valence_electrons,
    mol_charge,
    allow_charged_fragments=True,
    use_atom_maps=True,
):
    """
    based on code written by Paolo Toscani
    From bond order, atoms, valence structure and total charge, generate an
    rdkit molecule.
    args:
        mol - rdkit molecule
        BO_matrix - bond order matrix of molecule
        atoms - list of integer atomic symbols
        atomic_valence_electrons -
        mol_charge - total charge of molecule
    optional:
        allow_charged_fragments - bool - allow charged fragments
    returns
        mol - updated rdkit molecule with bond connectivity
    """

    l = len(BO_matrix)
    l2 = len(atoms)
    BO_valences = list(BO_matrix.sum(axis=1))

    if l != l2:
        raise RuntimeError(
            "sizes of adjMat ({0:d}) and Atoms {1:d} differ".format(l, l2)
        )

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    for i in range(l):
        for j in range(i + 1, l):
            bo = int(round(BO_matrix[i, j]))
            if bo == 0:
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    if allow_charged_fragments:
        mol = set_atomic_charges(
            mol,
            atoms,
            atomic_valence_electrons,
            BO_valences,
            BO_matrix,
            mol_charge,
            use_atom_maps,
        )
    else:
        mol = set_atomic_radicals(
            mol, atoms, atomic_valence_electrons, BO_valences, use_atom_maps
        )

    return mol


def set_atomic_charges(
    mol,
    atoms,
    atomic_valence_electrons,
    BO_valences,
    BO_matrix,
    mol_charge,
    use_atom_maps,
    metals=False,
):
    """ """
    q = 0
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i + 1)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                q += 1
                charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1
        if metals == True:
            if atom == 12:
                number_of_single_bonds_to_Mg = list(BO_matrix[i, :]).count(1)
                if number_of_single_bonds_to_Mg == 5:
                    charge = 1
                else:
                    charge = 2
            if atom == 3:
                charge = 1

        if abs(charge) > 0:
            a.SetFormalCharge(int(charge))

    # mol = clean_charges(mol)

    return mol


def set_atomic_radicals(
    mol, atoms, atomic_valence_electrons, BO_valences, use_atom_maps
):
    """
    The number of radical electrons = absolute atomic charge
    """
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i + 1)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])

        if abs(charge) > 0:
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol


def get_bonds(UA, AC):
    """ """
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k + 1 :]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))

    return bonds


def get_UA_pairs(UA, AC, use_graph=True):
    """ """

    bonds = get_bonds(UA, AC)

    if len(bonds) == 0:
        return [()]

    if use_graph:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]

        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs


def AC2BO(AC, atoms, charge, allow_charged_fragments=True, use_graph=True):
    """
    implemenation of algorithm shown in Figure 2
    UA: unsaturated atoms
    DU: degree of unsaturation (u matrix in Figure)
    best_BO: Bcurr in Figure
    """

    global atomic_valence
    global atomic_valence_electrons

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    AC_valence = list(AC.sum(axis=1))
    fail = 0

    for i, (atomicNum, valence) in enumerate(zip(atoms, AC_valence)):
        # valence can't be smaller than number of neighbourgs
        possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]
        if not possible_valence:
            #print(atomicNum, valence, possible_valence)
            print(
                "Valence of atom",
                i,
                "is",
                valence,
                "which bigger than allowed max",
                max(atomic_valence[atomicNum]),
                ". Stopping",
            )
            fail = 1
            # sys.exit()
        valences_list_of_lists.append(possible_valence)

    # convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = itertools.product(*valences_list_of_lists)

    best_BO = AC.copy()

    for valences in valences_list:

        UA, DU_from_AC = get_UA(valences, AC_valence)

        check_len = len(UA) == 0
        if check_len:
            check_bo = BO_is_OK(
                AC,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
            )
        else:
            check_bo = None

        if check_len and check_bo:
            return AC, atomic_valence_electrons, fail

        UA_pairs_list = get_UA_pairs(UA, AC, use_graph=use_graph)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, use_graph=use_graph)
            status = BO_is_OK(
                BO,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
            )
            charge_OK = charge_is_OK(
                BO,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
            )

            if status:
                return BO, atomic_valence_electrons, fail
            elif (
                BO.sum() >= best_BO.sum()
                and valences_not_too_large(BO, valences)
                and charge_OK
            ):
                best_BO = BO.copy()

    return best_BO, atomic_valence_electrons, fail


def AC2mol(
    mol,
    AC,
    atoms,
    charge,
    allow_charged_fragments=True,
    use_graph=True,
    use_atom_maps=True,
):
    """ """

    # convert AC matrix to bond order (BO) matrix
    BO, atomic_valence_electrons, fail = AC2BO(
        AC,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph,
    )

    # add BO connectivity and charge info to mol object
    mol = BO2mol(
        mol,
        BO,
        atoms,
        atomic_valence_electrons,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_atom_maps=use_atom_maps,
    )

    # If charge is not correct don't return mol
    # if Chem.GetFormalCharge(mol) != charge:
    #    return []

    # BO2mol returns an arbitrary resonance form. Let's make the rest
    mols = rdchem.ResonanceMolSupplier(
        mol, Chem.UNCONSTRAINED_CATIONS, Chem.UNCONSTRAINED_ANIONS
    )
    mols = [mol for mol in mols]

    return mols, fail


def get_proto_mol(atoms):
    """ """
    mol = Chem.MolFromSmarts("[#" + str(atoms[0]) + "]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atoms)):
        a = Chem.Atom(atoms[i])
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def read_xyz_file(filename, look_for_charge=True):
    """ """

    atomic_symbols = []
    xyz_coordinates = []
    charge = 0
    title = ""

    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                title = line
                if "charge=" in line:
                    charge = int(line.split("=")[1])
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, charge, xyz_coordinates


def xyz2AC(atoms, xyz, charge, use_huckel=False):
    """
    atoms and coordinates to atom connectivity (AC)
    args:
        atoms - int atom types
        xyz - coordinates
        charge - molecule charge
    optional:
        use_huckel - Use Huckel method for atom connecitivty
    returns
        ac - atom connectivity matrix
        mol - rdkit molecule
    """

    if use_huckel:
        return xyz2AC_huckel(atoms, xyz, charge)
    else:
        return xyz2AC_vdW(atoms, xyz)


def xyz2AC_vdW(atoms, xyz):

    # Get mol template
    mol = get_proto_mol(atoms)

    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)
    AC = get_AC(mol)
    return AC, mol


def get_AC(mol, covalent_factor=1.3):
    """
    Generate adjacent matrix from atoms and coordinates.
    AC is a (num_atoms, num_atoms) matrix with 1 being covalent bond and 0 is not
    covalent_factor - 1.3 is an arbitrary factor
    args:
        mol - rdkit molobj with 3D conformer
    optional
        covalent_factor - increase covalent bond length threshold with facto
    returns:
        AC - adjacent matrix
    """

    # Calculate distance matrix
    dMat = Get3DDistanceMatrix(mol)

    pt = Chem.GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    AC = np.zeros((num_atoms, num_atoms), dtype=int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * covalent_factor
        for j in range(i + 1, num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * covalent_factor
            if dMat[i, j] <= Rcov_i + Rcov_j:
                AC[i, j] = 1
                AC[j, i] = 1

    return AC


def xyz2AC_huckel(atomicNumList, xyz, charge):
    """
    args
        atomicNumList - atom type list
        xyz - coordinates
        charge - molecule charge
    returns
        ac - atom connectivity
        mol - rdkit molecule
    """
    mol = get_proto_mol(atomicNumList)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    num_atoms = len(atomicNumList)
    AC = np.zeros((num_atoms, num_atoms)).astype(int)

    mol_huckel = Chem.Mol(mol)
    mol_huckel.GetAtomWithIdx(0).SetFormalCharge(
        charge
    )  # mol charge arbitrarily added to 1st atom

    passed, result = rdEHTTools.RunMol(mol_huckel)
    opop = result.GetReducedOverlapPopulationMatrix()
    tri = np.zeros((num_atoms, num_atoms))
    tri[
        np.tril(np.ones((num_atoms, num_atoms), dtype=bool))
    ] = opop  # lower triangular to square matrix
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            pair_pop = abs(tri[j, i])
            if pair_pop >= 0.15:  # arbitry cutoff for bond. May need adjustment
                AC[i, j] = 1
                AC[j, i] = 1

    return AC, mol


def chiral_stereo_check(mol):
    """
    Find and embed chiral information into the model based on the coordinates
    args:
        mol - rdkit molecule, with embeded conformer
    """
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)

    return


def xyz2mol(
    atoms,
    coordinates,
    charge=0,
    allow_charged_fragments=True,
    use_graph=True,
    use_huckel=False,
    embed_chiral=False,
    use_atom_maps=True,
):
    """
    Generate a rdkit molobj from atoms, coordinates and a total_charge.
    args:
        atoms - list of atom types (int)
        coordinates - 3xN Cartesian coordinates
        charge - total charge of the system (default: 0)
    optional:
        allow_charged_fragments - alternatively radicals are made
        use_graph - use graph (networkx)
        use_huckel - Use Huckel method for atom connectivity prediction
        embed_chiral - embed chiral information to the molecule
    returns:
        mols - list of rdkit molobjects
    """

    # Get atom connectivity (AC) matrix, list of atomic numbers, molecular charge,
    # and mol object with no connectivity information
    AC, mol = xyz2AC(atoms, coordinates, charge, use_huckel=use_huckel)

    # Convert AC to bond order matrix and add connectivity and charge info to
    # mol object
    new_mols, fail = AC2mol(
        mol,
        AC,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph,
        use_atom_maps=use_atom_maps,
    )
    # Check for stereocenters and chiral centers
    if embed_chiral:
        for new_mol in new_mols:
            chiral_stereo_check(new_mol)
    if fail:
        return []

    return new_mols


def sdf_writer():
    """
    write sdf file for a molecule w/custom bonding
    takes:
        atoms -
        bonds -
        features -
    returns


    """



