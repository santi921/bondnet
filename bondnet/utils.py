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




def parse_settings(file="./input_files/input_1.txt"):

    # some default values that get written over if in the file
    test = False
    restore = False
    on_gpu = False
    distributed = False
    save_hyper_params = './hyper.pkl'
    epochs = 10
    start_epoch = 0
    embedding_size = 24
    batch_size = 10
    lr = 0.00001
    num_gpu = 1

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

                if i.split()[0] == "restore":
                    restore = "True" == i.split()[1]
                if i.split()[0] == "on_gpu":
                    on_gpu = "True" == i.split()[1]
                if i.split()[0] == "test":
                    test = "True" == i.split()[1]
                if i.split()[0] == "distributed":
                    distributed = "True" == i.split()[1]
                if i.split()[0] == "save_hyper_params":
                    save_hyper_params = i.split()[1]
                if i.split()[0] == "num_gpu":
                    num_gpu = int(i.split()[1])

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
                if i.split()[0] == "gated_dropout":
                    gated_dropout = float(i.split()[1])
                if i.split()[0] == "gated_graph_norm":
                    gated_graph_norm = int(i.split()[1])
                if i.split()[0] == "gated_batch_norm":
                    gated_batch_norm = int(i.split()[1])
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
                    fc_batch_norm = int(i.split()[1])
                if i.split()[0] == "fc_dropout":
                    fc_dropout = float(i.split()[1])

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
        print("restore: " + str(restore))
        print("distributed: " + str(distributed))
        print("batch size: " + str(batch_size))
        print("on gpu: " + str(on_gpu))
        print("num gpu: " + str(num_gpu))
        print("hyperparam save file: " + str(save_hyper_params))

        print("Small Dataset?: " + str(test))
        print("epochs: {:1d}".format(epochs))
        print("embedding size: {:1d}".format(embedding_size))
        print("lr: {:7f}".format(lr))
        print("weight decay: {:.3f}".format(weight_decay))

        print("fc layers: {:1d}".format(fc_layers))
        print("fc hidden layer: " + str(fc_hidden_size))
        print("fc activation: " + str(fc_activation))
        print("fc batch norm: " + str(fc_batch_norm))
        print("fc dropout: {:.2f}".format(fc_dropout))

        print("gated layers: {:1d}".format(gated_num_layers))
        print("gated hidden layers: " + str(gated_hidden_size))
        print("gated activation: " + str(gated_activation))
        print("gated dropout: {:.2f}".format(gated_dropout))
        print("gated batch norm: " + str(gated_batch_norm))
        print("gated graph norm: " + str(gated_graph_norm))
        print("gated fc layers: " + str(gated_num_fc_layers))
        print("gated resid: " + str(gated_residual))

        print("num lstm iters: " + str(num_lstm_iters))
        print("num lstm layer: " + str(num_lstm_layers))
        print("--" * 20)

        dict_ret = {}
        dict_ret["test"] = test
        dict_ret["on_gpu"] = on_gpu
        dict_ret["num_gpu"] = num_gpu
        dict_ret["epochs"] = epochs
        dict_ret["distributed"] = distributed
        dict_ret["save_hyper_params"] = save_hyper_params

        dict_ret["start_epoch"] = start_epoch
        dict_ret["embedding_size"] = embedding_size
        dict_ret["batch_size"] = batch_size
        dict_ret["lr"] = lr
        dict_ret["weight_decay"] = weight_decay
        dict_ret["restore"] = restore

        dict_ret["fc_hidden_size"] = fc_hidden_size
        dict_ret["fc_layers"] = fc_layers
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