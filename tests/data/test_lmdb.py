import unittest
import torch


from bondnet.data.lmdb import (
    LmdbMoleculeDataset,
    LmdbReactionDataset
)
from bondnet.data.reaction_network import ReactionLMDB
from bondnet.data.dataset import ReactionDatasetLMDBDataset
from bondnet.data.lmdb import construct_lmdb_and_save_reaction_dataset
from bondnet.test_utils import get_test_reaction_data

torch.multiprocessing.set_sharing_strategy("file_system")


class TestLMDB(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.dataset = get_test_reaction_data(allowed_charges=[0, 1, 2, -1, -2])
        construct_lmdb_and_save_reaction_dataset(self.dataset, "./testdata/lmdb/")

    def test_featurization(self): 
        config = {
            "src": "./testdata/lmdb/molecule.lmdb"
        }
        config_rxn = {
            "src": "./testdata/lmdb/reaction.lmdb"
        }

        mol = LmdbMoleculeDataset(config=config)
        reaction = LmdbReactionDataset(config=config_rxn)
        rxn_ntwk = ReactionLMDB(mol, reaction)
        dataset = ReactionDatasetLMDBDataset(rxn_ntwk)
        features = rxn_ntwk.reactions.feature_name
        assert "charge one hot" in features["global"]





