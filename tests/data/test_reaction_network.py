import numpy as np
from bondnet.data.reaction_network import ReactionInNetwork, ReactionNetwork


class TestReaction:
    def test_mapping_as_list(self):
        def assert_one(mappings, ref, mode="atom"):
            mp_list = ReactionInNetwork._mapping_as_list(mappings, mode)
            assert mp_list == ref

        # no missing
        assert_one([{0: 1, 1: 3}, {0: 2, 1: 0}], [3, 0, 2, 1])

        # missing last item (i.e. 4)
        assert_one([{0: 1, 1: 3}, {0: 2, 1: 0}], [3, 0, 2, 1, 4], "bond")

        # missing middle item (i.e. 3)
        assert_one([{0: 1, 1: 4}, {0: 2, 1: 0}], [3, 0, 2, 4, 1], "bond")

        # one has no bond at all
        assert_one([{0: 0, 1: 1}, {}], [0, 1, 2], "bond")
        assert_one([{}, {0: 0, 1: 1}], [0, 1, 2], "bond")
        assert_one([{0: 1, 1: 2}, {}], [2, 0, 1], "bond")
        assert_one([{}, {0: 1, 1: 2}, {}], [2, 0, 1], "bond")

