from bondnet.core.reaction import create_reactions_from_reactant

from bondnet.test_utils import (
    create_reactions_nonsymmetric_reactant,
    create_reactions_symmetric_reactant,
)


class TestReaction:
    def test_atom_mapping(self):
        A2B, A2BC = create_reactions_nonsymmetric_reactant()

        # m0 to m1
        ref_mapping = [{0: 0, 1: 2, 2: 1, 3: 3}]
        reaction = A2B[0]
        mapping = reaction.atom_mapping()
        assert mapping == ref_mapping

        # m0 to m2 m4
        ref_mapping = [{0: 0, 1: 2, 2: 1}, {0: 3}]
        reaction = A2BC[0]
        mapping = reaction.atom_mapping()
        assert mapping == ref_mapping

    def test_bond_mapping_int_index(self):
        A2B, A2BC = create_reactions_nonsymmetric_reactant()

        # m0 to m1
        ref_mapping = [{0: 1, 1: 2, 2: 3}]
        assert A2B[0].bond_mapping_by_int_index() == ref_mapping

        # m0 to m2 and m3
        ref_mapping = [{0: 1, 1: 0, 2: 2}, {}]
        assert A2BC[0].bond_mapping_by_int_index() == ref_mapping

    def test_bond_mapping_tuple_index(self):
        A2B, A2BC = create_reactions_nonsymmetric_reactant()

        # m0 to m1
        ref_mapping = [{(0, 1): (0, 2), (1, 2): (1, 2), (1, 3): (2, 3)}]
        assert A2B[0].bond_mapping_by_tuple_index() == ref_mapping

        # m0 to m2 and m3
        ref_mapping = [{(0, 1): (0, 2), (0, 2): (0, 1), (1, 2): (1, 2)}, {}]
        assert A2BC[0].bond_mapping_by_tuple_index() == ref_mapping

    def test_bond_mapping_sdf_int_index(self):
        """
        m0
        RDKit          3D

          0  0  0  0  0  0  0  0  0  0999 V3000
        M  V30 BEGIN CTAB
        M  V30 COUNTS 4 4 0 0 0
        M  V30 BEGIN ATOM
        M  V30 1 C 0 1 0 0 RAD=3 VAL=2
        M  V30 2 O -1 0 0 0 RAD=2 VAL=1
        M  V30 3 N 1 0 0 0 RAD=2 VAL=2
        M  V30 4 H 1 0 1 0
        M  V30 END ATOM
        M  V30 BEGIN BOND
        M  V30 1 1 1 2
        M  V30 2 1 1 3
        M  V30 3 0 2 3
        M  V30 4 1 3 4
        M  V30 END BOND
        M  V30 END CTAB
        M  END
        $$$$

        m1
             RDKit          3D

          0  0  0  0  0  0  0  0  0  0999 V3000
        M  V30 BEGIN CTAB
        M  V30 COUNTS 4 3 0 0 0
        M  V30 BEGIN ATOM
        M  V30 1 C 0 1 0 0 RAD=2 VAL=1
        M  V30 2 N 1 0 0 0 RAD=2 VAL=2
        M  V30 3 O -1 0 0 0 RAD=3
        M  V30 4 H 1 0 1 0
        M  V30 END ATOM
        M  V30 BEGIN BOND
        M  V30 1 1 1 2
        M  V30 2 0 2 3
        M  V30 3 1 2 4
        M  V30 END BOND
        M  V30 END CTAB
        M  END
        $$$$

        m2
             RDKit          3D

          0  0  0  0  0  0  0  0  0  0999 V3000
        M  V30 BEGIN CTAB
        M  V30 COUNTS 3 3 0 0 0
        M  V30 BEGIN ATOM
        M  V30 1 C 0 1 0 0 RAD=3 VAL=2
        M  V30 2 N -1 0 0 0 RAD=3 VAL=1
        M  V30 3 O 1 0 0 0 RAD=2 VAL=1
        M  V30 END ATOM
        M  V30 BEGIN BOND
        M  V30 1 1 1 2
        M  V30 2 1 1 3
        M  V30 3 0 2 3
        M  V30 END BOND
        M  V30 END CTAB
        M  END
        $$$$

        m3
             RDKit          3D

          0  0  0  0  0  0  0  0  0  0999 V3000
        M  V30 BEGIN CTAB
        M  V30 COUNTS 1 0 0 0 0
        M  V30 BEGIN ATOM
        M  V30 1 H 1 0 1 0 VAL=-1
        M  V30 END ATOM
        M  V30 END CTAB
        M  END
        $$$$


        """
        A2B, A2BC = create_reactions_nonsymmetric_reactant()

        # m0 to m1
        ref_mapping = [{0: 1, 1: 2, 2: 3}]
        assert A2B[0].bond_mapping_by_sdf_int_index() == ref_mapping

        # m0 to m2 m3
        ref_mapping = [{0: 1, 1: 0, 2: 2}, {}]
        assert A2BC[0].bond_mapping_by_sdf_int_index() == ref_mapping







