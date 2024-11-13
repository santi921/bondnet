# Generalized Bondnet
This is the repository for the generalized variant of Bondnet. Here we have iterated on the original algorithm with generalized treatment of atom mappings, bonding, and descriptors. This variant still works with a minimal set of features and can handle rdkit for vanilla bond definitions but can also intake custom bond definitions to work with metals, noncovalent bonds, etc. We can also now handle arbitrary number of species on each side of the reaction. 

# Table of Contents

- [Installation](#installation)
- [Data Format for Generalized](#Data-Format-for-Generalized-Model)

BonDNet is a graph neural network model for the prediction of bond dissociation
energies (BDEs). It can be applied to both homolytic and heterolytic bond dissociations
for molecules of any charge. This model is described in the paper:
[BonDNet: a graph neural network for the prediction of bond dissociation
energies for charged molecules, _Chemical Science_, 2021.](https://doi.org/10.1039/D0SC05251E)

<p align="center">
<img src="bondnet.png" alt="BonDNet" width="600">
</p>

# Installation

Currently, we support installation from source:

1. create a conda environment using yaml provided and install pip dependencies

   ```bash
   cd ./enviro
   conda env create --file=environment.yml
   pip install -r requirements.txt
   ```
2. install this repo

   ```bash
   git clone https://github.com/santi921/bondnet/
   cd bondnet
   pip install -e ./
   ```

# Data Format for Generalized Model

Data for the generalized model should provide the model a json/bson with a few columns. Here we define the bonds as :

- `bonds_broken` - A list of lists with atoms broken in the reactants vs. the products
- `bonds_formed` - A list of lists with atoms formed in the reactants vs. the products
- `reactant/product_bonds` - A full list of lists with all the bonds in the reactants/products. This is a single list of lists on each side of the reaction.
- `reactant/product_bonds_no_metal` - A full list of lists with all the bonds in the reactants/products EXCLUDING bonds with metals in them. This separate list is used by the featurizer and if your reactions don't have metals, simply copy `reactant/product_bonds` here. This is a single list of lists on each side of the reaction.
- Pymatgen objects under the key `combined_reactants/products_graph` or `reactants/products_molecular_graph`
- `reactant/product_id` - These are either strings, ints, or lists depending on the number of products/reactants on each side.
- `charge` - defines the charge of the reaction
- A target variable that you can specify in the settings.json file, we use `dG_sp` or `product_energy` - `reactant_energy` You can also specify a transfer learning variable in the settings file to train on before training on the actual target.
- Optionally, you can specify extra bond and atom features to use in training. These are specified with `extra_feat_atom_reactant/product` and `extra_feat_bond_reactant/product` for atoms and bond features respectively. For `extra_feat_atom` features you order atoms in the same order as the bonds define atoms. `extra_feat_bond_products/reactants` should be ordered in the same order as the bonds are specified.
- You can also keep extra features to carry through training in order to track descriptors such as functional groups, for example `functional_group_reacted`. These are not used by the model. 

Note on atom mapping: there is no explicit atom mappings used by this variant of bondnet, we assume that the user handles that such a bond [0, 10] in products and a bond [0, 10] in the reactants are between the same atoms. The user has to ensure atom mapping is consistent between bonds on each side of the reaction AND that the pymatgen objects and extra features follow this ordering as well 


