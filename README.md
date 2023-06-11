Use the pretrained model:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/pretrained?filepath=bondnet%2Fscripts%2Fpredict_binder.ipynb)

Train the model:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/pretrained?filepath=bondnet%2Fscripts%2Ftrain_bde.ipynb)

# Table of Contents

- [Installation](#installation)
- [Use pretrained model for prediction](#use-pretrained-model-for-prediction)
- [Train the model](#train-the-model)
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

1. create a conda environment

   ```bash
   conda create --name bondnet
   conda activate bondnet
   conda install python==3.7
   ```
2. install dependencies (see their websites for more options)

   ```bash
   conda install pytorch==1.6.0 torchvision -c pytorch   # https://pytorch.org
   conda install dgl==0.5.0 -c dglteam                   # https://www.dgl.ai/pages/start.html
   conda install pymatgen==2020.8.13 -c conda-forge      # https://pymatgen.org/installation.html
   conda install rdkit==2020.03.5 -c conda-forge         # https://rdkit.org/docs/Install.html
   conda install openbabel==3.1.1 -c conda-forge         # http://openbabel.org/wiki/Category:Installation
   ```
3. install this repo

   ```bash
   git clone https://github.com/mjwen/bondnet.git
   pip install -e bondnet
   ```

# Use pretrained model for prediction

For a quick prediction of the BDEs for a single molecule, try the live demo at:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/pretrained?filepath=bondnet%2Fscripts%2Fpredict_binder.ipynb)

Alternatively, a command line interface (CLI) `bondnet` is provided for batch predictions.
(Optional. We suggest switching to the `pretrained` branch for using the `bondnet` CLI
to make predictions. It should be more stable. To install the `pretrained` branch
, do `cd bondnet`, then `git checkout pretrained`, and finally `pip install -e  .`)

- A single molecule given by a `SMILES` or `InChI` string, e.g.:

  ```bash
  bondnet single "C1COC(=O)O1"
  ```
- Multiple molecules listed in a file. Supported molecule format includes `sdf`, `pdb`, `smiles` and `inchi`, e.g.:

  ```bash
  bondnet multiple molecules.sdf -o results.sdf
  ```
- Explicitly specifying the bond dissociation reactions. In this mode, a `moleclue` file
  listing all the molecules and a `reaction` file listing all the bond dissociation
  reactions are needed. Supported molecule format includes `graph `, `sdf`, `pdb`,
  `smiles `, and `inchi`. e.g.

  ```bash
  bondnet reaction -t graph molecule_graphs.json reactions.csv
  bondnet reaction -t sdf molecules.sdf reactions.csv
  ```

More detailed instructions, example input files, and description of the file formats,
can be found [here](./bondnet/scripts/examples/predict).

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

# Train the model

The [train_bde.ipynb](./bondnet/scripts/train_bde.ipynb) Jupyter notebook shows
how to train BonDNet on a BDE dataset of both neutral and charged molecules.
Try it at: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/pretrained?filepath=bondnet%2Fscripts%2Ftrain_bde.ipynb)

The [train_bde.ipynb](./bondnet/scripts/train_bde.ipynb) Jupyter notebook trains a model on CPU.
If you want to train on GPUs (a single GPU or distributed), take a look at
[train_bde_distributed.py](./bondnet/scripts/train_bde_distributed.py). A model can be trained by

```bash
python train_bde_distributed.py  molecules.sdf  molecule_attributes.yaml  reactions.yaml
```

More detailed instructions, example input files, and description of the file formats,
can be found [here](./bondnet/scripts/examples/train).
