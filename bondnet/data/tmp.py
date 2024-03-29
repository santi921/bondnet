device = 'cuda:0'
sorted_index_reaction = [torch.tensor([value for key, value in sorted(d.items())], device=device) for d in atom_map]

reactant_ids = torch.tensor(reactant_ids, device=device)
#atom_batch_indices = torch.tensor(atom_batch_indices, device=device)

matches = ((atom_batch_indices[:, None] == reactant_ids[None, :]).any(dim=1).nonzero(as_tuple=False).squeeze())

breakpoint()
sorted_values_concat = torch.cat(sorted_index_reaction)
reaction_indices = matches[sorted_values_concat]
# device = 'cuda:0'
# atom_batch_indices = torch.tensor(atom_batch_indices, device=device)
# reactant_ids = torch.tensor(reactant_ids, device=device)
# matches = ((atom_batch_indices[:, None] == reactant_ids[None, :]).any(dim=1).nonzero().squeeze())

# reaction_indices = matches[sorted_values_list] #+ sign of coeff

        # combined_atom_map = []
        # combined_bond_map = []
        # combined_total_bonds = []
        # combined_total_atoms = []
        # combined_num_bonds_total = []
        # combined_num_atoms_total = []

        # for reaction in reactions:
        #     mapplings = reaction["mappings"]

        #     atom_map = mapplings['atom_map']
        #     bond_map = mapplings['bond_map']
        #     total_atoms = mapplings['total_atoms']
        #     total_bonds = mapplings['total_bonds']
        #     _num_bonds_total = mapplings['num_bonds_total']
        #     num_atoms_total = mapplings['num_atoms_total']


            
        #     # Adjust mappings with the current offset
        #     adjusted_atom_map = [{k+offset_atoms: v+offset_atoms for k, v in map_pair.items()} for map_pair in atom_map]
        #     adjusted_bond_map = [{k+offset_bonds: v+offset_bonds for k, v in map_pair.items()} for map_pair in bond_map]

        #     # Update the combined mappings
        #     combined_atom_map.extend(adjusted_atom_map)
        #     combined_bond_map.extend(adjusted_bond_map)

        #     # Adjust total_atoms and total_bonds with the current offset
        #     adjusted_total_atoms = [i + offset_atoms for i in total_atoms]
        #     adjusted_total_bonds = [i + offset_bonds for i in total_bonds]

        #     combined_total_atoms.extend(adjusted_total_atoms)
        #     combined_total_bonds.extend(adjusted_total_bonds)

        #     # Update offsets for the next iteration
        #     offset_atoms += max(total_atoms) + 1  # Assuming total_atoms is a list of atom indices
        #     offset_bonds += max(total_bonds) + 1  # Assuming total_bonds is a list of bond indices

        # # Construct batched features based on adjusted mappings. This step will depend on how your features are structured
        # # and may involve aggregating features from individual reactions into a combined tensor that aligns with the batched_graph.
        # # Placeholder for feature construction
        # batched_features = None  # Construct batched features based on the combined mappings and individual reaction features

        # return batched_graphs, combined_atom_map, combined_bond_map, combined_total_atoms, combined_total_bonds, batched_features
reaction1 = {'bond_map': [[{0: 17, 1: 33, 2: 23, 3: 28, 4: 13, 5: 19, 6: 11, 7: 24, 8: 18, 9: 36, 10: 43, 11: 8, 12: 12, 13: 27, 14: 41}, {0: 0, 1: 37, 2: 21, 3: 29, 4: 39, 5: 16, 6: 9, 7: 22, 8: 7, 9: 34, 10: 32, 11: 4, 12: 25, 13: 44, 14: 14, 15: 6, 16: 20, 17: 31, 18: 15, 19: 2, 20: 38, 21: 30, 22: 1, 23: 42, 24: 26, 25: 35, 26: 40, 27: 5, 28: 10}], [{0: 19, 1: 33, 2: 13, 3: 23, 4: 17, 5: 12, 6: 27, 7: 41, 8: 11, 9: 24, 10: 18, 11: 36, 12: 43, 13: 8}, {0: 0, 1: 37, 2: 21, 3: 29, 4: 39, 5: 16, 6: 9, 7: 22, 8: 7, 9: 34, 10: 32, 11: 4, 12: 25, 13: 44, 14: 14, 15: 6, 16: 20, 17: 3, 18: 31, 19: 15, 20: 2, 21: 38, 22: 1, 23: 30, 24: 42, 25: 26, 26: 35, 27: 10, 28: 40, 29: 5}]], 'atom_map': [[{0: 0, 1: 1, 2: 2, 6: 6, 10: 10, 11: 11, 4: 4, 5: 5, 3: 3, 7: 7, 9: 9, 8: 8, 14: 14, 15: 15, 13: 13, 12: 12}, {0: 16, 4: 20, 6: 22, 7: 23, 9: 25, 11: 27, 12: 28, 13: 29, 15: 31, 17: 33, 20: 36, 24: 40, 25: 41, 26: 42, 27: 43, 2: 18, 1: 17, 3: 19, 5: 21, 8: 24, 10: 26, 14: 30, 16: 32, 18: 34, 19: 35, 22: 38, 21: 37, 23: 39}], [{1: 0, 0: 1, 4: 2, 5: 6, 2: 10, 3: 11, 9: 4, 10: 5, 11: 3, 12: 7, 13: 9, 14: 8, 6: 14, 8: 15, 7: 13}, {13: 12, 0: 16, 4: 20, 6: 22, 7: 23, 9: 25, 11: 27, 12: 28, 14: 29, 16: 31, 18: 33, 21: 36, 25: 40, 26: 41, 27: 42, 28: 43, 1: 18, 3: 17, 2: 19, 5: 21, 8: 24, 10: 26, 15: 30, 17: 32, 20: 34, 19: 35, 23: 38, 22: 37, 24: 39}]] 'num_bonds_total': 45, 'num_atoms_total': 44}


reaction2 = {'bond_map': [[{0: 34, 1: 36, 2: 12, 3: 19, 4: 11, 5: 4, 6: 2, 7: 45, 8: 30, 9: 40, 10: 0, 11: 28, 12: 3, 13: 18, 14: 10, 15: 29, 16: 24, 17: 44, 18: 16, 19: 26, 20: 31, 21: 33, 22: 1, 23: 41, 24: 25, 25: 32, 26: 46, 27: 37, 28: 6, 29: 13, 30: 38, 31: 15}, {0: 22, 1: 42, 2: 8, 3: 14, 4: 21, 5: 5, 6: 9, 7: 35, 8: 47, 9: 23, 10: 7, 11: 20, 12: 39, 13: 17, 14: 27}], [{0: 16, 1: 26, 2: 31, 3: 41, 4: 33, 5: 1, 6: 32, 7: 25, 8: 46, 9: 6, 10: 13, 11: 37, 12: 38, 13: 15}, {0: 34, 1: 12, 2: 43, 3: 11, 4: 4, 5: 19, 6: 2, 7: 45, 8: 30, 9: 40, 10: 0, 11: 28, 12: 3, 13: 18, 14: 10, 15: 29, 16: 24, 17: 44, 18: 42, 19: 8, 20: 14, 21: 22, 22: 21, 23: 5, 24: 9, 25: 35, 26: 47, 27: 7, 28: 23, 29: 20, 30: 39, 31: 17, 32: 27}]], 'atom_map': [[{18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 8: 8, 12: 12, 13: 13, 14: 14, 7: 7, 6: 6, 5: 5, 9: 9, 10: 10, 11: 11, 15: 15, 16: 16, 17: 17}, {0: 32, 6: 38, 7: 39, 8: 40, 9: 41, 10: 42, 11: 43, 12: 44, 13: 45, 14: 46, 1: 33, 2: 34, 3: 35, 4: 36, 5: 37}], [{0: 18, 1: 19, 2: 20, 3: 21, 4: 22, 5: 23, 6: 24, 7: 25, 8: 26, 9: 27, 10: 28, 11: 29, 12: 30, 13: 31}, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 8: 8, 12: 12, 13: 13, 14: 14, 7: 7, 5: 6, 6: 5, 11: 9, 9: 10, 10: 11, 15: 15, 17: 16, 16: 17, 18: 32, 26: 38, 24: 39, 25: 40, 27: 41, 28: 42, 29: 43, 30: 44, 31: 45, 32: 46, 19: 33, 20: 34, 21: 35, 22: 36, 23: 37}]], 'num_bonds_total': 48, 'num_atoms_total': 47}





import numpy as np 
atom_map = {0:2, 1:3, 2:0, 3:1}
atom_map_reactant = atom_map.keys()
atom_map_reaction = atom_map.values()

atom_batch_index = [0,0,1,1,1,1,2,2,2] #based on molecule index.
mol_idx = 1


possible solutions:
coloum_number_reactant = np.where(atom_batch_index==mol_idx)

coloum_number_reactant= (array([2, 3, 4, 5]),)
then
coloum_number_reaction = [4,5,2,3]


import numpy as np

# Provided data
atom_map = {0:2, 1:3, 2:0, 3:1}
atom_map_reactant = atom_map.keys()
atom_map_reaction = atom_map.values()

atom_batch_index = np.array([0,0,1,1,1,1,2,2,2])
coloum_number = np.arange(0, len(atom_batch_index))
reactant_idx = 1

# Filtering indices for the specified reactant_idx
filtered_indices = np.where(atom_batch_index == reactant_idx)[0]

# Mapping the filtered indices to their new positions according to atom_map
coloum_number_after_map = [atom_map[idx] for idx in filtered_indices]

coloum_number_after_map







#global features:
import torch
reactants=None
products=None
mappings=None
has_bonds=None,
device=None,
ntypes=("global", "atom", "bond"),
ft_name="ft",
reverse=False,
reactant_only=False,
zero_fts=False,
empty_graph_fts=None,

verbose = False

#initilize reactants_ft and products_ft
reactants_ft, products_ft = [], []
feats = dict()

# note, this assumes we have one reactant
num_products = int(len(products))
num_reactants = int(len(reactants))

# take empty graph
graph = empty_graph_fts["empty_graph"]

nt = "global"

# get reactants_ft from molecule graph, but this should be done
# in a different way in batched graph version.
reactants_ft = [p.nodes[nt].data[ft_name] for p in reactants]
products_ft = [p.nodes[nt].data[ft_name] for p in products]

#assume has bond is true and assume num_products > 1

reactants_ft = [
    torch.sum(reactant_ft, dim=0, keepdim=True)
    for reactant_ft in reactants_ft
]
#!跟reactants处理方法一样
products_ft = [
    torch.sum(product_ft, dim=0, keepdim=True) for product_ft in products_ft]

len_feature_nt = reactants_ft[0].shape[1]

net_ft_full = torch.zeros(1, len_feature_nt).type_as(reactants_ft[0])

#if not zero_fits.
coef = torch.tensor([1]).type_as(reactants_ft[0])
if reverse == True:
    coef = torch.tensor([-1]).type_as(reactants_ft[0])

if reactant_only == False:
    for product_ft in products_ft:
        net_ft_full += coef * torch.sum(product_ft, dim=0, keepdim=True)

for reactant_ft in reactants_ft:
    #!(1x512)
    net_ft_full -= coef * torch.sum(reactant_ft, dim=0, keepdim=True)

feats[nt] = net_ft_full









batched_graph = dgl.batch(all_molecules)  # all_molecules contains both reactants and products
reactant_batch_indices = ...  # Tensor indicating which molecules are reactants
product_batch_indices = ...  # Tensor indicating which molecules are products










#bond feature
import itertools
import torch
import copy
reactants=None
products=None
mappings=None
has_bonds=None,
device=None,
ntypes=("global", "atom", "bond"),
ft_name="ft",
reverse=False,
reactant_only=False,
zero_fts=False,
empty_graph_fts=None,

verbose = False

#initilize reactants_ft and products_ft
reactants_ft, products_ft = [], []
feats = dict()

# note, this assumes we have one reactant
num_products = int(len(products))
num_reactants = int(len(reactants))

# take empty graph
graph = empty_graph_fts["empty_graph"]

nt = "bond"

# get reactants_ft from molecule graph, but this should be done
# in a different way in batched graph version.
reactants_ft = [p.nodes[nt].data[ft_name] for p in reactants]
products_ft = [p.nodes[nt].data[ft_name] for p in products]


if nt == "bond":
    if has_bonds == None:
        #print("manually constructing has bond maps")
        has_bonds = {
            "reactants": [True if len(mp) > 0 else False for mp in mappings["bond_map"][0]],
            "products": [True if len(mp) > 0 else False for mp in mappings["bond_map"][1]],
        }
        #print(has_bonds)
        #print(mappings["bond_map"])
        
    if num_products > 1:
        products_ft = list(
            itertools.compress(products_ft, has_bonds["products"])
        )
        filter_maps = list(
            itertools.compress(mappings[nt + "_map"][1], has_bonds["products"])
        )
        mappings[nt + "_map"] = [mappings[nt + "_map"][0], filter_maps]
        

        #if mappings[nt + "_map"] == [] and True in has_bonds["products"]:
        #    ind_t = has_bonds["products"].index(True)
        #    mappings[nt + "_map"] = mappings[nt + "_map"][0][ind_t]

    if num_reactants > 1:
        reactants_ft = list(
            itertools.compress(reactants_ft, has_bonds["reactants"])
        )
        filter_maps = list(
            itertools.compress(mappings[nt + "_map"][0], has_bonds["reactants"])
        )
        mappings[nt + "_map"] = [filter_maps, mappings[nt + "_map"][1]]




len_feature_nt = reactants_ft[0].shape[1]
net_ft_full = torch.zeros(mappings["num_bonds_total"], 
                          len_feature_nt).type_as(reactants_ft[0])
net_ft_full_zeros = copy.deepcopy(net_ft_full)

coef = torch.tensor([1]).type_as(reactants_ft[0])
if reverse == True:
    coef = torch.tensor([-1]).type_as(reactants_ft[0])

#if device is not None:
#    coef = coef.to(device)

# reactants
#print("maps full {}".format(mappings[nt + "_map"]))
#print("nt {}".format(nt))
#print("nt maps {}".format(mappings[nt + "_map"]))
for ind, reactant_ft in enumerate(reactants_ft):
    #print(mappings[nt + "_map"][0], ind)

    net_ft_full_temp = copy.deepcopy(net_ft_full_zeros)
    mappings_raw = mappings[nt + "_map"][0][ind]
    mappings_react = list(mappings_raw.keys())
    mappings_total = [
        mappings_raw[mapping] for mapping in mappings_react
    ]
    assert np.max(np.array(mappings_total)) < len(
        net_ft_full_temp
    ), f"invalid index  {mappings}"
    #print("reactant ft ", net_ft_full_temp.shape, reactant_ft.shape, len(mappings_react), len(mappings_total), num_reactants, nt)
    #net_ft_full_temp[mappings_total]
    #reactant_ft[mappings_react]
    net_ft_full_temp[mappings_total] = reactant_ft[mappings_react]
    net_ft_full[mappings_total] -= (
        coef * net_ft_full_temp[mappings_total]
    )

# products
if reactant_only == False:
    for ind, product_ft in enumerate(products_ft):
        net_ft_full_temp = copy.deepcopy(net_ft_full_zeros)
        #print(mappings[nt + "_map"])
        #print(mappings[nt + "_map"][1], ind)
        mappings_raw = mappings[nt + "_map"][1][ind]
        mappings_prod = list(mappings_raw.keys())
        mappings_total = [
            mappings_raw[mapping] for mapping in mappings_prod
        ]
        assert np.max(np.array(mappings_total)) < len(
            net_ft_full_temp
        ), f"invalid index  {mappings}"
        #print(product_ft.shape, len(mappings_prod), len(mappings_total), num_products, nt)
        net_ft_full_temp[mappings_total] = product_ft[mappings_prod]
        net_ft_full[mappings_total] += (
            coef * net_ft_full_temp[mappings_total]
        )

feats[nt] = net_ft_full






import numpy as np

# Example data
atom_map = {0: 2, 1: 1, 2: 0, 3: 3}
atom_batch_index = [0, 0, 1, 1, 1, 1, 2, 2, 2]
mol_idx = 1

#convert mapping to list
#atom_map_reactant = np.array(list(atom_map.keys()))
atom_map_reaction = np.array(list(atom_map.values()))

#find indices of reactant and map back to reaction
column_number_reactant = np.where(np.array(atom_batch_index) == mol_idx)[0]

column_number_reaction = atom_map_reaction[column_number_reactant - np.min(column_number_reactant)]

column_number_reaction += np.min(column_number_reactant)

print("Column numbers for reactants:", column_number_reactant)
print("Mapped column numbers for reactions:", column_number_reaction)







import numpy as np

# Example data
atom_map = {0: 5, 1: 4, 2: 3, 3: 6}
atom_batch_index = [0, 0, 1, 1, 1, 1, 2, 2, 2]
mol_idx = 1

#convert mapping to list
#atom_map_reactant = np.array(list(atom_map.keys()))
atom_map_reaction = np.array(list(atom_map.values()))

#find indices of reactant and map back to reaction
column_number_reactant = np.where(np.array(atom_batch_index) == mol_idx)[0]

column_number_reaction = atom_map_reaction[column_number_reactant - np.min(column_number_reactant)]

column_number_reaction += np.min(column_number_reactant)

print("Column numbers for reactants:", column_number_reactant)
print("Mapped column numbers for reactions:", column_number_reaction)













# import numpy as np

# # Example data
# atom_map = {0: 2, 1: 1, 2: 0, 3: 3}
# atom_batch_index = [0, 0, 1, 1, 1, 1, 2, 2, 2]
# mol_idx = 1

# atom_map_reaction = np.array(list(atom_map.values()))

# column_number_reactant = np.where(np.array(atom_batch_index) == mol_idx)[0]



# #This might now be true as reaction_part_1 may not complete
# column_number_reaction = atom_map_reaction[column_number_reactant - np.min(column_number_reactant)]


# column_number_reactant[mapping_array]

# column_number_reaction += np.min(column_number_reactant)

# print("Column numbers for reactants:", column_number_reactant)
# print("Mapped column numbers for reactions:", column_number_reaction)






import dgl
import torch

def create_batched_reaction_data(reactions):
    # Placeholder for batched graph construction. You will need to implement how individual graphs are aggregated.
    batched_graphs = dgl.batch([reaction['graph'] for reaction in reactions])

    # Initialize containers for the combined mappings and features
    combined_atom_map = []
    combined_bond_map = []
    combined_total_bonds = []
    combined_total_atoms = []

    # Iterate through reactions to adjust and aggregate mappings
    offset_atoms = 0  # Keeps track of the offset for atom indices
    offset_bonds = 0  # Keeps track of the offset for bond indices
    for reaction in reactions:
        atom_map = reaction['atom_map']
        bond_map = reaction['bond_map']
        total_atoms = reaction['total_atoms']
        total_bonds = reaction['total_bonds']
        
        # Adjust mappings with the current offset
        adjusted_atom_map = [{k+offset_atoms: v+offset_atoms for k, v in map_pair.items()} for map_pair in atom_map]
        adjusted_bond_map = [{k+offset_bonds: v+offset_bonds for k, v in map_pair.items()} for map_pair in bond_map]

        # Update the combined mappings
        combined_atom_map.extend(adjusted_atom_map)
        combined_bond_map.extend(adjusted_bond_map)

        # Adjust total_atoms and total_bonds with the current offset
        adjusted_total_atoms = [i + offset_atoms for i in total_atoms]
        adjusted_total_bonds = [i + offset_bonds for i in total_bonds]

        combined_total_atoms.extend(adjusted_total_atoms)
        combined_total_bonds.extend(adjusted_total_bonds)

        # Update offsets for the next iteration
        offset_atoms += max(total_atoms) + 1  # Assuming total_atoms is a list of atom indices
        offset_bonds += max(total_bonds) + 1  # Assuming total_bonds is a list of bond indices

    # Construct batched features based on adjusted mappings. This step will depend on how your features are structured
    # and may involve aggregating features from individual reactions into a combined tensor that aligns with the batched_graph.
    # Placeholder for feature construction
    batched_features = None  # Construct batched features based on the combined mappings and individual reaction features

    return batched_graphs, combined_atom_map, combined_bond_map, combined_total_atoms, combined_total_bonds, batched_features

# Assuming 'reactions' is your list of detailed reaction data
batched_graphs, combined_atom_map, combined_bond_map, combined_total_atoms, combined_total_bonds, batched_features = create_batched_reaction_data(reactions)