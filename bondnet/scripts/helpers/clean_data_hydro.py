import pandas as pd
import ast, json


def get_bonds_changes(reactant_bonds, product_bonds):
    bonds_formed = []
    bonds_broken = []
    for i in reactant_bonds:
        if (i not in product_bonds) or ([i[-1], i[0]] not in product_bonds):
            bonds_broken.append(i)
    for i in product_bonds:
        if (i not in reactant_bonds) or ([i[-1], i[0]] not in reactant_bonds):
            bonds_formed.append(i)
    return bonds_formed, bonds_broken


def get_mol_graphs(row, ind):
    try:
        reac_graph_mol = json.loads('"' + row["reactant_molecule_graph"] + '"')
    except:
        try:
            reac_graph_mol = json.loads(row["reactant_molecule_graph"])[0]
        except:
            try:
                reac_graph_mol = row["reactant_molecule_graph"]
            except:
                print("reactant: " + str(ind))
                reac_graph_mol = None
    try:
        prod_graph_mol = json.loads('"' + row["product_molecule_graph"] + '"')
    except:
        try:
            prod_graph_mol = json.loads(row["product_molecule_graph"])[0]
        except:
            try:
                prod_graph_mol = row["product_molecule_graph"]
            except:
                print("product: " + str(ind))
                prod_graph_mol = None

    return reac_graph_mol, prod_graph_mol


def main():
    overwrite = False
    path_df = "/home/santiagovargas/dev/bondnet/bondnet/dataset/HEPOM/Holdout_set_protonated_1000.json"
    hydro_df = pd.read_json(path_df)

    product_list, water_list, reac_list = [], [], []
    product_bond_list, water_bond_list, reac_bond_list = [], [], []
    product_pymat_list, reactant_pymat_list = [], []
    comp_list, combined_product_list, combined_reactant_list = [], [], []
    bonds_formed_list, bonds_broken_list = [], []
    reactant_sym_list, product_sym_list = [], []
    reactant_element_list = []

    for ind, row in hydro_df.iterrows():
        reac_graph_mol, prod_graph_mol = get_mol_graphs(row, ind)
        reactant_pymat_list.append(reac_graph_mol)
        product_pymat_list.append(prod_graph_mol)
        # if there is a column called combined_product_bonds_global us that instead
        if "combined_product_bonds_global" in hydro_df.columns:
            product_bonds = row.combined_product_bonds_global
        else:
            product_bonds = row.product_bonds
        if "combined_reactant_bonds_global" in hydro_df.columns:
            reactant_bonds = row.combined_reactant_bonds_global
        else:
            reactant_bonds = row.reactant_bonds

        combined_product_list.append(ast.literal_eval(str(row.combined_products_graph)))

        combined_reactant_list.append(
            ast.literal_eval(str(row.combined_reactants_graph))
        )
        reac_mol = row.reactant_structure
        water_mol = row.water_structure
        water_bonds = row.water_bonds
        comp_dict = row.reactant_composition
        product_mol = row.product_structure
        if "reactant_symmetry" in hydro_df.columns:
            reactant_symmetry = row.reactant_symmetry
            if type(reactant_symmetry) == str:
                reactant_symmetry = ast.literal_eval(reactant_symmetry)
            reactant_sym_list.append(reactant_symmetry)

        if "product_symmetry" in hydro_df.columns:
            product_symmetry = row.product_symmetry
            if type(product_symmetry) == str:
                product_symmetry = ast.literal_eval(product_symmetry)
            product_sym_list.append(product_symmetry)

        reactant_elements = row.reactant_elements

        if type(reactant_bonds) == str:
            reactant_bonds = ast.literal_eval(reactant_bonds)
        if type(reac_mol) == str:
            reac_mol = ast.literal_eval(row.reactant_structure)
        if type(water_mol) == str:
            water_mol = ast.literal_eval(row.water_structure)
        if type(water_bonds) == str:
            water_bonds = ast.literal_eval(row.water_bonds)
        if type(comp_dict) == str:
            comp_dict = ast.literal_eval(row.reactant_composition)
        if type(product_bonds) == str:
            product_bonds = ast.literal_eval(product_bonds)
        if type(product_mol) == str:
            product_mol = ast.literal_eval(row.product_structure)
        if type(reactant_elements) == str:
            reactant_elements = ast.literal_eval(reactant_elements)

        bonds_formed, bonds_broken = get_bonds_changes(reactant_bonds, product_bonds)

        reactant_element_list.append(reactant_elements)
        comp_list.append(comp_dict)
        water_bond_list.append(water_bonds)
        water_list.append(water_mol)
        reac_list.append(reac_mol)
        reac_bond_list.append(reactant_bonds)
        # combined_reactant_list.append(
        #    ast.literal_eval(str(row.combined_reactants_graph))
        # )

        bonds_formed_list.append(bonds_formed)
        bonds_broken_list.append(bonds_broken)
        product_list.append(product_mol)
        product_bond_list.append(product_bonds)
        # combined_product_list.append(ast.literal_eval(str(row.combined_products_graph)))

    hydro_df["water_structure"] = water_list  # fine
    hydro_df["water_bonds"] = water_bond_list  # fine
    hydro_df["reactant_structure"] = reac_list  # fine
    hydro_df["reactant_bonds"] = reac_bond_list  # fine
    hydro_df["reactant_bonds_nometal"] = reac_bond_list  # fine
    hydro_df["product_structure"] = product_list  # fine
    hydro_df["product_bonds"] = product_bond_list  # fine
    hydro_df["product_bonds_nometal"] = product_bond_list  # fine
    hydro_df["composition"] = comp_list  # fine
    hydro_df["reactant_composition"] = comp_list  # fine
    hydro_df["composition"] = hydro_df["reactant_composition"]  # fine
    hydro_df["charge"] = hydro_df.reactant_charge  # fine
    hydro_df["bonds_formed"] = bonds_formed_list  # fine
    hydro_df["bonds_broken"] = bonds_broken_list  # fine

    hydro_df["reactant_elements"] = reactant_element_list  # fine
    hydro_df["combined_reactants_graph"] = combined_reactant_list  # fine
    hydro_df["combined_products_graph"] = combined_product_list  # fine
    hydro_df["product_molecule_graph"] = product_pymat_list  # fine
    hydro_df["reactant_molecule_graph"] = reactant_pymat_list  # fine

    if "reactant_symmetry" in hydro_df.columns:
        hydro_df["reactant_symmetry"] = reactant_sym_list  # fine
    if "product_symmetry" in hydro_df.columns:
        hydro_df["product_symmetry"] = product_sym_list  # fine

    # to overwrite the old file
    if overwrite:
        hydro_df.to_json(path_df)
    else:
        hydro_df.to_json(path_df[:-5] + "_cleaned.json")


main()
