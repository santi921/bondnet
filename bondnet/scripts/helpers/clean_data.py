import pandas as pd
import ast, json


def main():
    path_df = "../../dataset/" + "protonated_hydrolysis_reactions_complete.json"
    hydro_df = pd.read_json(path_df)

    product_list, water_list, reac_list = [], [], []
    product_bond_list, water_bond_list, reac_bond_list = [], [], []
    product_pymat_list, reactant_pymat_list = [], []
    comp_list, combined_product_list, combined_reactant_list = [], [], []
    print(hydro_df.shape)
    for ind, row in hydro_df.iterrows():
        # print(ind)
        try:
            reac_graph_mol = json.loads('"' + row["reactant_molecule_graph"] + '"')
        except:
            try:
                reac_graph_mol = json.loads(row["reactant_molecule_graph"])[0]
            except:
                print(ind)
        try:
            prod_graph_mol = json.loads('"' + row["product_molecule_graph"] + '"')
        except:
            try:
                prod_graph_mol = json.loads(row["product_molecule_graph"])[0]
            except:
                print(ind)
        reactant_pymat_list.append(reac_graph_mol)
        product_pymat_list.append(prod_graph_mol)

        product_bonds = row.product_bonds
        reactant_bonds = row.reactant_bonds

        bonds_formed = []
        bonds_broken = []
        product_bonds = row.product_bonds
        reactant_bonds = row.reactant_bonds

        # product_bonds = ast.literal_eval(row.product_bonds)
        # reactant_bonds = ast.literal_eval(row.reactant_bonds)
        reactant_bonds = row.reactant_bonds
        product_bonds = row.product_bonds
        reac_mol = row.reactant_structure
        water_mol = row.water_structure
        product_mol = row.product_structure
        water_bonds = row.water_bonds
        comp_dict = row.reactant_composition

        if type(product_bonds) == str:
            product_bonds = ast.literal_eval(product_bonds)
        if type(reactant_bonds) == str:
            reactant_bonds = ast.literal_eval(reactant_bonds)
        if type(reac_mol) == str:
            reac_mol = ast.literal_eval(row.reactant_structure)
        if type(water_mol) == str:
            water_mol = ast.literal_eval(row.water_structure)
        if type(product_mol) == str:
            product_mol = ast.literal_eval(row.product_structure)
        if type(water_bonds) == str:
            water_bonds = ast.literal_eval(row.water_bonds)
        if type(comp_dict) == str:
            comp_dict = ast.literal_eval(row.reactant_composition)

        for i in reactant_bonds:
            if (i not in product_bonds) or ([i[-1], i[0]] not in product_bonds):
                bonds_broken.append(i)
        for i in product_bonds:
            if (i not in reactant_bonds) or ([i[-1], i[0]] not in reactant_bonds):
                bonds_formed.append(i)

        # edit

        comp_list.append(comp_dict)
        reac_list.append(reac_mol)
        water_list.append(water_mol)
        product_list.append(product_mol)
        product_bond_list.append(product_bonds)
        water_bond_list.append(water_bonds)
        reac_bond_list.append(reactant_bonds)

        combined_product_list.append(ast.literal_eval(str(row.combined_products_graph)))
        combined_reactant_list.append(
            ast.literal_eval(str(row.combined_reactants_graph))
        )

    hydro_df["reactant_structure"] = reac_list
    hydro_df["water_structure"] = water_list
    hydro_df["product_structure"] = product_list
    hydro_df["reactant_bonds"] = reac_bond_list
    hydro_df["water_bonds"] = water_bond_list
    hydro_df["product_bonds"] = product_bond_list
    hydro_df["composition"] = comp_list
    hydro_df["reactant_composition"] = comp_list
    hydro_df["combined_reactants_graph"] = combined_product_list
    hydro_df["combined_products_graph"] = combined_reactant_list
    hydro_df["composition"] = hydro_df["reactant_composition"]
    hydro_df["reactant_bonds_nometal"] = hydro_df["reactant_bonds"]
    hydro_df["product_bonds_nometal"] = hydro_df["product_bonds"]
    # hydro_df["charge"] = hydro_df.reactant_charge

    # to overwrite the old file
    hydro_df.to_json(path_df)


main()
