import pandas as pd 

def get_dataset_species(molecules):
    """
    Get all the species of atoms appearing in the the molecules.

    Args:
        molecules (list): rdkit molecules

    Returns:
        list: a sequence of species string
    """
    system_species = set()
    for mol in molecules:
        if mol is None:
            continue
        species = [a.GetSymbol() for a in mol.GetAtoms()]
        system_species.update(species)

    return sorted(system_species)

def get_dataset_species_from_json(json_file):
    """
    Get all the species of atoms appearing in the the molecules.

    Args:
        json_file: file containing reaction w/ column

    Returns:
        list: a sequence of species string
    """
    df = pd.read_json(json_file)
    
    list(df['composition'][0].keys())

    system_species = set()
    for _, row in df.iterrows():
        if row is None:
            continue
        species = list(row['composition'].keys())
        system_species.update(species)

    return sorted(system_species)
