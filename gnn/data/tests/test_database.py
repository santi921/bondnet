import pytest
from gnn.data.query_db import DatabaseOperation

# def test_query_database():
#     db = DatabaseOperation.from_query()
#     db.to_file(db.entries)


def test_get_job_types():
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    filename = "~/Applications/mongo_db_access/extracted_data/job_types.yml"
    db.get_job_types(filename)


def test_select(n=200):
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    filename = "~/Applications/mongo_db_access/extracted_data/database_n{}.pkl".format(n)
    db.to_file(filename, size=n)


def test_filter():
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    db.filter(keys=["formula_pretty"], value="LiH4(CO)3")
    db.to_file(filename="~/Applications/mongo_db_access/extracted_data/database_LiEC.pkl")


def test_create_dataset():

    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    mols = db.to_molecules(purify=True)
    # mols = mols[:6]
    db.create_sdf_csv_dataset(
        mols,
        "~/Applications/mongo_db_access/extracted_data/electrolyte_LiEC.sdf",
        "~/Applications/mongo_db_access/extracted_data/electrolyte_LiEC.csv",
    )
    print("entries saved:", len(mols))


def test_molecules():
    # db_path = "~/Applications/mongo_db_access/extracted_data/database_LiEC.pkl"
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    mols = db.to_molecules(optimized=True)
    for m in mols:
        # fname = '~/Applications/mongo_db_access/extracted_data/images/{}_{}.svg'.format(
        #     m.formula, m.id
        # )
        # m.draw(fname, show_atom_idx=True)

        fname = "/Users/mjwen/Applications/mongo_db_access/extracted_data/pdb/{}_{}.pdb".format(
            m.formula, m.id
        )
        m.write(fname, file_format="pdb")


if __name__ == "__main__":
    # test_select()
    # test_get_job_types()
    # test_filter()
    # test_create_dataset()
    test_molecules()
