
class Object:
    pass

NJOBS_MULTIPROC = 8

project_paths = {
    "SMILES_TOKENS_PATH": "logics_pack/logics_tokens.txt",
    
    ### following data files are provided in GitHub repo
    "CHEMBL_RAW_PATH": "data/guacamol_v1_all.smiles",
    "KOR_RAW_PATH": "data/data_clean_kop.csv",
    "PIK3CA_RAW_PATH": "data/pubchem_5290.txt"
}

def init_project_paths(project_dir="./"):
    for key, path in project_paths.items():
        project_paths[key] = project_dir + path


SMILES_TOKENS_PATH = "logics_pack/logics_tokens.txt"
# GENERATOR_PRETRAIN_JSON = ASDF

### following data files are provided in GitHub repo
CHEMBL_RAW_PATH = "data/guacamol_v1_all.smiles"
KOR_RAW_PATH = "data/data_clean_kop.csv"
PIK3CA_RAW_PATH = "data/pubchem_5290.txt"

### following files will be generated through experiments
