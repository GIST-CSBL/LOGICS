
class Object:
    pass

NJOBS_MULTIPROC = 8

PROJECT_PATHS = {
    "SMILES_TOKENS_PATH": "logics_pack/logics_tokens.txt",
    
    ### following data files are provided in GitHub repo
    "CHEMBL_RAW_PATH": "data/chembl/guacamol_v1_all.smiles",
    "KOR_RAW_PATH": "data/kor/data_clean_kop.csv",
    "PIK3CA_RAW_PATH": "data/pik3ca/pubchem_5290.txt"
}

def build_project_paths(project_dir="./"):
    project_paths = {}
    for key, path in PROJECT_PATHS.items():
        project_paths[key] = project_dir + path
    return project_paths


SMILES_TOKENS_PATH = "logics_pack/logics_tokens.txt"
# GENERATOR_PRETRAIN_JSON = ASDF

### following data files are provided in GitHub repo
CHEMBL_RAW_PATH = "data/guacamol_v1_all.smiles"
KOR_RAW_PATH = "data/data_clean_kop.csv"
PIK3CA_RAW_PATH = "data/pubchem_5290.txt"

### following files will be generated through experiments
