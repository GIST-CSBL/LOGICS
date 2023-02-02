import json

class Object:
    pass

NJOBS_MULTIPROC = 8

NUM_DATA_FOLDS = 6
TEST_FOLD_IDX = 5

PROJECT_PATHS = {
    "SMILES_TOKENS_PATH": "logics_pack/logics_tokens.txt",
    "EXPERIMENT_SETTINGS_JSON": "logics_pack/experiment_settings.json",
    "PRETRAIN_SETTING_JSON": "logics_pack/pretrain_setting.json",
    
    ### following data files are provided in GitHub repo
    "CHEMBL_RAW_PATH": "data/chembl/guacamol_v1_all.smiles",
    "KOR_RAW_PATH": "data/kor/data_clean_kop.csv",
    "PIK3CA_RAW_PATH": "data/pik3ca/pubchem_5290.txt",

    ### following files will be generated through experiments
    ## initial data
    "CHEMBL_DATA_PATH": "data/chembl/chembl_new.smi",
    "KOR_DATA_PATH": "data/kor/kor_affinity_new.csv",
    "PIK3CA_DATA_PATH": "data/pik3ca/pik3ca_affinity_new.csv",
    "KOR_FOLD_JSON": "data/kor/kor_fold_splits.json",
    "PIK3CA_FOLD_JSON": "data/pik3ca/pik3ca_fold_splits.json",
    "KOR_DATA_FP": "data/kor/kor_aff_npfps.npy",
    "PIK3CA_DATA_FP": "data/pik3ca/pik3ca_aff_npfps.npy",

    ## pre-training phase
    "PRETRAINING_DATA_PATH": "data/chembl/pre-training.smi"
}

def build_project_paths(project_dir="./"):
    project_paths = {}
    for key, path in PROJECT_PATHS.items():
        project_paths[key] = project_dir + path
    project_paths["PROJECT_DIR"] = project_dir
    return project_paths

class ExperimentSettings:
    def __init__(self, expset_json_path):
        with open(expset_json_path, 'r') as f:
            self.expset = json.load(f)
        self.path = expset_json_path
    
    def get_keys(self):
        return self.expset.keys()
    
    def update_setting(self, key, item):
        # int will be converted to str
        self.expset[key] = str(item)
        with open(self.path, 'w') as f:
            json.dump(self.expset, f, indent=2)

    def get_setting(self, key):
        return self.expset[key]
