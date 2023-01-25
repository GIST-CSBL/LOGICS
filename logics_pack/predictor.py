import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from . import chemistry

def featurizer(smiles):
    """
        smiles: should be VALID smiles list
        This function returns the Morgan Fingerprints numpy array.
    """
    return np.array(chemistry.get_fps_from_smilist(smiles))

def train_predictor(config):
    """
        Train the RFR predictor with rdkit fingerprint.
        Note that one of the folds will be used as a test fold.
        Necessary members in config:
            config.affinity_path (csv)
            config.fingerprint_path (npy)
            config.fold_path (json)
            config.test_fold_id (str) <- json files will have str as key e.g. "0"
    """
    affinity_df = pd.read_csv(config.affinity_path)
    pred_labels = np.array(affinity_df['affinity'])
    fp_features = np.load(config.fingerprint_path)

    with open(config.fold_path, 'r') as f:
        fold_split = json.load(f)
    tf_ids = np.array(fold_split[config.test_fold_id]) # test fold data indices
    vf_keys = list(fold_split.keys())
    vf_keys.remove(config.test_fold_id)
    print(vf_keys)

    rfr_fold = []
    vmse = [] # validation mse
    vr2 = [] # validation r2
    for i, key in enumerate(vf_keys):
        # i-th fold is used as a validation set.
        vf_ids = np.array(fold_split[key]) # validation fold
        v_labels = pred_labels[vf_ids]
        v_features = fp_features[vf_ids]

        nt_ids = np.append(tf_ids, vf_ids) # non-training indices
        tr_labels = np.delete(pred_labels.copy(), nt_ids, axis=0)
        tr_features = np.delete(fp_features.copy(), nt_ids, axis=0)

        # train RFR in a default setup
        rfr = RandomForestRegressor(n_estimators=100) # n_estimators=100 is default for sklearn version>=0.22
        rfr.fit(tr_features, tr_labels)
        rfr_fold.append(rfr)

        # validation performance
        v_preds = rfr.predict(v_features)
        vmse.append(mean_squared_error(v_labels, v_preds))
        vr2.append(r2_score(v_labels, v_preds))

    return rfr_fold, vmse, vr2, vf_keys
