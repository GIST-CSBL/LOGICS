"""
    This file includes:
    1. functions processing the generated smiles and saving subsidiary files for evaluation
    2. functions performing the evaluation of the generations
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from . import analysis, chemistry

def sample_file_processing(sample_fmt, vacan_fmt, npfps_fmt, epochs):
    """
        Gets sample_fmt (.txt with epoch wildcard), 
        1. writes files with valid and canonical smiles (.smi),
        2. writes files with morgan fingerprint numpy array (.npy).
        epochs should be a list of integers.
    """
    for epo in epochs:
        with open(sample_fmt%epo, 'r') as f:
            samples = [line.strip() for line in f.readlines()]
        vacans, _ = chemistry.get_valid_canons(samples)
        mgfps = chemistry.get_fps_from_smilist(vacans)
        npfps = np.array(mgfps)
        with open(vacan_fmt%epo, 'w') as f:
            f.writelines([line+'\n' for line in vacans])
        np.save(npfps_fmt%epo, npfps)

def intsimmat1k_save(npfps_fmt, simmat_fmt, epochs):
    """
        Gets npfps_fmt (.npy with epoch wildcard), 
        writes files with internal similarity matrix (.npy).
        We are only interested in first 1k samples.
        epochs should be a list of integers.
    """
    for epo in epochs:
        npfps = np.load(npfps_fmt%epo)
        # extract first 1k
        npfps1k = npfps[:1000]
        mgfps1k = chemistry.np2rdkfps(npfps1k)
        # internal similarity matrix
        simmat = analysis.calculate_simmat(mgfps1k, mgfps1k)
        np.save(simmat_fmt%epo, simmat)

def extsimmat1k_save(npfps_fmt, ua_fps, simmat_fmt, epochs):
    """
        npfps_fmt (.npy) -> file containing generations' fingerprints
        ua_fps is the unseen actives fingerprints.
        We are only interested in the first 1k samples.
        rows: generations, columns: unseen actives
    """
    for epo in epochs:
        npfps = np.load(npfps_fmt%epo)
        # extract first 1k
        npfps1k = npfps[:1000]
        mgfps1k = chemistry.np2rdkfps(npfps1k)
        # external similarity
        simmat = analysis.calculate_simmat(mgfps1k, ua_fps)
        np.save(simmat_fmt%epo, simmat)

def predact_save(npfps_fmt, predictor, predact_fmt, epochs):
    """
        npfps_fmt (.npy) -> file containing generations' fingerprints
        Calculated predicted activity.
    """
    for epo in epochs:
        npfps = np.load(npfps_fmt%epo)
        preds = predictor.predict(npfps)
        np.save(predact_fmt%epo, preds)

def transport_distmat_save(ts_to_dist, npfps_fmt, ua_fps, distmat_fmt, epochs, num_repeats=1):
    """
        ts_to_dist - function that transforms a Tanimoto similarity to transport distance (matrix operation)
        npfps_fmt - generations' fingerprints file path with epoch wildcard
        ua_fps - unseen actives fingerprints
        num_repeats - for one generation file, we can only use the amount of molecules up to demands.
            Usually, the demands (size of unseen actives) are far smaller than the generation size, 
            so we can split the generation file to get many repeated transportation mapping results.
            If the amount of available supply is smaller than demand*repeat, 
            then the repeat value is flexibly adjusted.
        rows: generations, columns: unseen actives
    """
    for epo in epochs:
        dsize = len(ua_fps)
        npfps = np.load(npfps_fmt%epo)
        total_supply = len(npfps)
        # check the available supply, and adjust the number of repeats
        if total_supply < dsize*num_repeats:
            num_repeats = int(total_supply/dsize)
        supply_size = dsize*num_repeats
        # extract (repeats*demand)
        sufps = npfps[:supply_size]
        sufps = chemistry.np2rdkfps(sufps)
        # external similarity
        simmat = analysis.calculate_simmat(sufps, ua_fps)
        # tanimoto distance
        distmat = ts_to_dist(simmat)
        np.save(distmat_fmt%epo, distmat)

def optimal_transport_save(distmat_fmt, rinl_fmt, cinl_fmt, totds_fmt, epochs):
    """
        Three files are saved for each epoch:
            - rinl (.npy): row index nested list (i-th elements are row indices for the i-th transport mapping)
            - cinl (.npy): column index nested list (i-th elements are column indices for i-th transport mapping)
            - totds (.npy): 1-d array of totd of each repeat
        repeats are calculated dynamically.
    """
    for epo in epochs:
        distmat = np.load(distmat_fmt%epo)
        rs, cs = distmat.shape
        repeat = int(rs/cs)
        row_ind_nest, col_ind_nest, totds = analysis.repeated_optimal_transport(distmat, repeat)
        np.save(rinl_fmt%epo, np.array(row_ind_nest))
        np.save(cinl_fmt%epo, np.array(col_ind_nest))
        np.save(totds_fmt%epo, np.array(totds))
        
def euclidean_transport_distmat_save(g_vectors_fmt, ua_vectors, distmat_fmt, epochs):
    """
        Calculate Euclidean distance matrix between 
        generation vectors and unseen active vectors.
    """
    for epo in epochs:
        dsize = len(ua_vectors)
        g_vecs = np.load(g_vectors_fmt%epo)[:dsize]
        distmat = distance_matrix(g_vecs, ua_vectors)
        np.save(distmat_fmt%epo, distmat)

def evaluation_basic(sample_size, vacans, pretrainset):
    """
        evaluate Validity, Uniqueness, Novelty
        - vacans: list of valid & canonical smiles
        - sample_size: # of the generator samples
    """
    validity = len(vacans) / sample_size
    unis = list(set(vacans))
    uniqueness = len(unis) / len(vacans)
    novs = list(set(unis).difference(set(pretrainset)))
    novelty = len(novs) / len(unis)
    return validity, uniqueness, novelty
    
def standard_metrics(epochs, sample_size, vc_fmt, pretrainset, intsimmat_fmt):
    vs, us, ns, ds = [], [], [], []
    for epo in epochs:
        with open(vc_fmt%epo, 'r') as f:
            vacans = [line.strip() for line in f.readlines()]
        v, u, n = evaluation_basic(sample_size, vacans, pretrainset)
        simmat = np.load(intsimmat_fmt%epo)
        d = (1-simmat).mean()
        vs.append(v)
        us.append(u)
        ns.append(n)
        ds.append(d)
    metdf = pd.DataFrame(epochs, columns=['epoch'])
    metdf['validity'] = vs
    metdf['uniqueness'] = us
    metdf['novelty'] = ns
    metdf['diversity'] = ds
    return metdf

def optimization_metrics(epochs, predact_fmt, extsim1k_fmt, totds_fmt, fcd_path, demand_size):
    """
        totds_fmt - save path for totd (total optimal transport distance) list, repeated n times.
        FCD should have been calculated and saved elsewhere.
        demand_size should simply be the number of unseen actives.
    """
    acts, sims, exp_motds, std_motds = [], [], [], []
    for epo in epochs:
        acts.append(np.load(predact_fmt%epo).mean())
        sims.append(np.load(extsim1k_fmt%epo).mean())
        
        totds = np.load(totds_fmt%epo)
        motds = totds / demand_size # convert total distance to mean distance
        exp_motd = np.mean(motds) # expected value for motd
        std_motd = np.std(motds) # standard deviation for motd values
        exp_motds.append(exp_motd)
        std_motds.append(std_motd)

    metdf = pd.DataFrame(epochs, columns=['epoch'])
    metdf['PredAct'] = acts
    metdf['PwSim'] = sims
    metdf['FCD'] = np.load(fcd_path)
    metdf['exp_mOTD'] = exp_motds # mean optimal transport distance
    metdf['std_mOTD'] = std_motds
    return metdf
