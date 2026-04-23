import cupy as cp
import numpy as np
import os
from scipy.stats import t


def calc_fc(ts_data, time_steps, starting_time):
    """
    Calculate the functional connectivity matrix from time series data.
    """
    n = ts_data.shape[1]
    fc_matrix = cp.zeros((n, n))
    fc_matrix = cp.corrcoef(ts_data[starting_time:time_steps + starting_time, :], rowvar=False)
    return fc_matrix.astype(cp.float32)


def load_connectomes(subject_id, time_steps=-1, starting_time=0):
    """
    Load the data for a given subject and window ratio.
    """
    # Data paths
    function_ts_path = f"ts_data/iPA_183/ts/ts_sub-{subject_id}_183.txt"
    sc_path = f"ts_data/iPA_183/sc/sub-{subject_id}_SC.csv"

    # Get sc & fc matrices and return them
    ts_data = cp.loadtxt(function_ts_path)
    fc = calc_fc(ts_data, time_steps, starting_time)
    sc = cp.loadtxt(sc_path)
    cp.fill_diagonal(sc, cp.mean(sc))
    return sc.astype(cp.float32), fc


def compute_pvals(distribution, matrix):
    n = matrix.shape[0]
    df = n - 1

    mean = distribution.mean(axis=1)
    std = distribution.std(axis=1, ddof=1)
    t_stat = (mean - matrix) / ((std + 1e-64) / np.sqrt(n))
    p_values = 2 * t.sf(np.abs(t_stat), df)
    return p_values


def compute_sig_rules(p_values, matrix):
    matrix[p_values > 0.05] = np.nan
    return matrix


def get_mat(path, subject_id):
    import h5py
    # Open the HDF5 file
    with h5py.File("27_SCHZ_CTRL_dataset.mat", "r") as f:
        # Access the object reference datasets
        refs = f[path]

        # Dereference the objects
        data = [cp.array(f[ref]) for ref in refs[:,0]][1][subject_id - 1]
    return data
