"""
This file handles all input/output for the program.
"""
import os
from pathlib import Path
import seaborn as sns
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import cupy as cp


def check_paths(paths) -> None:
    """Checks if the paths exist, and if not, creates them."""
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                Path(path).mkdir(parents=True, exist_ok=True)
    elif isinstance(paths, str):
        Path(paths).mkdir(parents=True, exist_ok=True)

def heatmaps(matrix: NDArray[np.float64], title: str, colored: bool, bounded: bool=False) -> Figure:
    """Creates heatmaps for the matrices."""
    # Create the axes for the plot
    ax = plt.axes()
    ax.set_title(title)
    # Define the color map
    cmap = 'vlag' if colored else 'Greys'
    center = 0 if colored else None
    # Create the heatmap
    if bounded:
        sns.heatmap(matrix, vmin=-1, vmax=1, center=center, cmap=cmap, square=True, ax=ax)
    else:
        sns.heatmap(matrix, center=center, cmap=cmap, square=True, ax=ax)
    # Return the plot
    return ax

def get_schaefer100_data(path: str) -> cp.ndarray:
    """
    Base function for getting scaefer data. Used by get_schaefer100_fc 
    and get_schaefer100_sc
    """
    data = cp.genfromtxt(path, delimiter=',')
    data_complement = cp.copy(data.T)
    cp.fill_diagonal(data_complement, 0)
    data += data_complement
    data = remove_medial_wall(data)
    return data

def get_schaefer100_fc(subject_id: str) -> cp.ndarray:
    """Gets the functional connnectivity matrix."""
    subject_id = f'{subject_id:03}' # Sets  the subject id to 3 digits
    path = f'data/micapipe/sub-HC{subject_id}/ses-01/'
    fc_path = path + f'func/sub-HC{subject_id}_ses-01_space-fsnative_atlas-schaefer100_desc-fc.txt'
    sfc100 = get_schaefer100_data(fc_path)
    cp.fill_diagonal(sfc100, 1)
    return sfc100

def get_schaefer100_sc(subject_id: str) -> cp.ndarray:
    """Gets the structural connectivity matrix."""
    subject_id = f'00{subject_id}' if subject_id < 10 else f'0{subject_id}'
    path = f'data/micapipe/sub-HC{subject_id}/ses-01/'
    sc_path = path + f'dwi/sub-HC{subject_id}_ses-01_space-dwinative_atlas-schaefer100_desc-sc.txt'
    ssc100 = get_schaefer100_data(sc_path)
    return ssc100

def remove_medial_wall(matrix: cp.ndarray) -> cp.ndarray:
    """
    Remove medial wall from the inputted matrix.
    """
    matrix = cp.delete(matrix, 65, 0)
    matrix = cp.delete(matrix, 65, 1)
    matrix = cp.delete(matrix, 14, 0)
    matrix = cp.delete(matrix, 14, 1)
    return matrix