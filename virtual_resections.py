"""
Virtual resection analysis and visualisation.

Functions here simulate removing brain regions from the structural
connectivity matrix and predict the downstream effect on functional
connectivity using a group of trained rule matrices.
"""
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import inout
from transformations import inverse_symmetric_modification
from brain_regions import BRAIN_REGIONS, REGION_LABELS
from constants import N_NODES, N_SUBJECTS

def keep_indices(matrix: cp.ndarray, indices_to_keep: list[int]) -> cp.ndarray:
    """
    Takes in a matrix, and returns a new matrix, that keeps only the entries
    involving the indices in indices_to_keep.
    """
    kept_matrix= cp.zeros(cp.shape(matrix))
    for indx in indices_to_keep:
        kept_matrix[indx, :] = matrix[indx, :]
        kept_matrix[:, indx] = matrix[:, indx]
    return kept_matrix

def plot_matrix(matrix: cp.ndarray, title, output_path, labels, diag_info=None):
    """
    Plots given resection matrix.
    """
    matrix = matrix.get()
    if not diag_info:
        cmap='vlag'
        ax = sns.heatmap(matrix, center=0, cmap="vlag", square=True, xticklabels=labels, yticklabels=labels, linewidth=0.5, linecolor='black')
    elif diag_info == 'neg':
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["#2F4F7F", "white"])
        ax = sns.heatmap(matrix, vmax=0, cmap=cmap, square=True, yticklabels=labels, linewidth=0.5, linecolor='black')
        plt.xticks(ticks=[], labels=[])
    elif diag_info == 'pos':
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "#8B0A1A"])
        ax = sns.heatmap(matrix, vmin=0, cmap=cmap, square=True, yticklabels=labels, linewidth=0.5, linecolor='black')
        plt.xticks(ticks=[], labels=[])

    ax.set_title(title)
    plt.savefig(output_path, bbox_inches='tight', dpi=500)
    plt.close()

def group_by_regions(predfc: cp.ndarray) -> cp.ndarray:
    """
    Aggregate a predicted-FC matrix into a (num_regions × num_regions) matrix
    by computing the mean value within each pair of brain-network regions.
    """
    num_regions = len(REGION_LABELS)
    grouped_predfc = cp.zeros(shape=(num_regions, num_regions))

    # Iterate over the indices of the subregions in region1 and region2
    for (i, region1) in enumerate(REGION_LABELS):
        for (j, region2) in enumerate(REGION_LABELS):
            region1_indices = BRAIN_REGIONS[region1]
            region2_indices = BRAIN_REGIONS[region2]

            # Find the mean value for indices in region1 and region2 overlap
            region_mean = 0
            num_indices = len(region1_indices) * len(region2_indices)

            for index1 in region1_indices:
                for index2 in region2_indices:
                    region_mean += predfc[index1][index2] / num_indices
            grouped_predfc[i][j] = region_mean
    return grouped_predfc

def mean_resects(indices_to_resect: list[int], resect_rules: bool=False) -> cp.ndarray:
    """
    Compute the across-subject mean predicted FC after virtually resecting
    the specified node indices.
    """
    mean_predfc = cp.zeros((N_NODES, N_NODES))
    for subject_id in range(1, N_SUBJECTS + 1):
        sc = inout.get_schaefer100_sc(subject_id)
        rules = cp.loadtxt(f"subject_level/rules_sub-{subject_id}")
        rules = inverse_symmetric_modification(rules, N_NODES)

        if not resect_rules:
            sc = keep_indices(sc, indices_to_resect)

        predfc = (sc @ rules @ sc) / N_SUBJECTS
        mean_predfc += predfc
    return mean_predfc

def signed_resects(indices_to_resect: list[int], resect_rules: bool=False) -> cp.ndarray:
    """
    Compute separate positive and negative components of the mean predicted FC
    after virtually resecting the specified nodes.
    """
    mean_pos_predfc = cp.zeros((N_NODES, N_NODES))
    mean_neg_predfc = cp.zeros((N_NODES, N_NODES))

    for subject_id in range(1, N_SUBJECTS + 1):
        sc = inout.get_schaefer100_sc(subject_id)
        rules = cp.loadtxt(f"subject_level/rules_sub-{subject_id}")
        rules = inverse_symmetric_modification(rules, N_NODES)

        if not resect_rules:
            sc = keep_indices(sc, indices_to_resect)
        predfc = (sc @ rules @ sc)

        mean_pos_predfc[predfc > 0] += predfc[predfc > 0] / N_SUBJECTS
        mean_neg_predfc[predfc < 0] += predfc[predfc < 0] / N_SUBJECTS

    return mean_pos_predfc, mean_neg_predfc

def make_resections(indices_to_resect: list[int], title:str, outpath: str, resect_rules: bool=False):
    """
    Orchestrate resection visualisation for a set of node indices: compute
    mean and signed predicted FCs, group them by brain network, and save
    all resulting heatmap figures to outpath.
    """
    mean_predfc = mean_resects(indices_to_resect, resect_rules)
    mean_pos_predfc, mean_neg_predfc = signed_resects(indices_to_resect, resect_rules)

    inout.heatmaps(mean_predfc.get(), title, True, False)
    plt.savefig(outpath + "mean_predfc", bbox_inches='tight', dpi=500)
    plt.close()

    num_regions = len(REGION_LABELS)
    grouped_predfc = group_by_regions(mean_predfc)
    grouped_pos_predfc = group_by_regions(mean_pos_predfc)
    grouped_neg_predfc = group_by_regions(mean_neg_predfc)

    grouped_diag = grouped_predfc.copy().diagonal().reshape((num_regions, 1))
    grouped_diag_pos = grouped_pos_predfc.copy().diagonal().reshape((num_regions, 1))
    grouped_diag_neg = grouped_neg_predfc.copy().diagonal().reshape((num_regions, 1))

    cp.fill_diagonal(grouped_predfc, 0)
    cp.fill_diagonal(grouped_pos_predfc, 0)
    cp.fill_diagonal(grouped_neg_predfc, 0)

    plot_matrix(grouped_predfc, title, outpath + "mean", REGION_LABELS)
    plot_matrix(grouped_pos_predfc, title, outpath + "pos", REGION_LABELS, diag_info='pos')
    plot_matrix(grouped_neg_predfc, title, outpath + "neg", REGION_LABELS, diag_info='neg')
    plot_matrix(grouped_diag, title, outpath + "mean_diag", REGION_LABELS)
    plot_matrix(grouped_diag_pos, title, outpath + "pos_diag", REGION_LABELS, diag_info='pos')
    plot_matrix(grouped_diag_neg, title, outpath + "neg_diag", REGION_LABELS, diag_info='neg')

if __name__ == "__main__":
    for region, indices in BRAIN_REGIONS.items():
        print(f"Making resection figure for {region}")
        title = f"Resection of {region} SC"
        outpath = f"figures/resection/{region}/"
        inout.check_paths(outpath)

        make_resections(indices, title, outpath, resect_rules=False)
        #make_resections(indices, title, outpath, resect_rules=True)
