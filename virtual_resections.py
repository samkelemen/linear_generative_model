## Make resection figs ##
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import inout
from transformations import inverse_symmetric_modification

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
    # Instantiate this list to declare the order in which to iterate over the regions.
    labels = ['Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Accumbens', 'Amygdala',\
            'Hypocampus','Sommatosensory', 'Visual Cortex', 'DAN', 'SAN', 'Limbic', 'Cont', \
            'DMN']

    regions = {'Thalamus': (0, 7), 
            'Caudate': (1, 8), 
            'Putamen': (2, 9),
            'Pallidum': (3, 10),
            'Accumbens': (6, 13), 
            'Amygdala': (5, 12), 
            'Hypocampus': (4, 11),
            'Sommatosensory':list(range(24, 29)) + list(range(73, 80)), 
            'Visual Cortex': list(range(15, 23)) + list(range(65, 72)), 
            'DAN': list(range(30, 37)) + list(range(81, 87)), 
            'SAN': list(range(38, 44)) + list(range(88, 92)), 
            'Limbic': list(range(45, 47)) + list(range(93, 94)), 
            'Cont': list(range(48, 51)) + list(range(95, 103)), 
            'DMN': list(range(52, 64)) + list(range(104, 114))}

    # Create zero matrices to then edit.
    num_regions = len(labels)
    grouped_predfc = cp.zeros(shape=(num_regions, num_regions))

    # Iterate over the indices of the subregions in region1 and region2
    for (i, region1) in enumerate(labels):
        for (j, region2) in enumerate(labels):
            # For both regions in the pair, define the indices in that region
            region1_indices = regions[region1]
            region2_indices = regions[region2]            
            
            # Find the mean value for indices in region1 and region2 overlap
            region_mean = 0
            num_indices = len(region1_indices) * len(region2_indices)

            for index1 in region1_indices:
                for index2 in region2_indices:    
                    region_mean += predfc[index1][index2] / num_indices
            grouped_predfc[i][j] = region_mean
    return grouped_predfc

def mean_resects(indices_to_resect: list[int], resect_rules: bool=False) -> cp.ndarray:
    mean_predfc = cp.zeros((114, 114))
    for subject_id in range(1, 51):
        sc = inout.get_schaefer100_sc(subject_id)
        rules = cp.loadtxt(f"subject_level/rules_sub-{subject_id}")
        rules = inverse_symmetric_modification(rules, 114)

        if not resect_rules:
            sc = keep_indices(sc, indices_to_resect)

        predfc = (sc @ rules @ sc) / 50
        mean_predfc += predfc
    return mean_predfc

def signed_resects(indices_to_resect: list[int], resect_rules: bool=False) -> cp.ndarray:
    mean_pos_predfc = cp.zeros((114, 114))
    mean_neg_predfc = cp.zeros((114, 114))

    for subject_id in range(1, 51):
        sc = inout.get_schaefer100_sc(subject_id)
        rules = cp.loadtxt(f"subject_level/rules_sub-{subject_id}")
        rules = inverse_symmetric_modification(rules, 114)

        if not resect_rules:
            sc = keep_indices(sc, indices_to_resect)
        predfc = (sc @ rules @ sc)

        mean_pos_predfc[predfc > 0] += predfc[predfc > 0] / 50
        mean_neg_predfc[predfc < 0] += predfc[predfc < 0] / 50

    return mean_pos_predfc, mean_neg_predfc
             
def make_resections(indices_to_resect: list[int], title:str, outpath: str, resect_rules: bool=False):
    mean_predfc = mean_resects(indices_to_resect, resect_rules)
    mean_pos_predfc, mean_neg_predfc = signed_resects(indices_to_resect, resect_rules)

    inout.heatmaps(mean_predfc.get(), title, True, False)
    plt.savefig(outpath + "mean_predfc", bbox_inches='tight', dpi=500)
    plt.close()

    labels = ['Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Accumbens', 'Amygdala',\
        'Hypocampus','Sommatosensory', 'Visual Cortex', 'DAN', 'SAN', 'Limbic', 'Cont', \
        'DMN']
    
    grouped_predfc = group_by_regions(mean_predfc)
    grouped_pos_predfc = group_by_regions(mean_pos_predfc)
    grouped_neg_predfc = group_by_regions(mean_neg_predfc)

    grouped_diag = grouped_predfc.copy().diagonal().reshape((14, 1))
    grouped_diag_pos = grouped_pos_predfc.copy().diagonal().reshape((14, 1))
    grouped_diag_neg = grouped_neg_predfc.copy().diagonal().reshape((14, 1))

    cp.fill_diagonal(grouped_predfc, 0)
    cp.fill_diagonal(grouped_pos_predfc, 0)
    cp.fill_diagonal(grouped_neg_predfc, 0)
    
    plot_matrix(grouped_predfc, title, outpath + "mean", labels)
    plot_matrix(grouped_pos_predfc, title, outpath + "pos", labels, diag_info='pos')
    plot_matrix(grouped_neg_predfc, title, outpath + "neg", labels, diag_info='neg')
    plot_matrix(grouped_diag, title, outpath + "mean_diag", labels)
    plot_matrix(grouped_diag_pos, title, outpath + "pos_diag", labels, diag_info='pos')
    plot_matrix(grouped_diag_neg, title, outpath + "neg_diag", labels, diag_info='neg')

if __name__ == "__main__":
    regions = {'Thalamus': (0, 7), 
            'Caudate': (1, 8), 
            'Putamen': (2, 9),
            'Pallidum': (3, 10),
            'Accumbens': (6, 13), 
            'Amygdala': (5, 12), 
            'Hypocampus': (4, 11),
            'Sommatosensory':list(range(24, 29)) + list(range(73, 80)), 
            'Visual Cortex': list(range(15, 23)) + list(range(65, 72)), 
            'DAN': list(range(30, 37)) + list(range(81, 87)), 
            'SAN': list(range(38, 44)) + list(range(88, 92)), 
            'Limbic': list(range(45, 47)) + list(range(93, 94)), 
            'Cont': list(range(48, 51)) + list(range(95, 103)), 
            'DMN': list(range(52, 64)) + list(range(104, 114))}

    for region, indices in regions.items():
        print(f"Making resection figure for {region}")
        title = f"Resection of {region} SC"
        outpath = f"figures/resection/{region}/"
        inout.check_paths(outpath)

        make_resections(indices, title, outpath, resect_rules=False)
        #make_resections(indices, title, outpath, resect_rules=True)
