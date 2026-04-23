import cupy as cp
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from transformations import inverse_symmetric_modification
from train_functions import calc_fc


def make_validation_plots(subject_id):
    fig1, axs_both_windows = plt.subplots(2, 4, figsize=(20, 10))
    axs_both_windows = axs_both_windows.flatten()

    fig2, axs_same_window = plt.subplots(2, 4, figsize=(20, 10))
    axs_same_window = axs_same_window.flatten()

    fig3, axs_diff_window = plt.subplots(2, 4, figsize=(20, 10))
    axs_diff_window = axs_diff_window.flatten()

    for plot_indx, window_size in enumerate((16, 65, 98, 130, 196, 326, 587)):
        # Data paths
        function_ts_path = f"ts_data/iPA_183/ts/ts_sub-{subject_id}_183.txt"
        ts_data = cp.loadtxt(function_ts_path)

        fc_same_window = calc_fc(ts_data, window_size, 0)
        fc_diff_window = calc_fc(ts_data, window_size, 652-window_size)

        sc_path = f"ts_data/iPA_183/sc/sub-{subject_id}_SC.csv"
        sc = cp.loadtxt(sc_path)
        cp.fill_diagonal(sc, 0)

        r2s_same_window = []
        r2s_diff_window = []
        sc_densities = []
        for indx in range(20):
            rules_path = f"sc_density_sweeps/{window_size}/rules_sub-{subject_id}_lambda-{indx}"
            rules = cp.loadtxt(rules_path)
            rules = inverse_symmetric_modification(rules, 183)

            predfc = sc @ rules @ sc

            r2_same_window = r2_score(fc_same_window.get().flatten(), predfc.get().flatten())
            r2s_same_window.append(r2_same_window)

            r2_diff_window = r2_score(fc_diff_window.get().flatten(), predfc.get().flatten())
            r2s_diff_window.append(r2_diff_window)

            density = cp.count_nonzero(rules) / (rules.shape[0] ** 2)
            sc_densities.append(density.get())

        axs_both_windows[plot_indx].scatter(sc_densities, r2s_same_window, color="blue")
        axs_both_windows[plot_indx].scatter(sc_densities, r2s_diff_window, color="red")
        axs_both_windows[plot_indx].set_title(f"R2 - Window Size: {window_size}/ 652")
        axs_both_windows[plot_indx].set_xlabel("Rules Density")

        axs_same_window[plot_indx].scatter(sc_densities, r2s_same_window, color="blue")
        axs_same_window[plot_indx].set_title(f"R2 - Window Size: {window_size}/ 652")
        axs_same_window[plot_indx].set_xlabel("Rules Density")

        axs_diff_window[plot_indx].scatter(sc_densities, r2s_diff_window, color="red")
        axs_diff_window[plot_indx].set_title(f"R2 - Window Size: {window_size}/ 652")
        axs_diff_window[plot_indx].set_xlabel("Rules Density")

    fig1.savefig(f"r2_density_both_windows_sub-{subject_id}.png")
    fig2.savefig(f"r2_density_same_window_sub-{subject_id}.png")
    fig3.savefig(f"r2_density_diff_window_sub-{subject_id}.png")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

SUBJECT_IDs = ['032304', '032302', '032307']
for subject_id in SUBJECT_IDs:
    make_validation_plots(subject_id)
