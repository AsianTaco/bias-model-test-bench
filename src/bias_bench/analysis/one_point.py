import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from typing import Sequence

from bias_bench.constants import *
from bias_bench.data_io import BiasModelData


def _compute_mean_and_variance(overdensity_field, count_field):
    # Bins of overdensity.
    bins = 10 ** np.arange(-3, 3, 0.1)

    mean, var, bin_c = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = np.where((overdensity_field >= lo) & (overdensity_field < hi))
        mean.append(np.mean(count_field[mask]))
        var.append(np.var(count_field[mask]))
        bin_c.append((lo + hi) / 2.)

    return bin_c, mean, var


def plot_one_point_stats(bias_model_list: Sequence[BiasModelData], params, dir_path):
    for bias_model_index, bias_model_data in enumerate(bias_model_list):
        bias_model_name = params[f'bias_model_{bias_model_index + 1}']['name']
        benchmark_model_name = params[f'bias_model_{bias_model_index + 1}']['count_field_benchmark_name']

        for res_i in range(bias_model_data.n_res):
            for mass_bin_i in range(bias_model_data.n_mass_bins):
                fig, axs = plt.subplots(2, figsize=(5, 9))
                for sim_i in range(bias_model_data.n_simulations):

                    dm_overdensity_flat = bias_model_data.dm_overdensity_fields[sim_i][res_i].flatten()
                    log_dm_overdensity_flat = np.log10(
                        dm_overdensity_flat + 1
                    )

                    try:
                        count_field = bias_model_data.count_fields_predicted[sim_i][res_i][mass_bin_i]
                        log_counts_overdensity = np.log10(count_field / count_field.mean())
                        axs[0].scatter(log_dm_overdensity_flat, log_counts_overdensity.flatten(),
                                       c='tab:blue',
                                       s=1,
                                       alpha=0.5)
                        bins, mean, var = _compute_mean_and_variance(dm_overdensity_flat, count_field.flatten())
                        axs[1].plot(bins, mean, c='tab:blue')
                        axs[1].plot(bins, var, c='tab:orange')
                    except IndexError:
                        print("No predicted count field found in BiasModelData. Skipping plots")

                    try:
                        ground_truth = bias_model_data.count_fields_truth[sim_i][res_i][mass_bin_i]
                        log_ground_truth_overdensity = np.log10(ground_truth / ground_truth.mean())
                        axs[0].scatter(log_dm_overdensity_flat, log_ground_truth_overdensity.flatten(),
                                       label='ground truth',
                                       c='tab:orange',
                                       s=1,
                                       alpha=0.5)
                    except IndexError:
                        print("No ground truth count field found in BiasModelData. Skipping plots")

                    try:
                        benchmark = bias_model_data.counts_field_benchmark[sim_i][res_i][mass_bin_i]
                        log_benchmark_overdensity = np.log10(benchmark / benchmark.mean())
                        axs[0].scatter(log_dm_overdensity_flat, log_benchmark_overdensity.flatten(),
                                       label=benchmark_model_name,
                                       c='tab:purple',
                                       s=1,
                                       alpha=0.5)
                        bins, mean, var = _compute_mean_and_variance(dm_overdensity_flat, benchmark.flatten())
                        axs[1].plot(bins, mean, c='tab:red')
                        axs[1].plot(bins, var, c='tab:purple')
                    except IndexError:
                        print("No benchmark count field found in BiasModelData. Skipping plots")

                # Finalize figure.
                axs[0].set_xlabel(r"$\log_{10}(1 + \delta_{m})$")
                axs[0].set_ylabel(r"$\log_{10}(1 + \delta_{h})$")
                axs[0].set_ylim(bottom=-3)

                legend_elements_0 = [
                    Line2D([0], [0], marker='o', color='w', label='predicted',
                           markerfacecolor='tab:blue',
                           markersize=5),
                    Line2D([0], [0], marker='o', color='w', label='ground truth',
                           markerfacecolor='tab:orange',
                           markersize=5),
                    Line2D([0], [0], marker='o', color='w', label='benchmark',
                           markerfacecolor='tab:purple',
                           markersize=5),
                ]
                axs[0].legend(handles=legend_elements_0, loc='upper center', bbox_to_anchor=(0.5, -0.3),
                              fancybox=True, shadow=True, ncol=3)

                axs[1].set_xscale('log')
                axs[1].set_xlabel(r"$\delta$")
                legend_elements_1 = [
                    Line2D([0], [0], marker='o', color='w', label='mean (predicted)',
                           markerfacecolor='tab:blue',
                           markersize=5),
                    Line2D([0], [0], marker='o', color='w', label='var (predicted)',
                           markerfacecolor='tab:orange',
                           markersize=5),
                    Line2D([0], [0], marker='o', color='w', label='mean (benchmark)',
                           markerfacecolor='tab:red',
                           markersize=5),
                    Line2D([0], [0], marker='o', color='w', label='var (benchmark)',
                           markerfacecolor='tab:purple',
                           markersize=5),
                ]
                axs[1].legend(handles=legend_elements_1, loc='upper center', bbox_to_anchor=(0.5, -0.3),
                              fancybox=True, shadow=True, ncol=2)

                field_attrs = bias_model_data.info[f'{res_base_name}_{res_i}']
                box = field_attrs[box_size_attr]
                ngrid = field_attrs[n_grid_attr]
                # FIXME: remove the hard-coded mass_bin string
                mass_lo_hi = field_attrs[f'mass_bin_{mass_bin_i}']
                resolution = box / ngrid
                fig.suptitle(
                    f'{bias_model_name} for \n'
                    f'{resolution} $h^{{-1}}\mathrm{{Mpc}}^3$ and mass bins {mass_lo_hi}')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(f"{dir_path}/one_point_{bias_model_name}.png")
