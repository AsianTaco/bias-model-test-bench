from typing import Sequence
import Pk_library as PKL

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bias_bench.data_io import BiasModelData
from bias_bench.constants import *
from bias_bench.utils import setup_plotting_folders


def compute_power_spectrum(field, l_box, MAS):
    # FIXME: Enable use of double precision
    overdensity = np.array(field, dtype=np.float32)

    pk = PKL.Pk(overdensity, l_box, axis=0, MAS=MAS)

    return pk.k3D, pk.Pk[:, 0]


line_styles = ['-', '--', ':', '-.']
ratio_colors = ['k', 'r', 'b', 'g']


def plot_power_spectrum(bias_model_list: Sequence[BiasModelData], params, parent_folder_path):
    show_density = params['power_spectrum']['show_density']
    MAS = params['power_spectrum']['MAS']

    for bias_model_index, bias_model_data in enumerate(bias_model_list):
        bias_model_name = params[f'bias_model_{bias_model_index + 1}']['name']
        benchmark_model_name = params[f'bias_model_{bias_model_index + 1}']['count_field_benchmark_name']
        dir_path = f"{parent_folder_path}/{bias_model_name}/two_point"
        setup_plotting_folders(dir_path, bias_model_data.n_simulations)

        for sim_i in range(bias_model_data.n_simulations):
            for res_i in range(bias_model_data.n_res):
                field_attrs = bias_model_data.info[f'{res_base_name}_{res_i}']
                box = field_attrs[box_size_attr]
                ngrid = field_attrs[n_grid_attr]
                resolution = box / ngrid

                nyquist_freq = np.pi / resolution
                k_min = 2 * np.pi / box

                for mass_bin_i in range(bias_model_data.n_mass_bins):
                    fig, ax = plt.subplots()
                    fig_ratio, ax_ratio = plt.subplots()

                    legend_elements = []
                    legend_elements_ratio = []

                    ground_truth_field_exists = False

                    if show_density:
                        overdensity_field = bias_model_data.dm_overdensity_fields[sim_i][res_i][mass_bin_i]
                        k_density, power_density = compute_power_spectrum(overdensity_field, box, MAS=MAS)
                        ax.loglog(k_density, power_density, label="density")

                    try:
                        count_field_truth = bias_model_data.count_fields_truth[sim_i][res_i][mass_bin_i]
                        count_overdensity_truth = count_field_truth / np.mean(count_field_truth) - 1
                        k_truth, power_truth = compute_power_spectrum(count_overdensity_truth, box, MAS=MAS)
                        ax.loglog(k_truth, power_truth, c='k', lw=1, linestyle=line_styles[0])
                        ground_truth_field_exists = True
                        legend_elements.append(Line2D([0], [0], color='k', lw=1, linestyle=line_styles[0],
                                                      label=f'ground truth'))
                    except IndexError:
                        print("No ground truth count field found in BiasModelData. Skipping plots")

                    try:
                        count_field = bias_model_data.count_fields_predicted[sim_i][res_i][mass_bin_i]

                        power = []

                        for sample_i in range(count_field.shape[0]):
                            sample = count_field[sample_i]
                            count_overdensity = sample / np.mean(sample) - 1
                            k_counts, power_counts = compute_power_spectrum(count_overdensity, box, MAS=MAS)
                            ax.loglog(k_counts, power_counts, c='grey', lw=1, linestyle='-', alpha=0.4)
                            power.append(power_counts)

                        power = np.array(power)
                        mean_power = power.mean(axis=0)
                        std_power = power.std(axis=0)

                        ax.loglog(k_counts, mean_power, c='r', lw=1, linestyle=line_styles[1])
                        ax.fill_between(k_counts, (mean_power - std_power), (mean_power + std_power), alpha=0.2,
                                        color='r')

                        legend_elements.append(Line2D([0], [0], color='r', lw=1, linestyle=line_styles[1],
                                                      label=f'prediction (mean)'))
                        prediction_field_exists = True

                        if ground_truth_field_exists:
                            ax_ratio.semilogx(k_truth, mean_power / power_truth - 1, c='r', lw=1,
                                              linestyle=line_styles[0])
                            legend_elements_ratio.append(
                                Line2D([0], [0], color='r', lw=1, linestyle=line_styles[0],
                                       label=f'prediction'))
                    except IndexError:
                        print("No predicted count field found in BiasModelData. Skipping plots")

                    try:
                        count_field_benchmark = bias_model_data.counts_field_benchmark[sim_i][res_i][mass_bin_i]
                        count_overdensity_benchmark = count_field_benchmark / count_field_benchmark.mean() - 1
                        k_benchmark, power_benchmark = compute_power_spectrum(count_overdensity_benchmark, box, MAS=MAS)
                        ax.loglog(k_benchmark, power_benchmark, color='b', lw=1, linestyle=line_styles[2])
                        legend_elements.append(Line2D([0], [0], color='b', lw=1, linestyle=line_styles[2],
                                                      label=f'benchmark ({benchmark_model_name})'))

                        if ground_truth_field_exists:
                            ax_ratio.semilogx(k_truth, power_benchmark / power_truth - 1, c='b', lw=1,
                                              linestyle=line_styles[1])
                            legend_elements_ratio.append(
                                Line2D([0], [0], color='b', lw=1, linestyle=line_styles[1],
                                       label=f'benchmark ({benchmark_model_name})'))
                    except (IndexError, AttributeError, TypeError):
                        print("No benchmark count field found in BiasModelData. Skipping plots")

                    if prediction_field_exists:
                        legend_elements.append(Patch(color='r', label='prediction (std)', alpha=0.2))

                    ax.legend(handles=legend_elements, fancybox=True, shadow=True)
                    ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
                    ax.set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
                    ax.set_xlim(k_min, nyquist_freq)

                    # FIXME: remove the hard-coded mass_bin key string
                    mass_lo_hi = [f'{n:.2e}' for n in field_attrs[f'mass_bin_{mass_bin_i}']]
                    fig.suptitle(f'Power spectrum comparison at ${resolution}$ Mpc/h voxel resolution\n'
                                 f'for halo masses between ${mass_lo_hi[0]} M_\\odot$ and ${mass_lo_hi[1]} M_\\odot$')
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig.savefig(f"{dir_path}/sim_{sim_i}/res_{res_i}_mass_{mass_bin_i}.png")

                    if ground_truth_field_exists:
                        ax_ratio.axhline(0, linewidth=.5, linestyle='--', color='black')
                        ax_ratio.axhline(-0.05, lw=.5, color='grey', alpha=.5)
                        ax_ratio.axhline(0.05, lw=.5, color='grey', alpha=.5)
                        ax_ratio.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
                        ax_ratio.set_ylabel(r"$P(k) / P_{truth}(k) - 1$ ")
                        ax_ratio.set_ylim(-0.5, 0.5)
                        ax_ratio.set_xlim(k_min, nyquist_freq)
                        ax_ratio.legend(handles=legend_elements_ratio, fancybox=True, shadow=True)
                        fig_ratio.suptitle(f'Power spectrum ratios at ${resolution}$ Mpc/h \n'
                                           f'for halo masses between ${mass_lo_hi[0]} M_\\odot$ and ${mass_lo_hi[1]} M_\\odot$')
                        fig_ratio.tight_layout(rect=[0, 0.03, 1, 0.95])
                        fig_ratio.savefig(f"{dir_path}/sim_{sim_i}/ratios_res_{res_i}_mass_{mass_bin_i}.png")

                    plt.close(fig)
                    plt.close(fig_ratio)
