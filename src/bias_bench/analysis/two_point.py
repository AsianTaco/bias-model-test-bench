import Pk_library as PKL

import numpy as np
import matplotlib.pyplot as plt

from bias_bench.constants import *


def compute_power_spectrum(field, l_box, MAS):
    # FIXME: Enable use of double precision
    overdensity = np.array(field, dtype=np.float32)

    pk = PKL.Pk(overdensity, l_box, axis=0, MAS=MAS)

    return pk.k3D, pk.Pk[:, 0]


def plot_power_spectrum(bias_model_list, params, dir_path):
    for bias_model_index, bias_model_data in enumerate(bias_model_list):
        bias_model_name = params[f'bias_model_{bias_model_index + 1}']['name']
        benchmark_model_name = params[f'bias_model_{bias_model_index + 1}']['count_field_benchmark_name']

        for res_i in range(bias_model_data.n_res):
            for mass_bin_i in range(bias_model_data.n_mass_bins):

                fig, ax = plt.subplots()
                fig_ratio, ax_ratio = plt.subplots()

                ground_truth_field_exists = False
                show_density = params['power_spectrum']['show_density']
                MAS = params['power_spectrum']['MAS']
                l_box = bias_model_data.info[f'{res_base_name}_{res_i}'][f'{box_size_attr}']

                if show_density:
                    overdensity_field = bias_model_data.overdensity_field
                    k_density, power_density = compute_power_spectrum(overdensity_field, l_box, MAS=MAS)
                    ax.loglog(k_density, power_density, label="density")

                for sim_i in range(bias_model_data.n_simulations):

                    try:
                        count_field_truth = bias_model_data.count_fields_truth[sim_i][res_i][mass_bin_i]
                        count_overdensity_truth = count_field_truth / np.mean(count_field_truth) - 1
                        k_truth, power_truth = compute_power_spectrum(count_overdensity_truth, l_box, MAS=MAS)
                        ax.loglog(k_truth, power_truth, label='ground truth')
                        ground_truth_field_exists = True
                    except IndexError:
                        print("No ground truth count field found in BiasModelData. Skipping plots")

                    try:
                        count_field = bias_model_data.count_fields_predicted[sim_i][res_i][mass_bin_i]
                        count_overdensity = count_field / np.mean(count_field) - 1
                        k_counts, power_counts = compute_power_spectrum(count_overdensity, l_box, MAS=MAS)
                        ax.loglog(k_counts, power_counts, label='predicted')

                        if ground_truth_field_exists:
                            ax_ratio.loglog(k_truth, power_counts / power_truth, label=f'predicted ({bias_model_name})')
                    except IndexError:
                        print("No predicted count field found in BiasModelData. Skipping plots")

                    try:
                        count_field_benchmark = bias_model_data.counts_field_benchmark[sim_i][res_i][mass_bin_i]
                        count_overdensity_benchmark = count_field_benchmark / np.mean(count_field_benchmark) - 1
                        k_benchmark, power_benchmark = compute_power_spectrum(count_overdensity_benchmark, l_box,
                                                                              MAS=MAS)
                        ax.loglog(k_benchmark, power_benchmark, label=benchmark_model_name)

                        if ground_truth_field_exists:
                            ax_ratio.loglog(k_truth, power_benchmark / power_truth,
                                            label=f'{benchmark_model_name} ({bias_model_name})')
                    except IndexError:
                        print("No benchmark count field found in BiasModelData. Skipping plots")

                ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
                ax.set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
                ax.legend()
                fig.suptitle(bias_model_name)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(f"{dir_path}/two_point_{bias_model_name}_res_{res_i}_mass_{mass_bin_i}.png")

                if ground_truth_field_exists:
                    ax_ratio.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
                    ax_ratio.set_ylabel(r"$P(k) / P_{truth}(k)$ ")
                    ax_ratio.axhline(1, linewidth=.5, linestyle='--', color='black')
                    ax_ratio.set_ylim(1e-1, 1e1)
                    ax_ratio.legend()
                    fig_ratio.tight_layout(pad=.1)
                    fig_ratio.savefig(f"{dir_path}/two_point_ratios_res_{res_i}_mass_{mass_bin_i}.png")
