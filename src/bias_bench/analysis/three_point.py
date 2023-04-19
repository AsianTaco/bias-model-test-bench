from typing import Sequence
import Pk_library as PKL

import numpy as np
import matplotlib.pyplot as plt

from bias_bench.data_io import BiasModelData
from bias_bench.constants import *


def compute_bispectrum(field, Lbox, k1, k2, Ntheta, MAS):
    # FIXME: Enable use of double precision for the field
    field = field.astype(np.float32)
    theta = np.linspace(0, np.pi, Ntheta)

    bbk = PKL.Bk(field, Lbox, k1=k1, k2=k2, theta=theta, MAS=MAS)

    # FIXME: combine this with power spectrum computation
    # Power spectrum for free
    # k = bbk.k  # k-bins for power spectrum
    # Pk = bbk.Pk # power spectrum

    return theta, {'bispectrum': bbk.B, 'reduced_bispectrum': bbk.Q}


def plot_bispectrum(bias_model_list: Sequence[BiasModelData], params, dir_path):
    show_density = params['bi_spectrum']['show_density']
    k1 = params['bi_spectrum']['k1']
    k2 = params['bi_spectrum']['k2']
    Ntheta = params['bi_spectrum']['Ntheta']
    MAS = params['bi_spectrum']['MAS']

    for bias_model_index, bias_model_data in enumerate(bias_model_list):
        bias_model_name = params[f'bias_model_{bias_model_index + 1}']['name']
        benchmark_model_name = params[f'bias_model_{bias_model_index + 1}']['count_field_benchmark_name']

        for res_i in range(bias_model_data.n_res):
            for mass_bin_i in range(bias_model_data.n_mass_bins):

                fig, ax = plt.subplots()
                fig_ratio, ax_ratio = plt.subplots()

                ground_truth_field_exists = False
                l_box = bias_model_data.info[f'{res_base_name}_{res_i}'][f'{box_size_attr}']

                for sim_i in range(bias_model_data.n_simulations):

                    if show_density:
                        overdensity_field = bias_model_data.dm_overdensity_fields[sim_i][res_i][mass_bin_i]
                        k_density, bispect_density = compute_bispectrum(overdensity_field, l_box, k1, k2, Ntheta,
                                                                        MAS=MAS)
                        ax.loglog(k_density, bispect_density['bispectrum'], label="density")

                    try:
                        count_field_truth = bias_model_data.count_fields_truth[sim_i][res_i][mass_bin_i]
                        count_overdensity_truth = count_field_truth / np.mean(count_field_truth) - 1
                        k_truth, bispec_truth = compute_bispectrum(count_overdensity_truth, l_box, k1, k2, Ntheta,
                                                                   MAS=MAS)
                        ax.loglog(k_truth, bispec_truth['bispectrum'], label='ground truth')
                        ground_truth_field_exists = True
                    except IndexError:
                        print("No ground truth count field found in BiasModelData. Skipping plots")

                    try:
                        count_field = bias_model_data.count_fields_predicted[sim_i][res_i][mass_bin_i]
                        count_overdensity = count_field / np.mean(count_field) - 1
                        k_counts, bispec_counts = compute_bispectrum(count_overdensity, l_box, k1, k2, Ntheta, MAS=MAS)
                        ax.loglog(k_counts, bispec_counts['bispectrum'], label='predicted')

                        if ground_truth_field_exists:
                            ax_ratio.loglog(k_truth, bispec_counts['bispectrum'] / bispec_truth['bispectrum'],
                                            label=f'predicted ({bias_model_name})')
                    except IndexError:
                        print("No predicted count field found in BiasModelData. Skipping plots")

                    try:
                        count_field_benchmark = bias_model_data.counts_field_benchmark[sim_i][res_i][mass_bin_i]
                        count_overdensity_benchmark = count_field_benchmark / np.mean(count_field_benchmark) - 1
                        k_benchmark, bispec_benchmark = compute_bispectrum(count_overdensity_benchmark, l_box, k1, k2,
                                                                           Ntheta, MAS=MAS)
                        ax.loglog(k_benchmark, bispec_benchmark['bispectrum'], label=benchmark_model_name)

                        if ground_truth_field_exists:
                            ax_ratio.loglog(k_truth, bispec_benchmark['bispectrum'] / bispec_truth['bispectrum'],
                                            label=f'{benchmark_model_name}')

                    except (IndexError, AttributeError, TypeError):
                        print("No benchmark count field found in BiasModelData. Skipping plots")

                ax.set_xlabel(r"Angle $\theta$")
                ax.set_ylabel(r"$B(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
                ax.legend()
                field_attrs = bias_model_data.info[f'{res_base_name}_{res_i}']
                box = field_attrs[box_size_attr]
                ngrid = field_attrs[n_grid_attr]
                # FIXME: remove the hard-coded mass_bin key string
                mass_lo_hi = [f'{n:.2e}' for n in field_attrs[f'mass_bin_{mass_bin_i}']]
                resolution = box / ngrid
                fig.suptitle(f'{bias_model_name} for\n'
                             f'voxel size {resolution:.2f}$h^{{-1}}\\mathrm{{Mpc}}^3$ and mass bins {mass_lo_hi}')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(f"{dir_path}/{bias_model_name}_res_{res_i}_mass_{mass_bin_i}.png")

                if ground_truth_field_exists:
                    ax_ratio.set_xlabel(r"Angle $\theta$")
                    ax_ratio.set_ylabel(r"$B(k) / B_{truth}(k)$ ")
                    ax_ratio.axhline(1, linewidth=.5, linestyle='--', color='black')
                    ax_ratio.set_ylim(1e-1, 1e1)
                    ax_ratio.legend()
                    fig_ratio.tight_layout(pad=.1)
                    fig_ratio.savefig(f"{dir_path}/ratios_res_{res_i}_mass_{mass_bin_i}.png")
