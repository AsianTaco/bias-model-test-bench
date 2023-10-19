from typing import Sequence
from collections import namedtuple
import Pk_library as PKL

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bias_bench.constants import *
from bias_bench.data_io import BiasModelData
from bias_bench.utils import setup_plotting_folders

PowerSpectrum = namedtuple("PowerSpectrum", "k power")
BiSpectrum = namedtuple("BiSpectrum", "theta bispectrum reduced_bispectrum")


def compute_overdensity(field):
    return field / np.mean(field) - 1


def compute_summaries(field, Lbox, k1, k2, Ntheta, MAS):
    field = field.astype(np.float32)
    theta = np.linspace(0, np.pi, Ntheta)

    k, pk = compute_power_spectrum(field, Lbox, MAS)
    bbk = PKL.Bk(field, Lbox, k1=k1, k2=k2, theta=theta, MAS=MAS)

    power_spectrum = PowerSpectrum(k, pk)
    bispectrum = BiSpectrum(theta, bbk.B, bbk.Q)

    return power_spectrum, bispectrum


def compute_power_spectrum(field, l_box, MAS):
    overdensity = np.array(field, dtype=np.float32)

    pk = PKL.Pk(overdensity, l_box, axis=0, MAS=MAS)

    return pk.k3D, pk.Pk[:, 0]


def compute_cross_correlation_coefficient(pred, truth, lbox, MAS):
    pk = PKL.XPk([np.array(pred, dtype=np.float32), np.array(truth, dtype=np.float32)], lbox, axis=0, MAS=MAS)

    pk_pred = pk.Pk[:, 0, 0]
    pk_truth = pk.Pk[:, 0, 1]

    xpk = pk.XPk[:, 0, 0]

    r_c = xpk / np.sqrt(pk_pred * pk_truth)

    return pk.k3D, r_c


line_styles = ['-', '--', ':', '-.']
ratio_colors = ['k', 'r', 'b', 'g']

ground_truth_legend = Line2D([0], [0], color=ratio_colors[0], ls=line_styles[0], label=f'ground truth')
prediction_mean_legend = Line2D([0], [0], color=ratio_colors[1], ls=line_styles[1], label=f'prediction (mean)')


def add_plots_with_std(ax, x, mean, std, ls=line_styles[1]):
    ax.loglog(x, mean, c='r', lw=1, linestyle=ls)
    ax.fill_between(x, (mean - std), (mean + std), alpha=0.4, color='r')
    ax.fill_between(x, (mean - 2 * std), (mean + 2 * std), alpha=0.2, color='r')


def finalise_figure(fig, ax, xlabel, ylabel, legends, xlim, ylim, mass_bins):
    ax.legend(handles=legends)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    m_lo = mass_bins[0]
    m_hi = mass_bins[1]
    m_lo_str = m_lo[0] + r"\cdot 10^{" + m_lo[1] + '}'

    try:
        m_hi_str = '<' + m_hi[0] + r"\cdot 10^{" + m_hi[1] + '}'
    except IndexError:
        m_hi_str = ""

    fig.suptitle('$' + m_lo_str + r"\leq M_{\mathrm{vir}} [M_\odot]" + m_hi_str + '$')


def plot_power_and_bi_spectrum(bias_model_list: Sequence[BiasModelData], params, parent_folder_path):
    # power-spectrum parameters
    show_density = params['power_spectrum']['show_density']
    MAS = params['power_spectrum']['MAS']

    # bi-spectrum parameters
    k1 = params['bi_spectrum']['k1']
    k2 = params['bi_spectrum']['k2']
    Ntheta = params['bi_spectrum']['Ntheta']

    for bias_model_index, bias_model_data in enumerate(bias_model_list):
        bias_model_name = params[f'bias_model_{bias_model_index + 1}']['name']
        benchmark_model_name = params[f'bias_model_{bias_model_index + 1}']['count_field_benchmark_name']

        two_point_dir = f"{parent_folder_path}/{bias_model_name}/two_point"
        setup_plotting_folders(two_point_dir, bias_model_data.n_simulations)

        three_point_dir = f"{parent_folder_path}/{bias_model_name}/three_point"
        setup_plotting_folders(three_point_dir, bias_model_data.n_simulations)

        for sim_i in range(bias_model_data.n_simulations):
            for res_i in range(bias_model_data.n_res):
                field_attrs = bias_model_data.info[f'{res_base_name}_{res_i}']
                box = field_attrs[box_size_attr]
                ngrid = field_attrs[n_grid_attr]
                resolution = box / ngrid

                nyquist_freq = np.pi / resolution
                k_min = 2 * np.pi / box

                for mass_bin_i in range(bias_model_data.n_mass_bins):
                    power_spec_fig, power_spec_ax = plt.subplots()
                    cross_correlation_fig, cross_correlation_ax = plt.subplots()
                    power_spec_ratio_fig, power_spec_ratio_ax = plt.subplots()

                    legend_elements_power_spec = []
                    legend_elements_power_spec_ratio = []

                    bi_spec_fig, bi_spec_ax = plt.subplots()
                    bi_spec_ratio_fig, bi_spec_ratio_ax = plt.subplots()
                    reduced_bi_spec_fig, reduced_bi_spec_ax = plt.subplots()

                    legend_elements_bi_spec = []
                    legend_elements_ratio_bi_spec = []

                    ground_truth_field_exists = False

                    if show_density:
                        overdensity_field = bias_model_data.dm_overdensity_fields[sim_i][res_i][mass_bin_i]
                        power_spec_density, bi_spec_density = compute_summaries(overdensity_field, box, k1, k2, Ntheta,
                                                                                MAS)
                        power_spec_ax.loglog(power_spec_density.k, power_spec_density.power, label="$\delta_m$")
                        bi_spec_ax.loglog(bi_spec_density.theta, bi_spec_density.bispectrum, label="$\delta_m$")

                    try:
                        count_field_truth = bias_model_data.count_fields_truth[sim_i][res_i][mass_bin_i]
                        count_overdensity_truth = compute_overdensity(count_field_truth)
                        power_spec_truth, bi_spec_truth = compute_summaries(count_overdensity_truth, box, k1, k2,
                                                                            Ntheta, MAS)

                        power_spec_ax.loglog(power_spec_truth.k, power_spec_truth.power, c='k', ls=line_styles[0])
                        legend_elements_power_spec.append(ground_truth_legend)

                        bi_spec_ax.loglog(bi_spec_truth.theta, bi_spec_truth.bispectrum, c='k', ls=line_styles[0])
                        legend_elements_bi_spec.append(ground_truth_legend)

                        ground_truth_field_exists = True

                    except IndexError:
                        print("No ground truth count field found in BiasModelData. Skipping plots")

                    try:
                        count_field = bias_model_data.count_fields_predicted[sim_i][res_i][mass_bin_i]
                        n_samples = count_field.shape[0]

                        power = []
                        bi_spec = []

                        for sample_i in range(n_samples):
                            sample = count_field[sample_i]
                            count_overdensity = sample / np.mean(sample) - 1
                            power_spec_counts, bi_spec_counts = compute_summaries(count_overdensity, box, k1, k2,
                                                                                  Ntheta, MAS)
                            power.append(power_spec_counts.power)
                            bi_spec.append(bi_spec_counts.bispectrum)
                            # power_spec_ax.loglog(power_spec_counts.k, power_spec_counts.power, c='grey',
                            #                      ls='-', alpha=0.4)
                            # bi_spec_ax.loglog(bi_spec_counts.theta, bi_spec_counts.bispectrum, c='grey',
                            #                   ls='-', alpha=0.4)

                        power = np.array(power)
                        mean_power = power.mean(axis=0)
                        std_power = power.std(axis=0)

                        add_plots_with_std(power_spec_ax, power_spec_counts.k, mean_power, std_power)
                        legend_elements_power_spec.append(prediction_mean_legend)

                        bi_spec = np.array(bi_spec)
                        mean_bi_spec = bi_spec.mean(axis=0)
                        std_bi_spec = bi_spec.std(axis=0)

                        add_plots_with_std(bi_spec_ax, bi_spec_counts.theta, mean_bi_spec, std_bi_spec)
                        legend_elements_bi_spec.append(prediction_mean_legend)

                        if ground_truth_field_exists:
                            mean_ratio = (mean_power / power_spec_truth.power) - 1
                            power_ratios = []
                            for sample_i in range(count_field.shape[0]):
                                # power_spec_ratio_ax.semilogx(power_spec_truth.k,
                                #                              (power[sample_i] / power_spec_truth.power) - 1, c='grey',
                                #                              lw=1,
                                #                              linestyle='-', alpha=0.2)
                                power_ratios.append((power[sample_i] / power_spec_truth.power) - 1)

                            power_ratios = np.array(power_ratios)
                            std_power_ratio = power_ratios.std(axis=0)

                            power_spec_ratio_ax.semilogx(power_spec_truth.k, mean_ratio, c='r', lw=1,
                                                         linestyle=line_styles[0])
                            power_spec_ratio_ax.fill_between(power_spec_truth.k, (mean_ratio - std_power_ratio),
                                                             (mean_ratio + std_power_ratio), alpha=0.4,
                                                             color='r')
                            power_spec_ratio_ax.fill_between(power_spec_truth.k, (mean_ratio - 2 * std_power_ratio),
                                                             (mean_ratio + 2 * std_power_ratio), alpha=0.2,
                                                             color='r')
                            # power_spec_ax.fill_between(k_counts, - std_power/ mean_power,
                            #                 std_power/ mean_power, alpha=0.2,
                            #                 color='r')
                            legend_elements_power_spec_ratio.append(
                                Line2D([0], [0], color='r', lw=1, linestyle=line_styles[0],
                                       label=f'prediction'))

                            cross_power = []
                            for sample_i in range(count_field.shape[0]):
                                sample = count_field[sample_i]
                                count_overdensity = sample / np.mean(sample) - 1
                                k_counts, power_counts = compute_cross_correlation_coefficient(count_overdensity,
                                                                                               count_overdensity_truth,
                                                                                               box,
                                                                                               MAS=['None', 'None'])
                                cross_correlation_ax.semilogx(k_counts, power_counts, c='grey', lw=1, linestyle='-',
                                                              alpha=0.2)
                                cross_power.append(power_counts)

                            cross_power = np.array(cross_power)
                            mean_cross_power = cross_power.mean(axis=0)

                            cross_correlation_ax.semilogx(k_counts, mean_cross_power, c='r', lw=1,
                                                          linestyle=line_styles[1])

                            # Bispectrum ratios
                            mean_bispec_ratio = (mean_bi_spec / bi_spec_truth.bispectrum) - 1

                            bi_spec_ratios = []
                            for sample_i in range(count_field.shape[0]):
                                bi_spec_ratios.append((bi_spec[sample_i] / bi_spec_truth.bispectrum) - 1)
                                # bi_spec_ratio_ax.semilogx(bi_spec_truth.theta,
                                #                           (bi_spec[sample_i] / bi_spec_truth.bispectrum) - 1,
                                #                           c='grey',
                                #                           lw=1,
                                #                           linestyle='-', alpha=0.2)
                            bi_spec_ratios = np.array(bi_spec_ratios)
                            std_bi_spec_ratio = bi_spec_ratios.std(axis=0)

                            bi_spec_ratio_ax.semilogx(bi_spec_truth.theta, mean_bispec_ratio, c='r', lw=1,
                                                      linestyle=line_styles[0])
                            bi_spec_ratio_ax.fill_between(bi_spec_truth.theta, (mean_bispec_ratio - std_bi_spec_ratio),
                                                          (mean_bispec_ratio + std_bi_spec_ratio), alpha=0.4,
                                                          color='r')
                            bi_spec_ratio_ax.fill_between(bi_spec_truth.theta,
                                                          (mean_bispec_ratio - 2 * std_bi_spec_ratio),
                                                          (mean_bispec_ratio + 2 * std_bi_spec_ratio), alpha=0.2,
                                                          color='r')

                            # power_spec_ax.fill_between(k_counts, - std_power/ mean_power,
                            #                 std_power/ mean_power, alpha=0.2,
                            #                 color='r')
                            legend_elements_ratio_bi_spec.append(
                                Line2D([0], [0], color='r', lw=1, linestyle=line_styles[0],
                                       label=f'prediction'))

                    except IndexError:
                        print("No predicted count field found in BiasModelData. Skipping plots")

                    try:
                        count_field_benchmark = bias_model_data.counts_field_benchmark[sim_i][res_i][mass_bin_i]
                        count_overdensity_benchmark = count_field_benchmark / count_field_benchmark.mean() - 1
                        power_spec_benchmark, bi_spec_benchmark = compute_summaries(count_overdensity_benchmark, box,
                                                                                    k1, k2,
                                                                                    Ntheta, MAS)
                        power_spec_ax.loglog(power_spec_benchmark.k, power_spec_benchmark.power, color='b', lw=1,
                                             linestyle=line_styles[2])
                        legend_elements_power_spec.append(Line2D([0], [0], color='b', lw=1, linestyle=line_styles[2],
                                                                 label=f'benchmark ({benchmark_model_name})'))

                        bi_spec_ax.loglog(bi_spec_benchmark.theta, bi_spec_benchmark.bispectrum, color='b', lw=1,
                                          linestyle=line_styles[2])
                        legend_elements_bi_spec.append(Line2D([0], [0], color='b', lw=1, linestyle=line_styles[2],
                                                              label=f'benchmark ({benchmark_model_name})'))

                        if ground_truth_field_exists:
                            power_spec_ratio_ax.semilogx(power_spec_truth.k,
                                                         power_spec_benchmark.power / power_spec_truth.power - 1,
                                                         c='b',
                                                         lw=1,
                                                         linestyle=line_styles[1])
                            legend_elements_power_spec_ratio.append(
                                Line2D([0], [0], color='b', lw=1, linestyle=line_styles[1],
                                       label=f'benchmark ({benchmark_model_name})'))

                            bi_spec_ratio_ax.semilogx(bi_spec_truth.theta,
                                                      bi_spec_benchmark.bispectrum / bi_spec_truth.bispectrum - 1,
                                                      c='b',
                                                      lw=1,
                                                      linestyle=line_styles[1])
                            legend_elements_ratio_bi_spec.append(
                                Line2D([0], [0], color='b', lw=1, linestyle=line_styles[1],
                                       label=f'benchmark ({benchmark_model_name})')
                            )

                    except (IndexError, AttributeError, TypeError):
                        print("No benchmark count field found in BiasModelData. Skipping plots")

                    # FIXME: remove the hard-coded mass_bin key string
                    mass_lo_hi = ['{:.0e}'.format(n).split('e+') for n in field_attrs[f'mass_bin_{mass_bin_i}']]

                    finalise_figure(power_spec_fig, power_spec_ax,
                                    r"$k$ [$h \ \mathrm{Mpc}^{-1}$]",
                                    r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]",
                                    legend_elements_power_spec, (2e-2, nyquist_freq), (1e3, None), mass_lo_hi)

                    power_spec_fig.savefig(f"{two_point_dir}/sim_{sim_i}/res_{res_i}_mass_{mass_bin_i}.png")

                    cross_correlation_ax.legend(handles=legend_elements_power_spec, fancybox=True, shadow=True)
                    cross_correlation_ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
                    cross_correlation_ax.set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
                    cross_correlation_ax.set_xlim(2e-2, nyquist_freq)
                    cross_correlation_ax.set_ylim(bottom=.5)

                    cross_correlation_fig.suptitle(f'Cross correlation'
                                                   f' (${mass_lo_hi[0]} M_\\odot < M_h <{mass_lo_hi[1]} M_\\odot$)')
                    cross_correlation_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    cross_correlation_fig.savefig(
                        f"{two_point_dir}/sim_{sim_i}/cross_res_{res_i}_mass_{mass_bin_i}.png")

                    finalise_figure(bi_spec_fig, bi_spec_ax,
                                    r"Angle $\theta$",
                                    r"$B_{k_1, k_2}(\theta)$ [$h^{-6}\mathrm{Mpc}^6$]",
                                    legend_elements_bi_spec, (None, None), (None, None), mass_lo_hi)

                    scale_info_text = bi_spec_ax.text(0.05, 0.1, '', transform=bi_spec_ax.transAxes)
                    scale_info_text = bi_spec_ax.annotate(
                        f"$k_1 = {k1}"+r"h \ \mathrm{Mpc}^{-1}$", xycoords=scale_info_text, xy=(0, -1.1),
                        verticalalignment="bottom")
                    bi_spec_ax.annotate(
                        f"$k_2 = {k2}" + r"h \ \mathrm{Mpc}^{-1}$", xycoords=scale_info_text, xy=(0, -1.1),
                        verticalalignment="bottom")

                    # bi_spec_fig.suptitle(f'Bi-spectrum comparison for $k_1 = {k1}, k_2 = {k2}$\n'
                    #                      f' (${mass_lo_hi[0]} M_\\odot < M_h <{mass_lo_hi[1]} M_\\odot$)')
                    # bi_spec_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    bi_spec_fig.savefig(f"{three_point_dir}/sim_{sim_i}/res_{res_i}_mass_{mass_bin_i}.png")

                    if ground_truth_field_exists:
                        power_spec_ratio_ax.axhline(0, linewidth=.5, linestyle='--', color='black')
                        # power_spec_ratio_ax.axhline(-0.05, lw=.5, color='grey', alpha=.5)
                        # power_spec_ratio_ax.axhline(0.05, lw=.5, color='grey', alpha=.5)

                        finalise_figure(power_spec_ratio_fig, power_spec_ratio_ax,
                                        r"$k$ [$h \ \mathrm{Mpc}^{-1}$]",
                                        r"$P(k) / P_{\mathrm{ref}}(k) - 1$ ",
                                        legend_elements_power_spec_ratio, (2e-2, nyquist_freq), (-0.2, 0.2), mass_lo_hi)

                        power_spec_ratio_fig.savefig(
                            f"{two_point_dir}/sim_{sim_i}/ratios_res_{res_i}_mass_{mass_bin_i}.png")

                        bi_spec_ratio_ax.axhline(0, linewidth=.5, linestyle='--', color='black')
                        scale_info_text = bi_spec_ratio_ax.text(0.05, 0.1, '', transform=bi_spec_ratio_ax.transAxes)
                        scale_info_text = bi_spec_ratio_ax.annotate(
                            f"$k_1 = {k1}" + r"h \ \mathrm{Mpc}^{-1}$", xycoords=scale_info_text, xy=(0, -1.1),
                            verticalalignment="bottom")
                        bi_spec_ratio_ax.annotate(
                            f"$k_2 = {k2}" + r"h \ \mathrm{Mpc}^{-1}$", xycoords=scale_info_text, xy=(0, -1.1),
                            verticalalignment="bottom")

                        finalise_figure(bi_spec_ratio_fig, bi_spec_ratio_ax,
                                        r"Angle $\theta$",
                                        r"$B(k) / B_{truth}(k) - 1$",
                                        legend_elements_ratio_bi_spec, (None, None), (-0.4, 0.4), mass_lo_hi)

                        bi_spec_ratio_fig.savefig(
                            f"{three_point_dir}/sim_{sim_i}/ratios_res_{res_i}_mass_{mass_bin_i}.png")

                    plt.close(power_spec_fig)
                    plt.close(cross_correlation_fig)
                    plt.close(power_spec_ratio_fig)

                    plt.close(bi_spec_fig)
                    plt.close(bi_spec_ratio_fig)
