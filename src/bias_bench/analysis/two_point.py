import Pk_library as PKL

import numpy as np
import matplotlib.pyplot as plt


def compute_power_spectrum(field, l_box, MAS):
    # FIXME: Enable use of double precision
    overdensity = np.array(field, dtype=np.float32)

    pk = PKL.Pk(overdensity, l_box, axis=0, MAS=MAS)

    return pk.k3D, pk.Pk[:, 0]


def plot_power_spectrum(bias_model_list, params):
    fig_ratio, ax_ratio = plt.subplots()

    show_density = params['power_spectrum']['show_density']
    MAS = params['power_spectrum']['MAS']

    for bias_model_index, bias_model_data in enumerate(bias_model_list):
        fig, ax = plt.subplots()

        l_box = bias_model_data.info['BoxSize']
        bias_model_name = params[f'bias_model_{bias_model_index + 1}']['name']
        benchmark_model_name = params[f'bias_model_{bias_model_index + 1}']['count_field_benchmark_name']

        ground_truth_field_exists = hasattr(bias_model_data, 'count_field_truth')

        if show_density:
            overdensity_field = bias_model_data.overdensity_field
            k_density, power_density = compute_power_spectrum(overdensity_field, l_box, MAS=MAS)
            ax.loglog(k_density, power_density, label="density")

        try:
            count_field_truth = bias_model_data.count_field_truth
            count_overdensity_truth = count_field_truth / np.mean(count_field_truth) - 1
            k_truth, power_truth = compute_power_spectrum(count_overdensity_truth, l_box, MAS=MAS)
            ax.loglog(k_truth, power_truth, label='ground truth')
        except AttributeError:
            print("No ground truth count field found in BiasModelData. Skipping plots")

        try:
            count_field = bias_model_data.count_field
            count_overdensity = count_field / np.mean(count_field) - 1
            k_counts, power_counts = compute_power_spectrum(count_overdensity, l_box, MAS=MAS)
            ax.loglog(k_counts, power_counts, label='predicted')

            if ground_truth_field_exists:
                ax_ratio.loglog(k_truth, power_counts / power_truth, label=f'predicted ({bias_model_name})')
        except AttributeError:
            print("No predicted count field found in BiasModelData. Skipping plots")

        try:
            count_field_benchmark = bias_model_data.count_field_benchmark
            count_overdensity_benchmark = count_field_benchmark / np.mean(count_field_benchmark) - 1
            k_benchmark, power_benchmark = compute_power_spectrum(count_overdensity_benchmark, l_box, MAS=MAS)
            ax.loglog(k_benchmark, power_benchmark, label=benchmark_model_name)

            if ground_truth_field_exists:
                ax_ratio.loglog(k_truth, power_benchmark / power_truth,
                                label=f'{benchmark_model_name} ({bias_model_name})')
        except AttributeError:
            print("No benchmark count field found in BiasModelData. Skipping plots")

        ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
        ax.set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
        ax.legend()
        fig.suptitle(bias_model_name)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # TODO: Correct naming
        fig.savefig(f"{params['out_dir']}/plots/two_point_{bias_model_name}.png")

    ax_ratio.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    ax_ratio.set_ylabel(r"$P(k) / P_{truth}(k)$ ")
    ax_ratio.axhline(1, linewidth=.5, linestyle='--', color='black')
    ax_ratio.set_ylim(1e-1, 1e1)
    ax_ratio.legend()
    fig_ratio.tight_layout(pad=.1)
    # TODO: Correct naming
    fig_ratio.savefig(f"{params['out_dir']}/plots/two_point_ratios.png")
