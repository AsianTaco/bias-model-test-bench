import Pk_library as PKL

import numpy as np
import matplotlib.pyplot as plt

from bias_bench.io import BiasModelData
from bias_bench.Params import BiasParams


def compute_power_spectrum(field, l_box, MAS):
    # FIXME: Enable use of double precision
    overdensity = np.array(field, dtype=np.float32)

    pk = PKL.Pk(overdensity, l_box, axis=0, MAS=MAS)

    return pk.k3D, pk.Pk[:, 0]


def plot_power_spectrum(bias_model_data: BiasModelData, plotting_params, benchmark_model_name):
    l_box = bias_model_data.info['BoxSize']
    show_density = plotting_params['show_density']
    MAS = plotting_params['MAS']

    if show_density:
        overdensity_field = bias_model_data.overdensity_field
        k_density, power_density = compute_power_spectrum(overdensity_field, l_box, MAS=MAS)
        plt.loglog(k_density, power_density, label="density")

    try:
        count_field = bias_model_data.count_field
        k_counts, power_counts = compute_power_spectrum(count_field, l_box, MAS=MAS)
        plt.loglog(k_counts, power_counts, label='predicted')
    except AttributeError:
        print("No predicted count field found in BiasModelData. Skipping plots")

    try:
        ground_truth = bias_model_data.count_field_truth
        k_truth, power_truth = compute_power_spectrum(ground_truth, l_box, MAS=MAS)
        plt.loglog(k_truth, power_truth, label='ground truth')
    except AttributeError:
        print("No ground truth count field found in BiasModelData. Skipping plots")

    try:
        benchmark = bias_model_data.count_field_benchmark
        k_benchmark, power_benchmark = compute_power_spectrum(benchmark, l_box, MAS=MAS)
        plt.loglog(k_benchmark, power_benchmark, label=benchmark_model_name)
    except AttributeError:
        print("No benchmark count field found in BiasModelData. Skipping plots")

    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
    plt.tight_layout(pad=0.1)
    plt.legend()
    plt.show()
