from typing import Sequence
import Pk_library as PKL

import numpy as np
import matplotlib.pyplot as plt

from src.bias_bench.data_io import BiasModelData


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


def plot_bispectrum(bias_model_list: Sequence[BiasModelData], params):
    # TODO: Find out what k1, k2 values would be sensible so set as default
    show_density = params['bi_spectrum']['show_density']
    k1 = params['bi_spectrum']['k1']
    k2 = params['bi_spectrum']['k2']
    Ntheta = params['bi_spectrum']['Ntheta']
    MAS = params['bi_spectrum']['MAS']

    for bias_model_index, bias_model_data in enumerate(bias_model_list):
        fig, ax = plt.subplots()

        l_box = bias_model_data.info['BoxSize']
        bias_model_name = params[f'bias_model_{bias_model_index + 1}']['name']
        benchmark_model_name = params[f'bias_model_{bias_model_index + 1}']['count_field_benchmark_name']

        try:
            count_field = bias_model_data.count_field
            k_counts, bispec_counts = compute_bispectrum(count_field, l_box, k1, k2, Ntheta, MAS=MAS)
            ax.loglog(k_counts, bispec_counts['bispectrum'], label='predicted')
        except AttributeError:
            print("No predicted count field found in BiasModelData. Skipping plots")

        try:
            ground_truth = bias_model_data.count_field_truth
            k_truth, bispec_truth = compute_bispectrum(ground_truth, l_box, k1, k2, Ntheta, MAS=MAS)
            ax.loglog(k_truth, bispec_truth['bispectrum'], label='ground truth')
        except AttributeError:
            print("No ground truth count field found in BiasModelData. Skipping plots")

        try:
            benchmark = bias_model_data.count_field_benchmark
            k_benchmark, bispec_benchmark = compute_bispectrum(benchmark, l_box, k1, k2, Ntheta, MAS=MAS)
            ax.loglog(k_benchmark, bispec_benchmark['bispectrum'], label=benchmark_model_name)
        except AttributeError:
            print("No benchmark count field found in BiasModelData. Skipping plots")

        if show_density:
            overdensity_field = bias_model_data.overdensity_field
            k_density, bispect_density = compute_bispectrum(overdensity_field, l_box, k1, k2, Ntheta, MAS=MAS)
            ax.loglog(k_density, bispect_density['bispectrum'], label="density")

        ax.set_xlabel(r"Angle $\theta$")
        ax.set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
        ax.legend()
        fig.suptitle(bias_model_name)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(f"{params['out_dir']}/plots/three_point_{bias_model_name}.png")
