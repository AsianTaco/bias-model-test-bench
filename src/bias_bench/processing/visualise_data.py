from bias_bench.data_io import BiasModelData
from bias_bench.Params import BiasParams
from bias_bench.likelihoods.selector import select_likelihood
from bias_bench.optimizer.selector import select_optimizer

import numpy as np
import matplotlib.pyplot as plt


def plot_hmf(Ms, halo_masses, box_size, label):
    n_masses, bins = np.histogram(halo_masses, bins=Ms)
    plt.loglog(0.5*(bins[:-1] + bins[1:]), n_masses / box_size ** 3 / np.diff(np.log(bins)), label=label)

    plt.xlabel("$M_\odot$")
    plt.ylabel("$n(M)$")


def fit_likelihood(bias_model_data: BiasModelData, bias_params: BiasParams, which_model=1):
    params = bias_params.data

    benchmark_model_loss = params[f'bias_model_{which_model}']['predict_counts_loss']
    benchmark_optimizer = params[f'bias_model_{which_model}']['benchmark_optimizer']
    benchmark_optimizer_args = params[f'bias_model_{which_model}']['benchmark_optimizer_args']

    likelihood = select_likelihood(benchmark_model_loss)
    mean_predictor = lambda x, p: p
    # optimizer = select_optimizer(benchmark_optimizer, likelihood, benchmark_model, benchmark_optimizer_args)

    for res_i in range(bias_model_data.n_res):
        for mass_bin_i in range(bias_model_data.n_mass_bins):
            counts_fields_truth = []

            for sim_i in range(bias_model_data.n_simulations):

                try:
                    counts_fields_truth.append(bias_model_data.count_fields_truth[sim_i][res_i][mass_bin_i])
                except IndexError:
                    print(f"No truth count field found. Skipping fit with {benchmark_model_loss} likelihood.")
                    return

            counts_fields_truth = np.array(counts_fields_truth)
            hist, edges = np.histogram(counts_fields_truth, np.arange(0, np.max(counts_fields_truth)))
            centralised_bins = 0.5 * (edges[:-1] + edges[1:])
