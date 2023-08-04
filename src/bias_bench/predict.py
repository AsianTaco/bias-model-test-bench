from bias_bench.data_io import BiasModelData
from bias_bench.Params import BiasParams
from bias_bench.benchmark_models.selector import select_bias_model
from bias_bench.likelihoods.selector import select_likelihood
from bias_bench.optimizer.scipy_minimize import BFGSScipy
from bias_bench.optimizer.selector import select_optimizer
from bias_bench.utils import bias_bench_print

import numpy as np


def predict_galaxy_counts(bias_model_data: BiasModelData, bias_params: BiasParams,
                          which_model=1):
    """ Predict ngal from density field. """

    params = bias_params.data

    benchmark_model_name = params[f'bias_model_{which_model}']['predict_counts_model']
    benchmark_model_loss = params[f'bias_model_{which_model}']['predict_counts_loss']
    init_params = np.array(params[f'bias_model_{which_model}']['predict_init_params'])

    likelihood = select_likelihood(benchmark_model_loss)
    benchmark_model = select_bias_model(benchmark_model_name)
    # TODO: Generalize this to different optimizers via a selector function
    optimizer = select_optimizer('emcee', likelihood, benchmark_model)

    counts_field_benchmark = [
        [
            [
                [] for _ in range(bias_model_data.n_mass_bins)
            ] for _ in range(bias_model_data.n_res)
        ] for _ in range(bias_model_data.n_simulations)
    ]

    n_benchmark_fits = 0

    for res_i in range(bias_model_data.n_res):
        for mass_bin_i in range(bias_model_data.n_mass_bins):
            overdensities = []
            counts_fields_truth = []

            for sim_i in range(bias_model_data.n_simulations):
                overdensities.append(bias_model_data.dm_overdensity_fields[sim_i][res_i])

                try:
                    counts_fields_truth.append(bias_model_data.count_fields_truth[sim_i][res_i][mass_bin_i])
                except IndexError:
                    print(f"No truth count field found. Skipping fit with {benchmark_model_name}.")
                    counts_fields_truth = None
                    return

            overdensities = np.array(overdensities)
            counts_fields_truth = np.array(counts_fields_truth)

            # Fit ngal vs delta relation.
            bias_bench_print(f"Fitting {benchmark_model.name} for res_{res_i}, mass_bin_{mass_bin_i}", verbose=True)
            fitted_lh_params, fitted_benchmark_model_params = optimizer.optimize(overdensities, counts_fields_truth,
                                                                                 init_params)

            if (fitted_lh_params is None) and (fitted_benchmark_model_params is None):
                print(f"Was not able to fit model for res_{res_i}, mass_bin_{mass_bin_i}")
            else:
                predicted_count_fields = benchmark_model.predict(overdensities, fitted_benchmark_model_params)
                sampled_count_fields = likelihood.sample(predicted_count_fields, fitted_lh_params)
                n_benchmark_fits += 1

                for i, prediction in enumerate(sampled_count_fields):
                    counts_field_benchmark[i][res_i][mass_bin_i] = prediction

    print(f'Successfully completed {n_benchmark_fits} benchmark model fits.')
    return counts_field_benchmark
    # TODO: Add option to save predicted benchmark models to the corresponding hdf5 file
