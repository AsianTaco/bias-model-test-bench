from bias_bench.io import BiasModelData
from bias_bench.Params import BiasParams
from bias_bench.benchmark_models.TruncatedPowerLaw import TruncatedPowerLaw


def predict_galaxy_counts(bias_model_data: BiasModelData, bias_params: BiasParams):
    """ Predict ngal from density field. """

    params = bias_params.data

    _allowed_benchmark_models = ['truncated_power_law']
    assert params['predict_counts_model'] in _allowed_benchmark_models

    # Flatten to 1D arrays.
    delta_flattened = bias_model_data.overdensity_field.flatten()
    counts_flattened = bias_model_data.count_field_truth.flatten()

    if params['predict_counts_model'] == "truncated_power_law":
        pl_model = TruncatedPowerLaw()

        # Fit ngal vs delta relation.
        fitted_params = pl_model.fit(delta_flattened, counts_flattened)

        # Get mean ngal for a given value of delta.
        count_mean = pl_model.predict(delta_flattened, fitted_params)

        # Poisson sample a realization using the mean value as the Poisson mu.
        predicted_count_field = pl_model.sample(delta_flattened, fitted_params).reshape(bias_model_data.count_field_truth.shape)

        # Store predicted count field.
        params['count_field_benchmark_name'] = 'truncated power law'
        bias_model_data.count_field_benchmark = predicted_count_field
