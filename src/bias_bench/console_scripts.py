from bias_bench.Params import BiasParams
from bias_bench.data_io import BiasModelData
from bias_bench.plot import plot_bias_model_metrics
from bias_bench.predict import predict_galaxy_counts

import sys


def run_test_bench():
    param_file = sys.argv[1]

    # Load information from parameter file.
    params = BiasParams(param_file)

    # Load data.
    bias_models = []
    for i in range(params.data['num_bias_models']):
        bias_model_id = i + 1
        bias_models.append(BiasModelData(params, model_index=bias_model_id))

        # Predict counts using benchmark models.
        # TODO: extract this to separate post-processing function
        if params.data[f'bias_model_{bias_model_id}']["predict_counts_model"] is not None:
            predict_galaxy_counts(bias_models[i], params, which_model=bias_model_id)

    plot_bias_model_metrics(bias_models, params)


if __name__ == '__main__':
    run_test_bench()
