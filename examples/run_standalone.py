from bias_bench.io import BiasModelData
from bias_bench.Params import BiasParams

from bias_bench.plot import plot_bias_model_metrics
from bias_bench.predict import predict_galaxy_counts

import sys

if __name__ == '__main__':

    param_file = sys.argv[1]

    # Load information from parameter file.
    params = BiasParams(param_file)
    bias_param_data = params.data

    # Load data.
    BM = BiasModelData(params)

    # Predict counts using benchmark models.
    # TODO: extract this to separate post-processing function
    if bias_param_data["predict_counts_model"] is not None:
        predict_galaxy_counts(BM, params)

    plot_bias_model_metrics(BM, params)
