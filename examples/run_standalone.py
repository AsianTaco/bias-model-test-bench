from bias_bench.io import BiasModelData
from bias_bench.Params import BiasParams

from bias_bench.plot import plot_bias_model_metrics
from bias_bench.predict import predict_galaxy_counts

import sys

if __name__ == '__main__':

    param_file = sys.argv[1]

    # Load information from parameter file.
    params = BiasParams(param_file)

    # Load data.
    BM_list = []
    for i in range(params.data['num_bias_models']):
        BM_list.append(BiasModelData(params, which_model=i+1))

        # Predict counts using benchmark models.
        # TODO: extract this to separate post-processing function
        if params.data[f'bias_model_{i+1}']["predict_counts_model"] is not None:
            predict_galaxy_counts(BM_list[-1], params)

    plot_bias_model_metrics(BM_list, params)
