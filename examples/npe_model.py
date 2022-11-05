from bias_bench.io import BiasModelData
from bias_bench.Params import BiasParams

from bias_bench.plot import plot_bias_model_metrics
from bias_bench.predict import predict_galaxy_counts

if __name__ == '__main__':

    # Load information from parameter file.
    params = BiasParams("examples/gadget_example.yml")
    bias_param_data = params.data

    # Load data.
    BM = BiasModelData(params)

    # if bias_param_data["predict_counts_model"] is not None:
    #     predict_galaxy_counts(BM, params)

    plot_bias_model_metrics(BM, params)