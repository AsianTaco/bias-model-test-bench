from bias_bench.Params import BiasParams
from bias_bench.data_io import BiasModelData
from bias_bench.plot import plot_bias_model_metrics
from bias_bench.predict import predict_galaxy_counts

import argparse
from os import path

import numpy as np


def run_test_bench():
    parser = argparse.ArgumentParser(
        description='Visualise the results of a given bias model (optional), train a benchmark model on the fly,'
                    'and generate comparison plots for different metrics.')

    parser.add_argument('--param_file', type=str, default='./examples/example_config.yml',
                        help='Full path to the config .yml file')
    parser.add_argument('--skip_benchmark', action='store_true',
                        help='Skip on the fly training of benchmark model.')
    parser.add_argument('--save_benchmark', action='store_true',
                        help='Save results of the benchmark model to avoid re-optimization each time')

    script_args = parser.parse_args()

    # Load information from parameter file.
    params = BiasParams(script_args.param_file)

    # Load data.
    bias_models = []
    for i in range(params.data['num_bias_models']):
        bias_model_id = i + 1
        bias_models.append(BiasModelData(params, model_index=bias_model_id))

        # Predict counts using benchmark models.
        # TODO: extract this to separate post-processing function
        benchmark_model = params.data[f'bias_model_{bias_model_id}']["predict_counts_model"]
        if (benchmark_model is not None) and not script_args.skip_benchmark:
            benchmark_file_abs_path = f"{params.data['out_dir']}/benchmark_{benchmark_model}.npy"

            if path.exists(benchmark_file_abs_path):
                print('Loading pretrained field for benchmark model \n'
                      f'model name: {benchmark_model} \n'
                      f'file location: {benchmark_file_abs_path}')
                bias_models[i].counts_field_benchmark = np.load(benchmark_file_abs_path)
            else:
                counts_field_benchmark = predict_galaxy_counts(bias_models[i], params, which_model=bias_model_id)
                bias_models[i].counts_field_benchmark = counts_field_benchmark
                if script_args.save_benchmark:
                    np.save(benchmark_file_abs_path, counts_field_benchmark)

    plot_bias_model_metrics(bias_models, params)


def optimise_borg_bias():
    import sys
    param_file = sys.argv[1]

    # Load information from parameter file.
    params = BiasParams(param_file)

    print(params.data)
    bias_model = BiasModelData(params, model_index=1)
    print(bias_model.params)

    # # Load data.
    # bias_models = []
    # for i in range(params.data['num_bias_models']):
    #     bias_model_id = i + 1
    #     bias_models.append(BiasModelData(params, model_index=bias_model_id))
    #     #TMP: As far as I understand, bias_models here is is one piece of data + predictions. It has 3 attributes,
    #     #     namely: - overdensity_field (3d array [Ngrid,Ngrid,Ngrid]), # dm overdensity field
    #     #             - count_field (3d array [Ngrid,Ngrid,Ngrid]),       # predicted count field
    #     #             - count_field_truth (3d array [Ngrid,Ngrid,Ngrid])  # ground truth


if __name__ == '__main__':
    run_test_bench()
