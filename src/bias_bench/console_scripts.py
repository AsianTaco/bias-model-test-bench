import argparse
from os import path

import numpy as np
import h5py as h5

from bias_bench.Params import BiasParams
from bias_bench.data_io import BiasModelData
from bias_bench.plot import plot_bias_model_metrics
from bias_bench.predict import predict_galaxy_counts
from bias_bench.processing import *
from bias_bench.utils import *


def bundle_catalogs():
    parser = argparse.ArgumentParser(
        description='Read in halo or galaxy catalogs and bundling them into a hdf5 file as required by the test bench.')

    parser.add_argument('--cat_base_path', type=str,
                        help='Base path to the catalogs')
    parser.add_argument('--n_cats', type=int, default=1,
                        help='Number of catalogs to load.')
    parser.add_argument('--catalog_format', type=str, default='.ascii',
                        help='Datatype of the provided catalogs.')

    script_args = parser.parse_args()

    cat_paths = [f'{script_args.cat_base_path}_{cat_i}{script_args.catalog_format}'
                 for cat_i in range(script_args.n_cats)]
    cats = read_rockstar_ascii_cat(cat_paths, 2e11, capital_id=True, convert_to_numpy=True)

    # TODO: make this a configurable parameter
    mass_bins = np.array([5.e+13, 7.e+13, 1.e+14, 5.e+14, 1.e+15])
    mesh_size = 128
    box_size = 1000
    coords_per_mass_bin = get_halo_coords_per_mass_bin(mass_bins, cats[0])
    halo_count_fields_per_mass_bin = [get_halo_count_field(coords_per_mass_bin[mass_bin_id], mesh_size) for
                                      mass_bin_id in range(mass_bins.size - 1)]

    halo_count_fields_per_mass_bin = np.asarray(halo_count_fields_per_mass_bin)

    # TODO: make this a configurable parameter
    save_path = './examples/test.hdf5'
    with h5.File(save_path, 'a') as f:
        # TODO: include more resolutions
        f.create_group(f"npe_0/res_0")
        f[f"npe_0/res_0"].attrs.create("ngrid", mesh_size)
        f[f"npe_0/res_0"].attrs.create("boxsize", box_size)

        # TODO: Add overdensities
        # f.create_dataset(f'npe_{seed_i}/res_0/dm_overdensity', data=over_density)

    add_count_fields_to_hdf5(save_path, f"npe_0/res_0", halo_count_fields_per_mass_bin, mass_bins,
                             is_ground_truth=True,
                             all_mass_bin=False)


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


def visualise_data():
    parser = argparse.ArgumentParser(
        description='Visualise the given data and fit likelihoods onto it.')

    parser.add_argument('--param_file', type=str, default='./examples/example_config.yml',
                        help='Full path to the config .yml file')
    parser.add_argument('--fit_likelihood', type=str, default=None,
                        help='Fit the likelihood to the data. Supported likelihoods are: Poisson, NegativeBinomial,...')
    parser.add_argument('--save_fits', action='store_true',
                        help='Save likelihood fits to avoid re-optimization each time')

    script_args = parser.parse_args()

    # Load information from parameter file.
    params = BiasParams(script_args.param_file)

    # Load data
    bias_models = []
    for i in range(params.data['num_bias_models']):
        bias_model_id = i + 1

        visualise_data(BiasModelData(params, model_index=bias_model_id))


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
    #     #TMP: As far as I understand, bias_models here is one piece of data + predictions. It has 3 attributes,
    #     #     namely: - overdensity_field (3d array [Ngrid,Ngrid,Ngrid]), # dm overdensity field
    #     #             - count_field (3d array [Ngrid,Ngrid,Ngrid]),       # predicted count field
    #     #             - count_field_truth (3d array [Ngrid,Ngrid,Ngrid])  # ground truth
