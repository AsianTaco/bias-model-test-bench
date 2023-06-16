from pathlib import Path
from typing import Sequence

from bias_bench.data_io import BiasModelData
from bias_bench.Params import BiasParams
from bias_bench.analysis import *

import matplotlib.pyplot as plt


def plot_bias_model_metrics(bias_model_data: Sequence[BiasModelData], bias_params: BiasParams):
    params = bias_params.data

    parent_folder_path = f"{params['out_dir']}/plots"
    Path(parent_folder_path).mkdir(parents=True, exist_ok=True)

    # FIXME: enable plotting style files
    plt.style.use(params['plotting_style'])

    # Make plots.
    # TODO: Add functionality to plot the mean quantities over all simulations
    if 'ngal_vs_rho' in params['plots']:
        dir_path = f"{parent_folder_path}/one_point"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        plot_one_point_stats(bias_model_data, params, dir_path)

    # FIXME: make other plotting routines work again
    if 'power_spectrum' in params['plots']:
        plot_power_spectrum(bias_model_data, params, parent_folder_path)
    if 'bi_spectrum' in params['plots']:
        dir_path = f"{parent_folder_path}/three_point"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        plot_bispectrum(bias_model_data, params, dir_path)
    # if 'density_images' in params['plots']:
    #     plot_density_field(bias_model_data, params)
