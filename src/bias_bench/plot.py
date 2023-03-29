from pathlib import Path
from typing import Sequence

from bias_bench.data_io import BiasModelData
from bias_bench.Params import BiasParams
from bias_bench.analysis import *


def plot_bias_model_metrics(bias_model_data: Sequence[BiasModelData], bias_params: BiasParams):
    params = bias_params.data

    parent_folder_path = f"{params['out_dir']}/plots"
    Path(parent_folder_path).mkdir(parents=True, exist_ok=True)

    # plt.style.use(f"./plot_styles/{params['plotting_style']}")

    # Make plots.
    # TODO: Add functionality to plot the mean quantities over all simulations
    if 'ngal_vs_rho' in params['plots']:
        dir_path = f"{parent_folder_path}/one_point"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        plot_one_point_stats(bias_model_data, params, dir_path)

    # FIXME: make other plotting routines work again
    # if 'power_spectrum' in params['plots']:
    #     plot_power_spectrum(bias_model_data, params)
    # if 'bi_spectrum' in params['plots']:
    #     plot_bispectrum(bias_model_data, params)
    # if 'density_images' in params['plots']:
    #     plot_density_field(bias_model_data, params)
