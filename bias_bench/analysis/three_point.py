from bias_bench.io import BiasModelData

import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL


def compute_bispectrum(bias_model_data: BiasModelData):
    l_box = bias_model_data.info['BoxSize']
    # FIXME: Enable use of double precision
    overdensity = np.array(bias_model_data.overdensity_field, dtype=np.float32)
    k1 = 0.5  # h/Mpc
    k2 = 0.6  # h/Mpc
    theta = np.linspace(0, np.pi, 25)  # array with the angles between k1 and k2

    # TODO: pipe out pylians parameters
    bbk = PKL.Bk(overdensity, l_box, k1=k1, k2=k2, theta=theta, MAS='CIC')

    # FIXME: combine this with power spectrum computation
    # Power spectrum for free
    # k = bbk.k  # k-bins for power spectrum
    # Pk = bbk.Pk # power spectrum

    return theta, {'bispectrum': bbk.B, 'reduced_bispectrum': bbk.Q}
