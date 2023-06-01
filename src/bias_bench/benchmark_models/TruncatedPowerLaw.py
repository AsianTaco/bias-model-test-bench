import numpy as np
from bias_bench.benchmark_models.BenchmarkModel import BenchmarkModel


def _get_mean_ngal(rho, nmean, beta, epsilon_g, rho_g):
    """
    Using the truncated power law bias model, compute the mean number of
    galaxies expected for a given density.

    Incoming density field is expected to be 1D, i.e., the flattened 3D array.

    Parameters
    ----------
    rho : ndarray
        The 3D density field
    nmean : float
        ...
    beta : float
        ...
    epsilon_g
        ...
    rho_g : float
        ...

    Returns
    -------
    ngal_mean : ndarray
        Expected number of galaxies for the given densities
    """

    # Return array
    ngal_mean = np.zeros_like(rho)

    # Non zero densites
    idx = np.where(rho > -1.0)

    d = 1 + rho[idx]
    x = (d / rho_g) ** (-epsilon_g)
    ngal_mean[idx] = nmean * d ** beta * np.exp(-x)

    return ngal_mean


class TruncatedPowerLaw(BenchmarkModel):

    def __init__(self):
        super().__init__("Truncated Power Law", 4)
        self.bounds = [(1., 1e8), (1e-6, 6.), (1e-6, 3.), (1e-6, 1e5)]

    def predict(self, delta, popt):
        # Predict expected value ngal from delta_dm (and model params popt).
        ngal = _get_mean_ngal(delta.flatten(), *popt)
        return ngal.reshape(delta.shape)
