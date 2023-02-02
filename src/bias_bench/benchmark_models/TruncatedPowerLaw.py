import numpy as np
import scipy
from numba import njit

@njit
def _poisson_loop(ngal_mean):
    """
    Predict the actual number of galaxies, given the expected number, using
    Poisson sampling.

    Parameters
    ----------
    ngal_mean : ndarray
        Expected number of galaxies

    Returns
    -------
    ngal : ndarray
        Predicted galaxy counts
    """

    ngal = np.zeros_like(ngal_mean)
    
    for i in range(len(ngal)):
        ngal[i] = np.random.poisson(ngal_mean[i])

    return ngal

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
    x = (d / rho_g)**(-epsilon_g)
    ngal_mean[idx] = nmean * d ** beta * np.exp(-x)

    return ngal_mean

class TruncatedPowerLaw:

    def fit(self, delta, count_field):
        # Fit using SciPy curvefit.
        popt, pcov = scipy.optimize.curve_fit(_get_mean_ngal, delta, count_field,
                p0=[1., 1, 1, 0.5])
        print(f"Power law bias fit params: {popt}")
        return popt

    def predict(self, delta, popt):
        # Predict expected value ngal from delta_dm (and model params popt).
        ngal = _get_mean_ngal(delta, *popt)
        return ngal

    def sample(self, delta, popt):
        # Poisson sample ngal from delta_dm around ngal mean (and model params popt).
        ngal_mean = self.predict(delta, popt)
        return _poisson_loop(ngal_mean)