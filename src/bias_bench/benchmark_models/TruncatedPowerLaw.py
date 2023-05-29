import numpy as np
from scipy.optimize import minimize

from bias_bench.utils import bias_bench_print
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

    def __init__(self, loss):
        super().__init__("Truncated Power Law", loss)

    def _callback(self, parameters):
        """
        Callback function to print the current parameters
        """
        # err = self.loss(parameters)
        # print("Loss: {}".format(err))
        print("Parameters: {}".format(parameters))

    def fit(self, delta, count_field):
        # TODO: extract this to separate module

        try:
            res = minimize(lambda x: self.compute_loss(delta, count_field, x), np.array([1., 1, 1, 0.5]),
                            method="L-BFGS-B", callback=self._callback,
                            tol=1e-10,
                            # bounds=bounds[bias_model], TODO: include bounds
                            options={'disp': True, 'eps': 1e-5, 'finite_diff_rel_step': 1e-5})
            popt = res.x
            # popt, pcov = scipy.optimize.curve_fit(_get_mean_ngal, delta, count_field, p0=, maxfev=4000)
            bias_bench_print(f"Power law bias fit params: {popt}")
            return popt
        except RuntimeError:
            return None

    def compute_loss(self, delta, count_field, params):
        delta = delta.flatten()
        count_field = count_field.flatten()

        prediction = _get_mean_ngal(delta, *params)
        return self.loss.negLogLike(count_field, prediction)

    def predict_poisson_intensity(self, delta, popt):
        # Predict expected value ngal from delta_dm (and model params popt).
        ngal = _get_mean_ngal(delta.flatten(), *popt)
        return ngal.reshape(delta.shape)

    def predict(self, delta, popt):
        # Poisson sample ngal from delta_dm around ngal mean (and model params popt).
        ngal_mean = self.predict_poisson_intensity(delta, popt)
        return np.random.poisson(ngal_mean)
