import numpy as np
from os import environ
environ["PYBORG_QUIET"] = "yes"
import borg
from scipy.optimize import minimize
from numba import njit

from bias_bench.utils import bias_bench_print
from bias_bench.benchmark_models.BenchmarkModel import BenchmarkModel

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

class BORGModel(BenchmarkModel):

    def __init__(self, model_name, L):
        """
        Parameters
        ----------
        model_name : str
            Name of the BORG model to use
        L : float
            Comoving length of the simulation boxes in Mpc/h
        """
        super().__init__("BORGModel")
        self.borg_model = model_name
        self.L = L

    def _initialize_models(self, N, L):
        self.bias_models = []
        for i in range(N):
            box = borg.forward.BoxModel()
            box.L = [L]*3 ; box.N = [N]*3
            bias_model = borg.forward.models.newModel(self.borg_model, box, {})
            bias_model.setName('bias_{}'.format(i))
            self.bias_models.append(bias_model)

    def _get_mean_ngal(self, i, parameters):
        """
        Using the chosen BORG bias model, compute the expected count field
        (Poisson intensity) for a given density.

        Parameters
        ----------
        i : int
            Index of current data seed

        Returns
        -------
        ngal_mean : ndarray
            Expected number of galaxies for the given densities
        """
        bias_model = self.bias_models[i]
        bias_model.setModelParams({'biasParameters': parameters})
        bias_model.forwardModel_v2(self.deltas[i])
        count = np.empty(bias_model.getOutputBoxModel().N)
        bias_model.getDensityFinal(count)
        return count
    
    def _Poisson_loss(self, parameters):
        err = 0
        for i in range(self.batch_size):
            count = self._get_mean_ngal(i, parameters)
            gt = self.count_fields[i]
            count[count<=0] = 1e-10 ; gt[gt<=0] = 1e-10
            err += np.sum(-gt*np.log(count)+count) / self.batch_size
        return err
    
    def _callback(self, parameters):
        """
        Callback function to print the current loss and parameters
        """
        err = self._Poisson_loss(parameters)
        print("Parameters: {}".format(parameters))
        print("Loss: {}".format(err))
        
    def fit(self, delta, count_field, bias_model, initial_guess, bounds):
        """
        Parameters
        ----------
        delta (input_xs) : ndarray
            Input fields to train on with shape (batch_size, N, N, N)
        count_field (output_truths) : ndarray
            Underlying ground truth fields to the input fields with shape (batch_size, N, N, N)
        bias_model : str
            Name of the BORG model to use. Possible values are:
                - "bias::Linear"
                - "bias::BrokenPowerLaw"
                - "bias::SecondOrderBias"
        initial_guess : ndarray
            Initial guess for the model parameters
        bounds : ndarray
            Bounds for the model parameters

        Returns
        -------
        ndarray
            Optimized parameters for the model
        """
        self.batch_size = delta.shape[0]
        self.deltas = delta
        self.count_fields = count_field

        self._initialize_models(delta.shape[1], self.L)

        try:
            res = minimize(self._Poisson_loss, initial_guess, method="L-BFGS-B", callback=self._callback,
                            tol=1e-10,
                            bounds=bounds[bias_model],
                            options={'disp': True, 'eps':1e-5, 'finite_diff_rel_step': 1e-5})
            bias_bench_print(f"BORG bias model best fit params: {res.x}")
            return res.x
        except RuntimeError:
            return None

    def predict_poisson_intensity(self, popt):
        # Predict expected value ngals_mean from delta_dm (and model params popt).
        ngals_mean = np.zeros_like(self.deltas)
        for i in range(self.batch_size):
            ngals_mean[i] = self._get_mean_ngal(i, popt)
        return ngals_mean

    def predict(self, delta, popt):
        # Poisson sample ngal from delta_dm around ngal mean (and model params popt).
        self.deltas = delta
        self.batch_size = delta.shape[0]
        ngals_mean = self.predict_poisson_intensity(popt)
        return _poisson_loop(ngals_mean).reshape(delta.shape)
