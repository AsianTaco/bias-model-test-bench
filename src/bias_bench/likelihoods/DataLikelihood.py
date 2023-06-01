from abc import ABC, abstractmethod


class DataLikelihood(ABC):
    def __init__(self, name, n_params):
        self.name = name
        self.n_params = n_params

    @abstractmethod
    def negLogLike(self, data, prediction, params):
        """
        Parameters
        ----------
        data : ndarray
            Underlying data given by observations/mocks.

        prediction : ndarray
            The predicted data produced from the bias model.


        Returns
        -------
        float
            The negative log-likelihood corresponding to the underlying data and likelihood model parameters
        """

    pass

    @abstractmethod
    def sample(self, prediction, params):
        """
        Parameters
        ----------
        prediction : ndarray
            The prediction made by the bias model.

        params : ndarray
            The free likelihood parameters e.g. variance, skewness etc.


        Returns
        -------
        ndarray
            A sampled realization generated from the prediction adding the correct stochasticity from the likelihood.
        """

    pass
