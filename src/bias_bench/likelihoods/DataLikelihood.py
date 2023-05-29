from abc import ABC, abstractmethod


class DataLikelihood(ABC):
    def __init__(self, name, params):
        self.name = name
        self.params = params

    @abstractmethod
    def negLogLike(self, data, prediction):
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
