from abc import ABC, abstractmethod


class DataLikelihood(ABC):
    def __init__(self, name, data, params):
        self.name = name
        self.data = data

    @abstractmethod
    def negLogLike(self, prediction):
        """
        Parameters
        ----------
        input_xs : ndarray
            The predicted data produced from the bias model

        Returns
        -------
        float
            The negative log-likelihood corresponding to the underlying data and likelihood model parameters
        """

    pass
