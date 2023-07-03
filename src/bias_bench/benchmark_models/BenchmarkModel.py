from abc import ABC, abstractmethod


class BenchmarkModel(ABC):
    bounds = []

    def __init__(self, name, n_params):
        self.name = name
        self.n_params = n_params

    pass

    @abstractmethod
    def predict(self, input_xs, params):
        """
            Parameters
            ----------
            input_xs : ndarray
                Input fields with shape (batch_size, N, N, N)
            params : ndarray
                (Optimized) model parameters used to produce predicted count fields


            Returns
            -------
            ndarray
                Predicted count field with shape (batch_size, N, N, N) same as the input shape
            """
        pass
