from abc import ABC, abstractmethod


class BenchmarkModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self, input_xs, output_truths):
        """
        Parameters
        ----------
        input_xs : ndarray
            Input fields to train on with shape (batch_size, N, N, N)
        output_truths : ndarray
            Underlying ground truth fields to the input fields with shape (batch_size, N, N, N)

        Returns
        -------
        ndarray
            Optimized parameters for the model
        """

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
