from abc import ABC, abstractmethod


class BenchmarkModel(ABC):
    """
    Abstract class for benchmark models.

    ...

    Attributes
    ----------
    name : str
        name of the model
    n_params : int
        number of parameters in the model
    bounds : list
        lower and upper bounds of each parameter given as a tuple
    age : int
        age of the person

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    bounds = []

    def __init__(self, name, n_params):
        self.name = name
        self.n_params = n_params

    pass

    def check_bounds(self, params):
        """
        Function to check that the given parameters are within the bounds of the model

        Parameters
        ----------

        bounds: list
            A list of tuples specifying the lower and upper bounds of the parameter.
        params: np.ndarray
            Array of parameters to be checked if they are within the given bounds

        Returns
        -------
        is_out_of_bounds : bool
            Boolean value if the set of parameters is out of bounds
        """

        assert self.n_params == params.size
        assert self.n_params == len(self.bounds)

        is_out_of_bounds = False

        for idx, param in enumerate(params):
            lower, upper = self.bounds[idx]

            if param < lower or param > upper:
                is_out_of_bounds= True

        return is_out_of_bounds

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
