from abc import ABC, abstractmethod
from collections import namedtuple

from bias_bench.benchmark_models.BenchmarkModel import BenchmarkModel
from bias_bench.likelihoods.DataLikelihood import DataLikelihood

# TODO: Create better typing for this
TargetParameters = namedtuple('TargetParameters', 'likelihood_params bias_params')


class BaseOptimizer(ABC):
    def __init__(self, likelihood: DataLikelihood, bias_model: BenchmarkModel):
        self.like = likelihood
        self.model = bias_model

    @abstractmethod
    def optimize(self, input_x, data, init_params) -> TargetParameters:
        """
        Parameters
        ----------
        input_x : ndarray
            Input fields that will be piped through the bias model and then compared to the data.

        data : ndarray
            Observed (mock) data that will be used for the likelihood to make a comparison with the prediction from the bias model.


        Returns
        -------
        TargetParameters
            A set of optimized parameters. The first entry are the likelihood parameters and the second entry the bias model parameters
        """
        pass
