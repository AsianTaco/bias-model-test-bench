import numpy as np

from bias_bench.benchmark_models.BenchmarkModel import BenchmarkModel


class MeanPredictor(BenchmarkModel):

    def __init__(self):
        super().__init__("Mean predictor", 1)
        self.bounds = [(0, 1e8)]

    # TODO: Extend this to be aware of the underlying overdensity
    def predict(self, delta, popt):
        return np.full_like(delta, popt)
