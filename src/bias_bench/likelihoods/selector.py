from bias_bench.likelihoods.NegativeBinomial import NegativeBinomial
from bias_bench.likelihoods.Poisson import Poisson
from bias_bench.likelihoods.Gaussian import Gaussian
from bias_bench.likelihoods.LogNormal import LogNormal
from bias_bench.likelihoods.LogSkewNormal import LogSkewNormal


def select_likelihood(likelihood_name):
    match likelihood_name:
        case 'poisson':
            return Poisson()
        case 'nb':
            return NegativeBinomial()
        case 'gaussian':
            return Gaussian()
        case 'lognormal':
            return LogNormal()
        case 'logskewnormal':
            return LogSkewNormal()
        case _:
            raise NotImplementedError(f"{likelihood_name} not part of implemented likelihoods.")
