from .NegativeBinomial import NegativeBinomial
from .Poisson import Poisson
from .Gaussian import Gaussian
from .LogNormal import LogNormal
from .LogSkewNormal import LogSkewNormal


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
