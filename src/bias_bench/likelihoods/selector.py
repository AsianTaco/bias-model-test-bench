from bias_bench.likelihoods.NegativeBinomial import NegativeBinomial
from bias_bench.likelihoods.Poisson import Poisson


def select_likelihood(likelihood_name):
    match likelihood_name:
        case 'poisson':
            return Poisson()
        case 'nb':
            return NegativeBinomial()
        case 'gaussian':
            return Gaussian()
        case _:
            raise NotImplementedError(f"{likelihood_name} not part of implemented likelihoods.")
