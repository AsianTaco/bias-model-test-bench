from bias_bench.likelihoods.Poisson import Poisson


def select_likelihood(likelihood_name, params):
    match likelihood_name:
        case 'poisson':
            return Poisson(params)
        case _:
            raise NotImplementedError(f"{likelihood_name} not part of implemented likelihoods.")
