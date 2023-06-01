from bias_bench.likelihoods.Poisson import Poisson


def select_likelihood(likelihood_name):
    match likelihood_name:
        case 'poisson':
            return Poisson()
        case _:
            raise NotImplementedError(f"{likelihood_name} not part of implemented likelihoods.")
