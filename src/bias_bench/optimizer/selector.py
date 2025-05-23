from bias_bench.optimizer.emcee_minimize import EMCEESampler
from bias_bench.optimizer.scipy_minimize import BFGSScipy


def select_optimizer(optimizer_name: str, likelihood, model, optimizer_args):
    match optimizer_name:
        case 'emcee':
            return EMCEESampler(likelihood, model, *optimizer_args)
        case 'BFGS':
            return BFGSScipy(likelihood, model)
        case _:
            raise NotImplementedError(f"{optimizer_name} not part of implemented bias models.")
