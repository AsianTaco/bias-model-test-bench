from bias_bench.benchmark_models.LinearBias import LinearBias
from bias_bench.benchmark_models.TruncatedPowerLaw import TruncatedPowerLaw


def select_bias_model(model_name: str):
    match model_name:
        case 'truncated_power_law':
            return TruncatedPowerLaw()
        case 'linear':
            return LinearBias()
        case _:
            raise NotImplementedError(f"{model_name} not part of implemented bias models.")
