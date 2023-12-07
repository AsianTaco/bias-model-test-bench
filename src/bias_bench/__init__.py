try:
    # For Python >= 3.8
    from importlib import metadata
except ImportError:
    # For Python < 3.8
    import importlib_metadata as metadata

__version__ = metadata.version("bias_bench")

from bias_bench.processing import *
from bias_bench.utils import *
from bias_bench import likelihoods, optimizer, benchmark_models, analysis
