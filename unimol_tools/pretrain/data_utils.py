import contextlib
import numpy as np


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Seed NumPy's RNG in a local context.

    This mirrors Uni-Core's utility so that random operations in datasets
    are deterministic given the training epoch and sample index.  The previous
    RNG state is restored when leaving the context.
    """
    if seed is None:
        yield
        return
    if addl_seeds:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
