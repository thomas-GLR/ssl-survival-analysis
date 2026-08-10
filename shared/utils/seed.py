import random

import numpy as np
import torch


def set_seed(seed: int | None) -> None:
    """Seed every random number generator a training run draws from.

    Called at the start of each ``train_model`` entry point so that a run is
    reproducible from the ``seed`` already present in the config's
    ``dataset_params``. Without this, only numpy is seeded (inside the dataset
    classes), leaving weight initialisation, dropout masks and DataLoader
    shuffling driven by OS entropy.

    ``torch.cuda.manual_seed`` and ``torch.cuda.manual_seed_all`` are safe to
    call when no CUDA device is available -- torch turns them into no-ops.

    Args:
        seed: The seed to apply. ``None`` leaves every generator untouched, so a
            caller that deliberately wants a non-deterministic run keeps it.

    Returns:
        None.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
