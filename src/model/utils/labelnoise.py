from typing import Any, Callable, Optional, Tuple

import numpy as np


def label_noise(train_dataset, noise_level):
    np.random.seed(2021)
    num_samples = len(train_dataset.targets)

    noise_level = noise_level * 1e-2
    if noise_level == 0:
        noise_level = 1

    rands = np.random.choice(num_samples, int(num_samples * noise_level), replace=False)

    for rand in rands:
        tmp = train_dataset.targets[rand]
        train_dataset.targets[rand] = np.random.choice(
            list(range(0, tmp)) + list(range(tmp + 1, 10))
        )

    return train_dataset
