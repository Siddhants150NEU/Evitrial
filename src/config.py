from __future__ import annotations

import os
import random

import yaml

DEFAULT_CONFIG_PATH = "configs/default.yaml"

def loadConfig(path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def setSeeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
