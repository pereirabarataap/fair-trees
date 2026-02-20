"""Fair Trees: fairness-aware decision tree and random forest classifiers."""

import importlib as _importlib
import logging
import os
import random

from ._config import config_context, get_config, set_config

logger = logging.getLogger(__name__)

__version__ = "3.1.8"

# OpenMP compatibility settings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

from . import (  # noqa: F401 E402
    __check_build,
    _distributor_init,
)

# Public API — only the two classifiers
from ._datasets import load_datasets # noqa: E402
from .tree._classes import FairDecisionTreeClassifier  # noqa: E402
from .ensemble._forest import FairRandomForestClassifier  # noqa: E402

__all__ = [
    "load_datasets",
    "FairDecisionTreeClassifier",
    "FairRandomForestClassifier",
]

_BUILT_WITH_MESON = False
try:
    import fair_trees._built_with_meson  # noqa: F401

    _BUILT_WITH_MESON = True
except ModuleNotFoundError:
    pass
