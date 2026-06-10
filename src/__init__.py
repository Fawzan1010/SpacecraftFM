"""SpacecraftFM: Foundation Model for Spacecraft Attitude Dynamics."""

__version__ = "0.1.0"
__author__ = "Fawzan1010"

from . import api
from . import data
from . import features
from . import models
from . import filtering
from . import visualization
from . import evaluation
from . import utils

__all__ = [
    "api",
    "data",
    "features",
    "models",
    "filtering",
    "visualization",
    "evaluation",
    "utils",
]
