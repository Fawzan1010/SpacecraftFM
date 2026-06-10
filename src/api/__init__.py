"""API clients for data acquisition."""

from . import auth
from . import nasa
from . import esa
from . import noaa
from . import igrf
from . import atmosphere

__all__ = [
    "auth",
    "nasa",
    "esa",
    "noaa",
    "igrf",
    "atmosphere",
]
