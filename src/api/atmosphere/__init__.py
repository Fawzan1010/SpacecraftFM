"""Atmospheric model clients."""

from .nrlmsise00_client import NRLMSISE00Client
from .jb2008_client import JB2008Client

__all__ = ["NRLMSISE00Client", "JB2008Client"]
