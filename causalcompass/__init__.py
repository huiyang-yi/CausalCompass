from importlib.metadata import PackageNotFoundError, version

from . import algorithms
from . import datasets

try:
    __version__ = version("causalcompass")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
