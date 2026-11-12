# Import quantities first so the built-in registry is populated.
from . import quantities as _quantities  # noqa: F401
from .core import STAGGER_HALO, BifrostData, UnknownVariableError
from .registry import Variable, register_variable, register_vector
from .responses import INSTRUMENT_RESPONSE_NAMES, InstrumentResponseTable
from .stagger import StaggerInterface
from .version import __version__


def register_colormaps():
    """Register Bifrost's terrain and instrument colormaps in Matplotlib."""
    from .colormaps import register_colormaps as _register

    return _register()


__all__ = [
    "INSTRUMENT_RESPONSE_NAMES",
    "STAGGER_HALO",
    "BifrostData",
    "InstrumentResponseTable",
    "StaggerInterface",
    "UnknownVariableError",
    "Variable",
    "__version__",
    "register_colormaps",
    "register_variable",
    "register_vector",
]
