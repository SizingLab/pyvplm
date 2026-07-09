"""pyVPLM: Variable Power-Law regression Models tool."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyvplm")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "unknown"
