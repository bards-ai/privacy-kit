"""privacy-kit — PII detection and anonymization."""

from importlib.metadata import PackageNotFoundError, version

from privacy_kit.model import Entity, PiiModel

try:
    __version__ = version("privacy-kit")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["Entity", "PiiModel", "__version__"]
