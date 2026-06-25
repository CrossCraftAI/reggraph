"""Domain plugin system: built-in and externally-registered regulatory domains."""

from . import builtin  # noqa: F401 — importing registers the built-in domains
from .base import Domain
from .registry import discover_plugins, get_domain, list_domains, register

discover_plugins()  # register any external (pip-installed) domains

__all__ = ["Domain", "get_domain", "list_domains", "register"]
