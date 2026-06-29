"""Domain plugin system: built-in and externally-registered regulatory domains."""

from . import builtin  # noqa: F401 — importing registers the built-in domains
from .base import DeadlineRule, Domain, RequiredCitationRule, SymbolicRules
from .registry import discover_plugins, get_domain, list_domains, register

discover_plugins()  # register any external (pip-installed) domains

__all__ = [
    "DeadlineRule",
    "Domain",
    "RequiredCitationRule",
    "SymbolicRules",
    "get_domain",
    "list_domains",
    "register",
]
