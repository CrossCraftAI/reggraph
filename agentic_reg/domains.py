"""Pluggable regulatory domains.

A domain describes a regulation: its source text, how to chunk it, and what
symbolic rules apply. Domains are registered at import time via ``register()``
or discovered from installed packages via the ``agentic_reg.domains`` entry-point
group.
"""

from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

from agentic_reg.config import PROJECT_ROOT


@dataclass
class Domain:
    """A regulatory domain with its source document and processing config."""

    name: str
    title: str
    description: str
    source_path: Path
    unit_label: str = "article"  # article, section, chapter
    chunk_size: int = 512
    chunk_overlap: int = 64

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Domain name must not be empty")


_registry: dict[str, Domain] = {}


def register(domain: Domain) -> None:
    """Register a domain so it is discoverable by ``get_domain``."""
    _registry[domain.name] = domain


def get_domain(name: str) -> Domain:
    """Return the domain registered under *name*.

    Raises ``KeyError`` if the domain is not found.
    """
    if not _registry:
        _discover_entry_points()
    if name not in _registry:
        raise KeyError(f"Unknown domain: {name!r}. Registered: {list(_registry)}")
    return _registry[name]


def list_domains() -> list[str]:
    """Return the names of all registered domains."""
    if not _registry:
        _discover_entry_points()
    return sorted(_registry)


def _discover_entry_points() -> None:
    """Discover domains from installed packages via ``agentic_reg.domains`` entry points."""
    try:
        eps = entry_points(group="agentic_reg.domains")
    except TypeError:
        return
    for ep in eps:
        try:
            ep.load()  # the entry point calls register() on import
        except Exception:  # noqa: BLE001
            pass


# ── built-in domains ────────────────────────────────────────────────────────
# Register the bundled domains so they are always available without extra setup.

register(
    Domain(
        name="gdpr",
        title="General Data Protection Regulation",
        description="EU regulation on data protection and privacy.",
        source_path=PROJECT_ROOT / "data" / "gdpr.md",
        unit_label="article",
    )
)
