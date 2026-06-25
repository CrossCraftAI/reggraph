"""Domain registry + plugin discovery.

Built-in domains register themselves on import. External packages can ship a
domain by advertising it under the ``agentic_reg.domains`` entry-point group,
so a third party can ``pip install agentic-reg-<domain>`` and have it appear
here automatically — extension without modifying core code.
"""

from importlib.metadata import entry_points

from .base import Domain

_DOMAINS: dict[str, Domain] = {}


def register(domain: Domain) -> None:
    _DOMAINS[domain.name] = domain


def get_domain(name: str) -> Domain:
    if name not in _DOMAINS:
        raise ValueError(f"Unknown domain {name!r}. Available: {sorted(_DOMAINS)}.")
    return _DOMAINS[name]


def list_domains() -> list[Domain]:
    return sorted(_DOMAINS.values(), key=lambda d: d.name)


def discover_plugins() -> None:
    """Register external domains advertised via the 'agentic_reg.domains' group.

    An entry point may point to a ``Domain`` instance or a zero-arg callable
    that returns one. Failures are ignored so a broken plugin can't crash the app.
    """
    try:
        points = entry_points(group="agentic_reg.domains")
    except Exception:
        return
    for point in points:
        try:
            obj = point.load()
            domain = obj if isinstance(obj, Domain) else obj()
            if isinstance(domain, Domain):
                register(domain)
        except Exception:
            continue
