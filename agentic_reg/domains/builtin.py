"""Built-in regulatory domains, registered on import."""

from ..config import PROJECT_ROOT
from .base import Domain
from .registry import register

GDPR = Domain(
    name="gdpr",
    title="General Data Protection Regulation",
    description="EU regulation on data protection and privacy.",
    source_path=PROJECT_ROOT / "data" / "gdpr.md",
    unit_label="article",
)

register(GDPR)
