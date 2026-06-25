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

UK_DPA = Domain(
    name="uk_dpa",
    title="UK Data Protection Act 2018 (excerpt)",
    description="Selected sections of the UK Data Protection Act 2018.",
    source_path=PROJECT_ROOT / "data" / "uk_dpa_excerpt.md",
    unit_label="section",
)

register(GDPR)
register(UK_DPA)
