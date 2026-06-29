"""Built-in regulatory domains, registered on import."""

from ..config import PROJECT_ROOT
from .base import DeadlineRule, Domain, RequiredCitationRule, SymbolicRules
from .registry import register

_SPECIAL_CATEGORY_TRIGGERS = ("special category", "special categories", "health", "biometric")

GDPR = Domain(
    name="gdpr",
    title="General Data Protection Regulation",
    description="EU regulation on data protection and privacy.",
    source_path=PROJECT_ROOT / "data" / "gdpr.md",
    unit_label="article",
    symbolic_rules=SymbolicRules(
        required_citation_rules=(
            RequiredCitationRule(
                rule_id="special_category_requires_basis",
                trigger_groups=(_SPECIAL_CATEGORY_TRIGGERS,),
                required_citations=("article-9", "article-6"),
                passed_message=(
                    "Special-category processing cites both the specific condition "
                    "and lawful basis."
                ),
                missing_message="Special-category answer should cite",
            ),
            RequiredCitationRule(
                rule_id="withdrawal_erasure_chain",
                trigger_groups=(
                    ("withdraw", "withdraws", "withdrawal"),
                    ("erase", "erasure", "deleted"),
                ),
                required_citations=("article-17", "article-7", "article-6"),
                passed_message=(
                    "Withdrawal/erasure answer cites erasure, consent, and lawful-basis clauses."
                ),
                missing_message="Withdrawal/erasure answer should cite",
            ),
        ),
        deadline_rules=(
            DeadlineRule(
                rule_id="breach_notification_deadline",
                trigger_groups=(("breach", "notify", "notification"),),
                citation="article-33",
                deadline_pattern=r"\b72\s+hours?\b",
                passed_message=(
                    "Breach notification answer includes the deadline and source clause."
                ),
                missing_message="Breach notification answer should",
                missing_deadline_message="mention the 72 hour deadline",
            ),
        ),
    ),
)

UK_DPA = Domain(
    name="uk_dpa",
    title="UK Data Protection Act 2018 (excerpt)",
    description="Selected sections of the UK Data Protection Act 2018.",
    source_path=PROJECT_ROOT / "data" / "uk_dpa_excerpt.md",
    unit_label="section",
    symbolic_rules=SymbolicRules(
        required_citation_rules=(
            RequiredCitationRule(
                rule_id="special_category_requires_basis",
                trigger_groups=(_SPECIAL_CATEGORY_TRIGGERS,),
                required_citations=("section-10", "section-8"),
                passed_message=(
                    "Special-category processing cites both the specific condition "
                    "and lawful basis."
                ),
                missing_message="Special-category answer should cite",
            ),
        ),
        deadline_rules=(
            DeadlineRule(
                rule_id="breach_notification_deadline",
                trigger_groups=(("breach", "notify", "notification"),),
                citation="section-67",
                deadline_pattern=r"\b72\s+hours?\b",
                passed_message=(
                    "Breach notification answer includes the deadline and source clause."
                ),
                missing_message="Breach notification answer should",
                missing_deadline_message="mention the 72 hour deadline",
            ),
        ),
    ),
)

register(GDPR)
register(UK_DPA)
