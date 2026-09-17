"""Prompt-injection attack corpus, defenses, and measurement/statistics
for the rank-shift study, organized into three modules:

- ``attacks``: attack corpora (instruction injection, comparative
  injection) and the filter-aware adaptive-evasion search.
- ``defenses``: prompt-level defenses (isolation instruction,
  self-reminder), the embedding-based semantic filter, and the
  off-the-shelf classifier filter.
- ``measurement``: rank-shift measurement and Wilcoxon/Holm-Bonferroni
  significance analysis.

This file re-exports the package's public surface for convenience, so
``from candidate_ranking.injection import X`` works without knowing
which of the three modules ``X`` lives in.
"""

from candidate_ranking.injection.attacks import (
    ATTACK_CATEGORIES,
    COMPARATIVE_INJECTION_PARAPHRASES,
    DEFAULT_SYNONYM_BANK,
    INSTRUCTION_INJECTION_PARAPHRASES,
    build_injected_candidate,
    marker_survived,
    optimize_evasive_attack,
    select_pair_subsample,
    stratified_sample_pairs,
)
from candidate_ranking.injection.defenses import (
    HARDENED_ASSESSMENT_GENERATION_PROMPT,
    REFERENCE_SUSPICIOUS_PHRASES,
    SELF_REMINDER_ASSESSMENT_GENERATION_PROMPT,
    build_classifier_filter,
    build_hardened_assessment_chain,
    build_self_reminder_assessment_chain,
    filter_suspicious_lines,
    filter_suspicious_lines_classifier,
)
from candidate_ranking.injection.measurement import (
    compute_rank_shift,
    holm_correct,
    paired_deltas_by_condition,
    paired_deltas_vs_control,
    rank_biserial,
    wilcoxon_result,
)

__all__ = [
    "ATTACK_CATEGORIES",
    "COMPARATIVE_INJECTION_PARAPHRASES",
    "DEFAULT_SYNONYM_BANK",
    "HARDENED_ASSESSMENT_GENERATION_PROMPT",
    "INSTRUCTION_INJECTION_PARAPHRASES",
    "REFERENCE_SUSPICIOUS_PHRASES",
    "SELF_REMINDER_ASSESSMENT_GENERATION_PROMPT",
    "build_classifier_filter",
    "build_hardened_assessment_chain",
    "build_injected_candidate",
    "build_self_reminder_assessment_chain",
    "compute_rank_shift",
    "filter_suspicious_lines",
    "filter_suspicious_lines_classifier",
    "holm_correct",
    "marker_survived",
    "optimize_evasive_attack",
    "paired_deltas_by_condition",
    "paired_deltas_vs_control",
    "rank_biserial",
    "select_pair_subsample",
    "stratified_sample_pairs",
    "wilcoxon_result",
]
