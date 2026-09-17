"""Prompt-injection attack corpus, defenses, and measurement/statistics
for the rank-shift study. Submodules stay import-by-path (e.g.
``candidate_ranking.injection.attack_corpus``) for existing call sites;
this file just re-exports the public surface for convenience."""

from candidate_ranking.injection.adaptive_attack import (
    DEFAULT_SYNONYM_BANK,
    optimize_evasive_attack,
)
from candidate_ranking.injection.attack_corpus import (
    ATTACK_CATEGORIES,
    COMPARATIVE_INJECTION_PARAPHRASES,
    INSTRUCTION_INJECTION_PARAPHRASES,
    build_injected_candidate,
    marker_survived,
    select_pair_subsample,
    stratified_sample_pairs,
)
from candidate_ranking.injection.classifier_filter import (
    build_classifier_filter,
    filter_suspicious_lines_classifier,
)
from candidate_ranking.injection.mitigation import (
    HARDENED_ASSESSMENT_GENERATION_PROMPT,
    SELF_REMINDER_ASSESSMENT_GENERATION_PROMPT,
    build_hardened_assessment_chain,
    build_self_reminder_assessment_chain,
)
from candidate_ranking.injection.rank_shift import compute_rank_shift
from candidate_ranking.injection.semantic_filter import filter_suspicious_lines
from candidate_ranking.injection.semantic_filter_reference import REFERENCE_SUSPICIOUS_PHRASES
from candidate_ranking.injection.stats import (
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
