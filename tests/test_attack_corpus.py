from candidate_ranking.injection.attack_corpus import (
    ATTACK_CATEGORIES,
    COMPARATIVE_INJECTION_PARAPHRASES,
    INSTRUCTION_INJECTION_PARAPHRASES,
    build_injected_candidate,
    select_pair_subsample,
)
from candidate_ranking.models import Candidate

CANDIDATE = Candidate(
    id="cv-001",
    source_path="/cv/1.pdf",
    raw_text="Jane Doe\n\nSUMMARY\nExperienced backend engineer.\n\nSKILLS\nPython, AWS.",
    num_pages=1,
    char_count=80,
    parse_status="ok",
)


def test_attack_categories_registry_has_both_categories():
    assert ATTACK_CATEGORIES["instruction_injection"] is INSTRUCTION_INJECTION_PARAPHRASES
    assert ATTACK_CATEGORIES["comparative_injection"] is COMPARATIVE_INJECTION_PARAPHRASES


def test_build_injected_candidate_defaults_to_instruction_injection():
    injected, marker = build_injected_candidate(CANDIDATE, jd_id="backend-engineer", seed=42)
    assert marker in {p["marker_substring"] for p in INSTRUCTION_INJECTION_PARAPHRASES}
    assert marker.lower() in injected.raw_text.lower()


def test_build_injected_candidate_comparative_category_uses_comparative_bank():
    injected, marker = build_injected_candidate(
        CANDIDATE, jd_id="backend-engineer", seed=42, category="comparative_injection",
    )
    assert marker in {p["marker_substring"] for p in COMPARATIVE_INJECTION_PARAPHRASES}
    assert marker.lower() in injected.raw_text.lower()


def test_build_injected_candidate_is_deterministic_for_same_seed_and_category():
    first, marker_first = build_injected_candidate(
        CANDIDATE, jd_id="backend-engineer", seed=7, category="comparative_injection",
    )
    second, marker_second = build_injected_candidate(
        CANDIDATE, jd_id="backend-engineer", seed=7, category="comparative_injection",
    )
    assert first.raw_text == second.raw_text
    assert marker_first == marker_second


def test_select_pair_subsample_keeps_first_n_per_profile_in_order():
    pairs = [
        ("backend-engineer", "cv-001"), ("backend-engineer", "cv-002"), ("backend-engineer", "cv-003"),
        ("backend-engineer", "cv-004"),
        ("cloud-engineer", "cv-101"), ("cloud-engineer", "cv-102"),
    ]
    subsample = select_pair_subsample(pairs, per_profile=3)
    assert subsample == [
        ("backend-engineer", "cv-001"), ("backend-engineer", "cv-002"), ("backend-engineer", "cv-003"),
        ("cloud-engineer", "cv-101"), ("cloud-engineer", "cv-102"),
    ]


def test_select_pair_subsample_is_a_strict_subset():
    pairs = [("a", "1"), ("a", "2"), ("a", "3")]
    subsample = select_pair_subsample(pairs, per_profile=2)
    assert all(pair in pairs for pair in subsample)
    assert len(subsample) == 2
