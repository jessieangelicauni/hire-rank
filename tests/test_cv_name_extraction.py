
from pathlib import Path

from candidate_ranking.ingestion.cv import _ExtractedName, load_or_extract_candidate_name
from candidate_ranking.models import Candidate

CANDIDATE = Candidate(
    id="cv-001", source_path="/cv/1.pdf", raw_text="irrelevant", num_pages=1, char_count=9, parse_status="ok",
)


class _FakeNameChain:
    def __init__(self, name: str | None):
        self._name = name

    def invoke(self, payload):
        return _ExtractedName(reasoning="fake reasoning", name=self._name)


def test_all_caps_name_is_title_cased(tmp_path: Path):
    name = load_or_extract_candidate_name(
        CANDIDATE, _FakeNameChain("DANIEL ADIF NUGROHO"), "fake-model", tmp_path / "cv_names.json"
    )
    assert name == "Daniel Adif Nugroho"


def test_already_correctly_cased_name_is_untouched(tmp_path: Path):
    name = load_or_extract_candidate_name(
        CANDIDATE, _FakeNameChain("John Rivera"), "fake-model", tmp_path / "cv_names.json"
    )
    assert name == "John Rivera"


def test_mixed_internal_capitals_are_left_alone_since_not_all_uppercase(tmp_path: Path):
    name = load_or_extract_candidate_name(
        CANDIDATE, _FakeNameChain("Ryan McDonald"), "fake-model", tmp_path / "cv_names.json"
    )
    assert name == "Ryan McDonald"


def test_missing_name_stays_none(tmp_path: Path):
    name = load_or_extract_candidate_name(CANDIDATE, _FakeNameChain(None), "fake-model", tmp_path / "cv_names.json")
    assert name is None


def test_normalization_applies_on_cache_hit_too(tmp_path: Path):
    cache_path = tmp_path / "cv_names.json"
    first = load_or_extract_candidate_name(CANDIDATE, _FakeNameChain("DANIEL ADIF NUGROHO"), "fake-model", cache_path)
    class _ExplodingChain:
        def invoke(self, payload):
            raise AssertionError("should not be called on a cache hit")

    second = load_or_extract_candidate_name(CANDIDATE, _ExplodingChain(), "fake-model", cache_path)
    assert first == second == "Daniel Adif Nugroho"
