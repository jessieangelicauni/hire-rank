from __future__ import annotations

from pathlib import Path

from candidate_ranking.config import RunConfig, apply_env_overrides


def test_full_config_has_correct_defaults():
    cfg = RunConfig.full(Path("/tmp/project"))
    assert cfg.ranking_temperature == 0.4
    assert not hasattr(cfg, "tournament_iterations")
    assert not hasattr(cfg, "pl_prior_variance")
    assert not hasattr(cfg, "ranking_pool_threshold")


def test_env_overrides_apply_ranking_temperature(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_RANKING_TEMPERATURE", "0.8")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_temperature == 0.8


def test_full_config_has_active_ranking_defaults():
    cfg = RunConfig.full(Path("/tmp/project"))
    assert cfg.ranking_subset_size == 8
    assert cfg.ranking_max_iterations == 40
    assert cfg.ranking_min_delta_u == 0.01
    assert cfg.ranking_pl_prior_variance == 1.0
    assert not hasattr(cfg, "ranking_ensemble_seeds")


def test_env_override_ranking_subset_size(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_SUBSET_SIZE", "6")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_subset_size == 6


def test_env_override_ranking_max_iterations(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_MAX_ITERATIONS", "60")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_max_iterations == 60


def test_env_override_ranking_min_delta_u(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_MIN_DELTA_U", "0.005")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_min_delta_u == 0.005


def test_env_override_pl_prior_variance(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE", "2.0")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_pl_prior_variance == 2.0
