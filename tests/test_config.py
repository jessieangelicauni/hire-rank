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


def test_full_config_has_sliding_window_defaults():
    cfg = RunConfig.full(Path("/tmp/project"))
    assert cfg.sliding_window_size == 15
    assert cfg.sliding_window_step == 7


def test_env_override_sliding_window_size(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_SLIDING_WINDOW_SIZE", "20")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.sliding_window_size == 20


def test_env_override_sliding_window_step(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_SLIDING_WINDOW_STEP", "3")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.sliding_window_step == 3


def test_full_config_has_ensemble_defaults():
    cfg = RunConfig.full(Path("/tmp/project"))
    assert cfg.ranking_ensemble_seeds == (42,)
    assert cfg.ranking_pl_prior_variance == 1.0


def test_env_override_ensemble_seeds(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_ENSEMBLE_SEEDS", "1,2,3")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_ensemble_seeds == (1, 2, 3)


def test_env_override_pl_prior_variance(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE", "2.0")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_pl_prior_variance == 2.0
