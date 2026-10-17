from __future__ import annotations

from pathlib import Path

from candidate_ranking.config import RunConfig, apply_env_overrides


def test_full_config_has_active_ranking_defaults():
    cfg = RunConfig.full(Path("/tmp/project"))
    assert cfg.ranking_min_delta_u == 0.01
    assert cfg.ranking_pl_prior_variance == 1.0
    assert not hasattr(cfg, "ranking_ensemble_seeds")


def test_env_override_ranking_min_delta_u(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_MIN_DELTA_U", "0.005")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_min_delta_u == 0.005


def test_env_override_pl_prior_variance(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE", "2.0")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_pl_prior_variance == 2.0


def test_jev_api_key_override():
    import os
    from candidate_ranking.config import RunConfig, apply_env_overrides
    from pathlib import Path
    cfg = RunConfig.full(Path("."))
    assert cfg.jev_api_key is None
    os.environ["CANDIDATE_RANKING_JEV_API_KEY"] = "tok-abc"
    try:
        cfg2 = apply_env_overrides(cfg)
        assert cfg2.jev_api_key == "tok-abc"
    finally:
        del os.environ["CANDIDATE_RANKING_JEV_API_KEY"]


def test_pairwise_max_iterations_default():
    from candidate_ranking.config import RunConfig
    from pathlib import Path
    cfg = RunConfig.full(Path("."))
    assert cfg.ranking_pairwise_max_iterations == 80
