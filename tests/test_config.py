from __future__ import annotations

from pathlib import Path

from candidate_ranking.config import RunConfig, apply_env_overrides


def test_full_config_has_empty_jev_credentials_by_default():
    cfg = RunConfig.full(Path("/tmp/project"))
    assert cfg.jev_api_key == ""
    assert not hasattr(cfg, "tournament_iterations")
    assert not hasattr(cfg, "pl_prior_variance")


def test_env_overrides_apply_jev_credentials(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_JEV_API_KEY", "key-from-env")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.jev_api_key == "key-from-env"
