from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATHS = sorted((PROJECT_ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("script_path", SCRIPT_PATHS, ids=lambda p: p.name)
def test_script_imports_without_error(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
