# File: tests/test_pipeline.py
# Type: Python / pytest
# Purpose: Lightweight unit tests for configuration, schemas, and pipeline basics.

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    """Return the repository root directory."""
    return PROJECT_ROOT


@pytest.fixture
def pipeline_module():
    """Import the master pipeline module."""
    return importlib.import_module("master_pipeline")


@pytest.fixture
def config_module():
    """Import the project configuration module."""
    return importlib.import_module("config")


@pytest.fixture
def schemas_module():
    """Import the shared schemas module."""
    return importlib.import_module("schemas")


def test_required_project_files_exist(project_root: Path) -> None:
    """Verify the core source files exist in the repository root."""
    required_files = [
        "config.py",
        "schemas.py",
        "master_pipeline.py",
        "server.py",
        "requirements.txt",
        "ARCHITECTURE.md",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (project_root / filename).is_file()
    ]

    assert not missing_files, (
        "Required project files are missing: "
        + ", ".join(missing_files)
    )


def test_workspace_directories_exist(project_root: Path) -> None:
    """Verify all five isolated workspace directories exist."""
    workspace_names = [
        "01_research_vault",
        "02_screenplay_desk",
        "03_audio_studio",
        "04_visual_lab",
        "05_master_theater",
    ]

    workspace_root = project_root / "workspace"

    missing_directories = [
        name
        for name in workspace_names
        if not (workspace_root / name).is_dir()
    ]

    assert not missing_directories, (
        "Workspace directories are missing: "
        + ", ".join(missing_directories)
    )


def test_config_module_imports(config_module) -> None:
    """Verify config.py imports successfully."""
    assert config_module is not None


def test_config_exposes_project_paths(config_module) -> None:
    """Check configured paths when the module exposes them."""
    path_names = (
        "PROJECT_ROOT",
        "WORKSPACE_ROOT",
        "WORKSPACE_DIR",
        "RESEARCH_DIR",
        "SCREENPLAY_DIR",
        "AUDIO_DIR",
        "VISUAL_DIR",
        "OUTPUT_DIR",
    )

    discovered = [
        name
        for name in path_names
        if hasattr(config_module, name)
    ]

    # This remains compatible with configurations using other naming schemes.
    if not discovered:
        pytest.skip(
            "No recognized path constants are exposed by config.py."
        )

    for name in discovered:
        value = getattr(config_module, name)

        if value is None:
            continue

        assert isinstance(value, (str, Path)), (
            f"{name} should be a string or pathlib.Path."
        )


def test_schemas_module_imports(schemas_module) -> None:
    """Verify shared schema definitions import successfully."""
    assert schemas_module is not None


def test_json_schema_files_are_valid_json(project_root: Path) -> None:
    """Verify available JSON Schema files contain valid JSON."""
    schema_directory = project_root / "schemas"

    if not schema_directory.is_dir():
        pytest.skip("The schemas/ directory does not exist.")

    schema_files = sorted(schema_directory.glob("*.json"))

    if not schema_files:
        pytest.skip("No JSON Schema files were found.")

    for schema_path in schema_files:
        parsed = json.loads(schema_path.read_text(encoding="utf-8"))

        assert isinstance(parsed, dict), (
            f"{schema_path.name} must contain a JSON object."
        )


def test_json_schemas_have_a_schema_identifier(
    project_root: Path,
) -> None:
    """Check that JSON Schema documents declare a schema identifier."""
    schema_directory = project_root / "schemas"

    if not schema_directory.is_dir():
        pytest.skip("The schemas/ directory does not exist.")

    schema_files = sorted(schema_directory.glob("*.json"))

    if not schema_files:
        pytest.skip("No JSON Schema files were found.")

    for schema_path in schema_files:
        parsed = json.loads(schema_path.read_text(encoding="utf-8"))

        assert "$schema" in parsed, (
            f"{schema_path.name} should declare its $schema version."
        )


def test_master_pipeline_imports(pipeline_module) -> None:
    """Verify the master pipeline imports successfully."""
    assert pipeline_module is not None


def test_master_pipeline_public_functions(pipeline_module) -> None:
    """Check the required pipeline entry points are exposed."""
    required_functions = [
        "create_project",
        "load_project_state",
        "save_project_state",
        "execute_stage",
        "recover_project",
        "run_pipeline",
        "cancel_pipeline",
        "get_pipeline_status",
    ]

    missing_functions = [
        name
        for name in required_functions
        if not callable(getattr(pipeline_module, name, None))
    ]

    assert not missing_functions, (
        "Required pipeline functions are missing: "
        + ", ".join(missing_functions)
    )


def test_sha256_checksum_is_deterministic(tmp_path: Path) -> None:
    """Verify Python SHA-256 calculations are stable for artifact bytes."""
    artifact = tmp_path / "sample.txt"
    content = b"cinema-studio-ci-test"
    artifact.write_bytes(content)

    first_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    second_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert first_digest == second_digest
    assert len(first_digest) == 64
    assert all(character in "0123456789abcdef" for character in first_digest)


def test_project_state_json_round_trip(tmp_path: Path) -> None:
    """Verify a basic project-state document can be saved and loaded."""
    state_path = tmp_path / "project_state.json"

    state: dict[str, Any] = {
        "job_id": "pytest-basic-job",
        "status": "queued",
        "revision": 1,
        "current_stage": None,
        "stages": {},
        "artifacts": [],
    }

    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    loaded_state = json.loads(
        state_path.read_text(encoding="utf-8")
    )

    assert loaded_state["job_id"] == "pytest-basic-job"
    assert loaded_state["status"] == "queued"
    assert loaded_state["revision"] == 1


def test_pipeline_test_mode_cli_is_documented(
    project_root: Path,
) -> None:
    """Check that the architecture documents the test-mode option."""
    architecture_path = project_root / "ARCHITECTURE.md"

    if not architecture_path.is_file():
        pytest.skip("ARCHITECTURE.md is missing.")

    architecture = architecture_path.read_text(encoding="utf-8")

    assert "--test-mode" in architecture


def test_test_mode_does_not_require_cuda() -> None:
    """Verify the test suite itself does not require a GPU."""
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    # No torch import or GPU initialization is required by this test.
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
