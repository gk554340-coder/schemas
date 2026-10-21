# config.py
"""Central configuration and workspace management for the Autonomous AI Cinema and Documentary Studio Engine."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


WORKSPACE_NAMES: Final[tuple[str, ...]] = (
    "01_research_vault",
    "02_screenplay_desk",
    "03_audio_studio",
    "04_visual_lab",
    "05_master_theater",
)

DEFAULT_WORKSPACE_DIRECTORY: Final[str] = "workspace"
DEFAULT_DEVICE: Final[str] = "cpu"
DEFAULT_SEED: Final[int] = 42
MIN_SEED: Final[int] = 0
MAX_SEED: Final[int] = 2**32 - 1


class StudioConfig(BaseModel):
    """Validated configuration for the studio engine."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
        str_strip_whitespace=True,
    )

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent,
        description="Absolute root directory containing the studio engine.",
    )
    workspace_root: Path | None = Field(
        default=None,
        description="Root directory for all isolated production workspaces.",
    )

    research_workspace: Path | None = Field(
        default=None,
        description="Workspace for research sources, facts, and research output.",
    )
    screenplay_workspace: Path | None = Field(
        default=None,
        description="Workspace for screenplay blueprints and scene planning.",
    )
    audio_workspace: Path | None = Field(
        default=None,
        description="Workspace for narration, audio segments, and audio timelines.",
    )
    visual_workspace: Path | None = Field(
        default=None,
        description="Workspace for visual prompts, generated assets, and visual metadata.",
    )
    master_workspace: Path | None = Field(
        default=None,
        description="Workspace for final assembly, exports, and delivery artifacts.",
    )

    test_mode: bool = Field(
        default=False,
        description="Run with deterministic, resource-light settings and no mandatory GPU.",
    )
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default=DEFAULT_DEVICE,
        description="Preferred compute device.",
    )
    seed: int = Field(
        default=DEFAULT_SEED,
        ge=MIN_SEED,
        le=MAX_SEED,
        description="Deterministic seed used by supported generation components.",
    )

    environment: Literal["development", "testing", "production"] = Field(
        default="development",
        description="Execution environment.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Application logging threshold.",
    )
    create_workspaces: bool = Field(
        default=True,
        description="Create missing workspace directories during initialization.",
    )
    restrict_workspaces_to_project: bool = Field(
        default=True,
        description="Prevent workspace directories from escaping project_root.",
    )
    allow_external_workspace_root: bool = Field(
        default=False,
        description="Explicitly permit workspace_root outside project_root when enabled.",
    )
    checkpoint_directory_name: str = Field(
        default="checkpoints",
        min_length=1,
        max_length=100,
        description="Directory name for persistent pipeline checkpoints.",
    )
    artifacts_directory_name: str = Field(
        default="artifacts",
        min_length=1,
        max_length=100,
        description="Directory name for generated artifacts.",
    )
    logs_directory_name: str = Field(
        default="logs",
        min_length=1,
        max_length=100,
        description="Directory name for runtime logs.",
    )

    @field_validator(
        "project_root",
        "workspace_root",
        "research_workspace",
        "screenplay_workspace",
        "audio_workspace",
        "visual_workspace",
        "master_workspace",
        mode="before",
    )
    @classmethod
    def normalize_paths(cls, value: object) -> object:
        """Expand user-home and environment-variable references in path values."""
        if value is None:
            return None

        if isinstance(value, Path):
            return value.expanduser()

        if isinstance(value, str):
            expanded = os.path.expandvars(value.strip())
            if not expanded:
                raise ValueError("Path values must not be empty.")
            return Path(expanded).expanduser()

        raise TypeError("Path values must be strings, pathlib.Path instances, or None.")

    @field_validator(
        "checkpoint_directory_name",
        "artifacts_directory_name",
        "logs_directory_name",
    )
    @classmethod
    def validate_directory_names(cls, value: str) -> str:
        """Reject directory names that could introduce path traversal."""
        candidate = Path(value)

        if candidate.is_absolute() or len(candidate.parts) != 1:
            raise ValueError("Directory names must be single relative path components.")

        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("Directory names must not contain path separators or traversal.")

        return value

    @model_validator(mode="after")
    def derive_and_validate_workspace_paths(self) -> StudioConfig:
        """Resolve default paths and validate workspace containment."""
        self.project_root = self.project_root.resolve(strict=False)

        if self.workspace_root is None:
            self.workspace_root = (
                self.project_root / DEFAULT_WORKSPACE_DIRECTORY
            ).resolve(strict=False)
        else:
            if not self.workspace_root.is_absolute():
                self.workspace_root = self.project_root / self.workspace_root
            self.workspace_root = self.workspace_root.resolve(strict=False)

        workspace_fields: tuple[tuple[str, str], ...] = (
            ("research_workspace", WORKSPACE_NAMES[0]),
            ("screenplay_workspace", WORKSPACE_NAMES[1]),
            ("audio_workspace", WORKSPACE_NAMES[2]),
            ("visual_workspace", WORKSPACE_NAMES[3]),
            ("master_workspace", WORKSPACE_NAMES[4]),
        )

        for field_name, directory_name in workspace_fields:
            configured_path = getattr(self, field_name)

            if configured_path is None:
                configured_path = self.workspace_root / directory_name
            elif not configured_path.is_absolute():
                configured_path = self.workspace_root / configured_path

            resolved_path = configured_path.resolve(strict=False)
            setattr(self, field_name, resolved_path)

        if self.restrict_workspaces_to_project and not self.allow_external_workspace_root:
            self._validate_path_containment(
                self.workspace_root,
                self.project_root,
                "workspace_root",
            )

        if self.restrict_workspaces_to_project:
            for field_name, _ in workspace_fields:
                workspace_path = getattr(self, field_name)
                self._validate_path_containment(
                    workspace_path,
                    self.workspace_root,
                    field_name,
                )

        if self.test_mode and self.device in {"cuda", "mps"}:
            raise ValueError(
                "test_mode requires device='cpu' or device='auto'. "
                "Use a separate non-test configuration for accelerator execution."
            )

        return self

    @staticmethod
    def _validate_path_containment(
        candidate: Path,
        allowed_root: Path,
        field_name: str,
    ) -> None:
        """Ensure candidate is inside allowed_root after resolving path components."""
        resolved_candidate = candidate.resolve(strict=False)
        resolved_root = allowed_root.resolve(strict=False)

        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} resolves outside its permitted root: "
                f"{resolved_candidate}"
            ) from exc


def load_config(
    config_path: Path | str | None = None,
    *,
    environ_prefix: str = "STUDIO_",
    overrides: dict[str, object] | None = None,
) -> StudioConfig:
    """Load configuration from environment variables, an optional JSON file, and overrides.

    Precedence, from lowest to highest:
        1. StudioConfig defaults
        2. Optional JSON configuration file
        3. Environment variables with the configured prefix
        4. Explicit overrides

    Args:
        config_path: Optional path to a JSON configuration file.
        environ_prefix: Prefix used to identify environment configuration variables.
        overrides: Explicit configuration values that take highest precedence.

    Returns:
        A validated StudioConfig instance.

    Raises:
        FileNotFoundError: If config_path points to a missing file.
        ValueError: If the JSON file is malformed or the configuration is invalid.
        OSError: If the configuration file cannot be read.
    """
    import json

    values: dict[str, object] = {}

    if config_path is not None:
        path = Path(config_path).expanduser().resolve(strict=False)

        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as config_file:
                loaded = json.load(config_file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON configuration in {path} at line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}"
            ) from exc

        if not isinstance(loaded, dict):
            raise ValueError("The configuration JSON root must be an object.")

        values.update(loaded)

    for field_name, field_info in StudioConfig.model_fields.items():
        environment_name = f"{environ_prefix}{field_name.upper()}"

        if environment_name not in os.environ:
            continue

        raw_value = os.environ[environment_name]
        annotation = field_info.annotation

        if annotation is bool:
            normalized = raw_value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                parsed_value: object = True
            elif normalized in {"0", "false", "no", "off"}:
                parsed_value = False
            else:
                raise ValueError(
                    f"Environment variable {environment_name} must be a boolean."
                )
        elif annotation is int:
            try:
                parsed_value = int(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Environment variable {environment_name} must be an integer."
                ) from exc
        else:
            parsed_value = raw_value

        values[field_name] = parsed_value

    if overrides:
        values.update(overrides)

    try:
        return StudioConfig.model_validate(values)
    except ValidationError as exc:
        raise ValueError(f"Studio configuration validation failed:\n{exc}") from exc


def validate_config(config: StudioConfig) -> StudioConfig:
    """Revalidate a configuration and enforce filesystem safety constraints.

    Args:
        config: Configuration object to validate.

    Returns:
        The validated configuration.

    Raises:
        ValueError: If configuration values or filesystem paths are invalid.
    """
    validated = StudioConfig.model_validate(config.model_dump())

    if validated.restrict_workspaces_to_project and not validated.allow_external_workspace_root:
        StudioConfig._validate_path_containment(
            validated.workspace_root,
            validated.project_root,
            "workspace_root",
        )

    workspace_paths: tuple[tuple[str, Path], ...] = (
        ("research_workspace", validated.research_workspace),
        ("screenplay_workspace", validated.screenplay_workspace),
        ("audio_workspace", validated.audio_workspace),
        ("visual_workspace", validated.visual_workspace),
        ("master_workspace", validated.master_workspace),
    )

    resolved_paths: set[Path] = set()

    for field_name, path in workspace_paths:
        resolved = path.resolve(strict=False)

        if resolved in resolved_paths:
            raise ValueError(
                f"Workspace paths must be distinct; duplicate path found in {field_name}: "
                f"{resolved}"
            )

        resolved_paths.add(resolved)

        if validated.restrict_workspaces_to_project:
            StudioConfig._validate_path_containment(
                resolved,
                validated.workspace_root,
                field_name,
            )

        if resolved.exists() and not resolved.is_dir():
            raise ValueError(
                f"{field_name} exists but is not a directory: {resolved}"
            )

    if validated.device == "auto" and validated.test_mode:
        validated.device = "cpu"

    return validated


def initialize_workspaces(config: StudioConfig) -> dict[str, Path]:
    """Create and verify the configured workspace directories.

    The function creates only the workspace root and its five direct child
    workspaces. It does not delete, overwrite, or clean existing user data.

    Args:
        config: Validated studio configuration.

    Returns:
        A mapping from workspace role to its resolved directory path.

    Raises:
        ValueError: If a workspace path violates containment rules.
        FileExistsError: If a required directory path is occupied by a file.
        OSError: If the operating system cannot create or inspect a directory.
    """
    validated = validate_config(config)

    workspaces: dict[str, Path] = {
        "research": validated.research_workspace,
        "screenplay": validated.screenplay_workspace,
        "audio": validated.audio_workspace,
        "visual": validated.visual_workspace,
        "master": validated.master_workspace,
    }

    directories_to_create = (
        validated.workspace_root,
        *workspaces.values(),
    )

    if validated.create_workspaces:
        for directory in directories_to_create:
            resolved_directory = directory.resolve(strict=False)

            if validated.restrict_workspaces_to_project:
                StudioConfig._validate_path_containment(
                    resolved_directory,
                    validated.project_root
                    if directory == validated.workspace_root
                    else validated.workspace_root,
                    "workspace_directory",
                )

            if resolved_directory.exists() and not resolved_directory.is_dir():
                raise FileExistsError(
                    f"Workspace path exists but is not a directory: {resolved_directory}"
                )

            resolved_directory.mkdir(parents=True, exist_ok=True)

            if not resolved_directory.is_dir():
                raise OSError(
                    f"Workspace initialization failed: {resolved_directory}"
                )

    else:
        for directory in directories_to_create:
            resolved_directory = directory.resolve(strict=False)

            if not resolved_directory.exists():
                raise FileNotFoundError(
                    f"Workspace creation is disabled and directory does not exist: "
                    f"{resolved_directory}"
                )

            if not resolved_directory.is_dir():
                raise NotADirectoryError(
                    f"Expected a workspace directory: {resolved_directory}"
                )

    return {
        name: path.resolve(strict=True)
        for name, path in workspaces.items()
    }


def resolve_workspace_path(
    config: StudioConfig,
    workspace: str,
    relative_path: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a workspace-relative path while preventing traversal and symlink escapes.

    Args:
        config: Validated studio configuration.
        workspace: Workspace role: research, screenplay, audio, visual, or master.
        relative_path: Relative path within the selected workspace.
        must_exist: If True, require the resolved target to exist.

    Returns:
        An absolute, normalized path contained within the selected workspace.

    Raises:
        ValueError: If the workspace name is unknown or the path escapes the workspace.
        FileNotFoundError: If must_exist is True and the target does not exist.
        NotADirectoryError: If the selected workspace does not exist or is not a directory.
    """
    validated = validate_config(config)

    workspace_map: dict[str, Path] = {
        "research": validated.research_workspace,
        "screenplay": validated.screenplay_workspace,
        "audio": validated.audio_workspace,
        "visual": validated.visual_workspace,
        "master": validated.master_workspace,
    }

    if workspace not in workspace_map:
        raise ValueError(
            f"Unknown workspace '{workspace}'. Expected one of: "
            f"{', '.join(workspace_map)}."
        )

    if isinstance(relative_path, str):
        if not relative_path.strip():
            raise ValueError("relative_path must not be empty.")

        normalized_input = relative_path.replace("\\", "/")
        candidate_input = Path(normalized_input)
    elif isinstance(relative_path, Path):
        candidate_input = relative_path
    else:
        raise TypeError("relative_path must be a string or pathlib.Path.")

    if candidate_input.is_absolute():
        raise ValueError("Absolute paths are not permitted for workspace-relative paths.")

    if any(part == ".." for part in candidate_input.parts):
        raise ValueError("Path traversal using '..' is not permitted.")

    if candidate_input == Path("."):
        raise ValueError("relative_path must identify a path inside the workspace.")

    workspace_root = workspace_map[workspace].resolve(strict=False)

    if not workspace_root.exists():
        raise NotADirectoryError(
            f"Workspace directory does not exist: {workspace_root}"
        )

    if not workspace_root.is_dir():
        raise NotADirectoryError(
            f"Workspace path is not a directory: {workspace_root}"
        )

    target = (workspace_root / candidate_input).resolve(strict=False)

    try:
        target.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(
            f"Resolved path escapes the '{workspace}' workspace: {target}"
        ) from exc

    if must_exist and not target.exists():
        raise FileNotFoundError(f"Required workspace path does not exist: {target}")

    return target


def generate_secure_run_id() -> str:
    """Generate a cryptographically random identifier for temporary run resources."""
    return secrets.token_hex(16)
