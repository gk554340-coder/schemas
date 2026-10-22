
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import inspect
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(
    os.getenv("CINEMA_WORKSPACE_DIR", str(ROOT_DIR / "workspace"))
).expanduser().resolve()

DEFAULT_OUTPUT_ROOT = WORKSPACE_DIR / "05_master_theater"
STATE_FILENAME = "project_state.json"
STATE_VERSION = "2.0.0"

PROFILES = {
    "reel": {
        "duration_seconds": 15,
        "aspect_ratio": "9:16",
        "width": 720,
        "height": 1280,
        "screenplay_profile": "short_form",
    },
    "documentary": {
        "duration_seconds": 60,
        "aspect_ratio": "16:9",
        "width": 1280,
        "height": 720,
        "screenplay_profile": "long",
    },
}

DEFAULT_STAGE_ORDER = [
    "slot1_research",
    "slot2_screenplay",
    "slot4_audio",
    "slot3_visuals",
    "slot5_editor",
]

STAGE_TIMEOUT_SECONDS = {
    "slot1_research": 300,
    "slot2_screenplay": 300,
    "slot4_audio": 600,
    "slot3_visuals": 900,
    "slot5_editor": 600,
}

STAGE_DIRECTORIES = {
    "slot1_research": "01_research_vault",
    "slot2_screenplay": "02_screenplay_desk",
    "slot4_audio": "03_audio_studio",
    "slot3_visuals": "04_visual_lab",
    "slot5_editor": "05_master_theater",
}

STAGE_MODULES = {
    "slot1_research": ("slot1_research", "run_research"),
    "slot2_screenplay": ("slot2_screenplay", "run_screenplay"),
    "slot4_audio": ("slot4_audio", "run_audio"),
    "slot3_visuals": ("slot3_visuals", "run_visuals"),
    "slot5_editor": ("slot5_editor", "run_editor"),
}

TERMINAL_STATES = {"completed", "failed", "cancelled"}
VALID_STATES = {
    "created", "queued", "running", "cancelling",
    "completed", "failed", "cancelled",
}

logger = logging.getLogger("cinema_pipeline")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

_ACTIVE_JOBS: dict[str, dict[str, Any]] = {}
_ACTIVE_JOBS_LOCK = threading.RLock()
_CANCEL_EVENTS: dict[str, threading.Event] = {}
_CANCEL_LOCK = threading.RLock()
_PROJECT_LOCKS: dict[str, threading.RLock] = {}
_PROJECT_LOCKS_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Exceptions and utilities
# ---------------------------------------------------------------------------

class PipelineError(RuntimeError):
    """Base pipeline exception."""


class PipelineCancelled(PipelineError):
    """Raised when cooperative cancellation is requested."""


class StateIntegrityError(PipelineError):
    """Raised when project state is invalid or corrupted."""


class StageExecutionError(PipelineError):
    """Raised when a production stage fails."""


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert values into JSON-compatible data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically to avoid partially written state files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                json_safe(data),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise StateIntegrityError(f"Expected a JSON object: {path}")
    return data


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    """Resolve a filesystem path."""
    return Path(value).expanduser().resolve()


def is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is contained in a trusted directory."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def project_lock(state_path: Path) -> threading.RLock:
    """Return a process-local lock for one project state file."""
    key = str(state_path.resolve())
    with _PROJECT_LOCKS_LOCK:
        return _PROJECT_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def locked_project(state_path: Path):
    """Serialize state updates for one project."""
    lock = project_lock(state_path)
    with lock:
        yield


def configured_stage_order() -> list[str]:
    """Load an optional stage order from environment configuration."""
    raw = os.getenv("CINEMA_STAGE_ORDER")
    if not raw:
        return list(DEFAULT_STAGE_ORDER)

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PipelineError("CINEMA_STAGE_ORDER must contain a JSON array.") from exc

    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise PipelineError("CINEMA_STAGE_ORDER must be a non-empty string array.")

    if len(value) != len(set(value)):
        raise PipelineError("CINEMA_STAGE_ORDER cannot contain duplicate stages.")

    for stage in value:
        if stage not in STAGE_MODULES:
            raise PipelineError(f"Unknown configured stage: {stage}")

    return value


def _project_workspace(state: dict[str, Any]) -> Path:
    """Resolve and validate the workspace assigned to a project."""
    workspace = resolve_path(state["project"]["workspace_dir"])
    allowed_root = resolve_path(state["project"]["output_root"])
    if not is_within(workspace, allowed_root):
        raise StateIntegrityError("Project workspace escapes its output root.")
    return workspace


def _get_state_path(state: dict[str, Any]) -> Path:
    """Return the state file path stored in a project record."""
    path = resolve_path(state["state_path"])
    workspace = _project_workspace(state)
    if not is_within(path, workspace):
        raise StateIntegrityError("Project state path escapes its workspace.")
    return path


def _default_stage_record() -> dict[str, Any]:
    """Create a consistent initial stage record."""
    return {
        "status": "pending",
        "started_at": None,
        "completed_at": None,
        "duration_seconds": None,
        "attempts": 0,
        "artifacts": [],
        "error": None,
    }


def _default_project_state(
    project_id: str,
    topic: str,
    language: str,
    profile: str,
    workspace: Path,
    output_root: Path,
    test_mode: bool,
    stage_order: list[str],
) -> dict[str, Any]:
    """Create the canonical project state."""
    now = utc_now()
    return {
        "state_version": STATE_VERSION,
        "project_id": project_id,
        "job_id": project_id,
        "revision": 0,
        "status": "created",
        "created_at": now,
        "updated_at": now,
        "state_path": str(workspace / STATE_FILENAME),
        "project": {
            "topic": topic,
            "language": language,
            "profile": profile,
            "profile_settings": PROFILES[profile].copy(),
            "workspace_dir": str(workspace),
            "output_root": str(output_root),
            "test_mode": bool(test_mode),
        },
        "pipeline": {
            "stage_order": list(stage_order),
            "current_stage": None,
            "completed_stages": [],
            "failed_stage": None,
            "cancel_reason": None,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": None,
            "last_error": None,
        },
        "stages": {
            stage: _default_stage_record()
            for stage in stage_order
        },
        "artifacts": {},
        "integrity": {
            "algorithm": "sha256",
            "state_revision": 0,
            "last_verified_at": None,
        },
        "recovery": {
            "recovered_at": None,
            "invalidated_stages": [],
        },
    }


def _validate_state_shape(state: dict[str, Any]) -> None:
    """Validate required state fields before using project data."""
    required = {
        "project_id", "job_id", "status", "project",
        "pipeline", "stages", "state_path",
    }
    missing = required.difference(state)
    if missing:
        raise StateIntegrityError(
            f"Project state is missing required fields: {sorted(missing)}"
        )

    if state["status"] not in VALID_STATES:
        raise StateIntegrityError(
            f"Unsupported project status: {state['status']}"
        )

    if not isinstance(state["project"], dict):
        raise StateIntegrityError("Project configuration must be an object.")
    if not isinstance(state["pipeline"], dict):
        raise StateIntegrityError("Pipeline state must be an object.")
    if not isinstance(state["stages"], dict):
        raise StateIntegrityError("Stages state must be an object.")


def save_project_state(state: dict[str, Any]) -> dict[str, Any]:
    """Update revision metadata and atomically persist project state."""
    _validate_state_shape(state)
    state_path = _get_state_path(state)

    with locked_project(state_path):
        state["updated_at"] = utc_now()
        state["revision"] = int(state.get("revision", 0)) + 1
        state.setdefault("integrity", {})
        state["integrity"]["algorithm"] = "sha256"
        state["integrity"]["state_revision"] = state["revision"]
        atomic_write_json(state_path, state)

    return state


def load_project_state(state_path: str | Path) -> dict[str, Any]:
    """Load and validate a project's persistent state."""
    path = resolve_path(state_path)
    state = read_json(path)
    _validate_state_shape(state)

    if resolve_path(state["state_path"]) != path:
        raise StateIntegrityError("Recorded state path does not match loaded file.")

    _project_workspace(state)
    state.setdefault("artifacts", {})
    state.setdefault("integrity", {})
    state.setdefault("recovery", {})
    return state


def _register_job(job_id: str, state_path: Path) -> None:
    """Register a pipeline job for API and CLI access."""
    with _ACTIVE_JOBS_LOCK:
        _ACTIVE_JOBS[job_id] = {"state_path": str(state_path)}
    with _CANCEL_LOCK:
        _CANCEL_EVENTS.setdefault(job_id, threading.Event())


def _cancel_event(job_id: str) -> threading.Event:
    """Return or create a cancellation event for a job."""
    with _CANCEL_LOCK:
        return _CANCEL_EVENTS.setdefault(job_id, threading.Event())


def _check_cancelled(job_id: str) -> None:
    """Raise when the job has been cancelled."""
    if _cancel_event(job_id).is_set():
        raise PipelineCancelled(
            f"Cancellation requested for pipeline job {job_id}."
        )


# ---------------------------------------------------------------------------
# Project creation and artifact handling
# ---------------------------------------------------------------------------

def create_project(
    topic: str,
    language: str = "Hindi",
    profile: str = "reel",
    output_root: str | Path | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Create an isolated project and register it immediately."""
    topic = str(topic).strip()
    language = str(language).strip()
    profile = str(profile).strip().lower()

    if not topic:
        raise ValueError("topic must not be empty.")
    if not language:
        raise ValueError("language must not be empty.")
    if profile not in PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    root = resolve_path(output_root or DEFAULT_OUTPUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    project_id = f"cinema-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:10]}"
    workspace = root / "projects" / project_id
    workspace.mkdir(parents=True, exist_ok=False)

    for stage in configured_stage_order():
        directory = STAGE_DIRECTORIES.get(stage, f"stages/{stage}")
        (workspace / directory).mkdir(parents=True, exist_ok=True)

    state = _default_project_state(
        project_id=project_id,
        topic=topic,
        language=language,
        profile=profile,
        workspace=workspace,
        output_root=root,
        test_mode=test_mode,
        stage_order=configured_stage_order(),
    )
    state_path = workspace / STATE_FILENAME
    atomic_write_json(state_path, state)
    _register_job(project_id, state_path)

    logger.info("Created project %s at %s", project_id, workspace)
    return state


def _artifact_record(path: Path, workspace: Path) -> dict[str, Any]:
    """Create a checksum record for a workspace artifact."""
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not is_within(resolved, workspace):
        raise StateIntegrityError(f"Artifact is outside project workspace: {path}")

    stat = resolved.stat()
    return {
        "path": str(resolved),
        "name": resolved.name,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
    }


def _verify_artifact_record(
    record: dict[str, Any],
    workspace: Path,
) -> bool:
    """Verify an artifact's path, size and checksum."""
    try:
        path = resolve_path(record["path"])
        if not is_within(path, workspace) or not path.is_file():
            return False
        if path.stat().st_size != int(record["size_bytes"]):
            return False
        return sha256_file(path) == record["sha256"]
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _write_json_artifact(path: Path, data: dict[str, Any]) -> Path:
    """Write a stage JSON output atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data)
    return path.resolve()


def _stage_workspace(state: dict[str, Any], stage: str) -> Path:
    """Resolve the output directory for one stage."""
    workspace = _project_workspace(state)
    directory = STAGE_DIRECTORIES.get(stage, f"stages/{stage}")
    path = (workspace / directory).resolve()
    if not is_within(path, workspace):
        raise StateIntegrityError(f"Unsafe stage directory: {stage}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_artifacts(state: dict[str, Any], stage: str) -> list[Path]:
    """Find current files produced in a stage directory."""
    directory = _stage_workspace(state, stage)
    results = []
    for path in directory.rglob("*"):
        if path.is_file() and path.name != STATE_FILENAME:
            if is_within(path, _project_workspace(state)):
                results.append(path.resolve())
    return sorted(results, key=lambda item: item.stat().st_mtime)


def _locate_latest_artifact(
    state: dict[str, Any],
    stages: list[str],
    patterns: tuple[str, ...],
    suffixes: tuple[str, ...] = (".json",),
) -> Path | None:
    """Find the latest matching artifact from preceding stages."""
    matches: list[Path] = []
    for stage in stages:
        for path in _find_artifacts(state, stage):
            if path.suffix.lower() not in suffixes:
                continue
            name = path.name.lower()
            if any(pattern in name for pattern in patterns):
                matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda item: item.stat().st_mtime)


def _invoke_supported(function: Callable[..., Any], values: dict[str, Any]) -> Any:
    """Call a slot using only accepted keyword arguments."""
    signature = inspect.signature(function)
    parameters = signature.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return function(**values)

    kwargs = {
        name: value
        for name, value in values.items()
        if name in parameters
        and parameters[name].kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }

    required = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        and name not in kwargs
    ]
    if required:
        raise StageExecutionError(
            f"{function.__module__}.{function.__name__} requires unsupported "
            f"arguments: {required}"
        )

    return function(**kwargs)


# ---------------------------------------------------------------------------
# Offline test mode
# ---------------------------------------------------------------------------

def _run_test_stage(
    state: dict[str, Any],
    stage: str,
    output_dir: Path,
) -> list[Path]:
    """Produce deterministic offline fixtures for one stage."""
    project = state["project"]
    profile = project["profile_settings"]
    topic = project["topic"]
    language = project["language"]
    previous = state["pipeline"]["completed_stages"]

    output_dir.mkdir(parents=True, exist_ok=True)

    if stage == "slot1_research":
        path = output_dir / "research_output.json"
        _write_json_artifact(path, {
            "topic": topic,
            "language": language,
            "test_mode": True,
            "sources": [],
            "summary": f"Offline research fixture for: {topic}",
            "created_at": utc_now(),
        })
        return [path]

    if stage == "slot2_screenplay":
        path = output_dir / "screenplay_blueprint.json"
        _write_json_artifact(path, {
            "topic": topic,
            "language": language,
            "profile": project["profile"],
            "target_duration_seconds": profile["duration_seconds"],
            "scenes": [
                {
                    "scene_id": "scene-001",
                    "title": topic,
                    "description": f"Test scene for {topic}",
                    "duration_seconds": profile["duration_seconds"],
                }
            ],
            "test_mode": True,
            "upstream_stages": previous,
        })
        return [path]

    if stage == "slot4_audio":
        path = output_dir / "audio_timeline.json"
        _write_json_artifact(path, {
            "topic": topic,
            "language": language,
            "duration_seconds": profile["duration_seconds"],
            "voice": "default",
            "segments": [],
            "test_mode": True,
        })
        return [path]

    if stage == "slot3_visuals":
        frame = output_dir / "test_scene_001.ppm"
        width, height = 64, 64
        frame.write_bytes(
            f"P6\n{width} {height}\n255\n".encode("ascii")
            + bytes([24, 32, 64]) * width * height
        )
        manifest = output_dir / "visual_manifest.json"
        _write_json_artifact(manifest, {
            "topic": topic,
            "language": language,
            "frames": [str(frame.resolve())],
            "test_mode": True,
        })
        return [manifest, frame]

    if stage == "slot5_editor":
        path = output_dir / "test_export_spec.json"
        _write_json_artifact(path, {
            "topic": topic,
            "profile": project["profile"],
            "width": profile["width"],
            "height": profile["height"],
            "duration_seconds": profile["duration_seconds"],
            "test_mode": True,
            "export_pending": True,
        })
        return [path]

    raise StageExecutionError(f"No test fixture is defined for stage: {stage}")


def _create_test_video(state: dict[str, Any]) -> Path:
    """Create and verify a small MP4 test render using FFmpeg."""
    workspace = _project_workspace(state)
    settings = state["project"]["profile_settings"]
    output_dir = _stage_workspace(state, "slot5_editor")
    output_path = output_dir / "cinema_test_export.mp4"

    ffmpeg = shutil.which(os.getenv("CINEMA_FFMPEG", "ffmpeg"))
    ffprobe = shutil.which(os.getenv("CINEMA_FFPROBE", "ffprobe"))

    if not ffmpeg or not ffprobe:
        raise StageExecutionError(
            "Test video export requires both FFmpeg and FFprobe."
        )

    width = int(settings["width"])
    height = int(settings["height"])
    duration = max(1, min(int(settings["duration_seconds"]), 5))

    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x182040:s={width}x{height}:d={duration}:r=24",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=max(60, duration * 20),
        check=False,
    )
    if result.returncode != 0:
        raise StageExecutionError(
            f"FFmpeg test export failed: {result.stderr[-2000:]}"
        )

    probe = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,width,height",
            "-of", "json", str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        raise StageExecutionError("FFprobe could not validate the test video.")

    try:
        metadata = json.loads(probe.stdout)
        if not metadata.get("streams") or not metadata.get("format"):
            raise ValueError("Missing stream or format metadata.")
    except (json.JSONDecodeError, ValueError) as exc:
        raise StageExecutionError("Test video metadata is invalid.") from exc

    if not is_within(output_path, workspace) or output_path.stat().st_size == 0:
        raise StateIntegrityError("Test export failed workspace validation.")

    return output_path.resolve()


# ---------------------------------------------------------------------------
# Production slot dispatch
# ---------------------------------------------------------------------------

def _resolve_stage_inputs(
    state: dict[str, Any],
    stage: str,
) -> dict[str, Path | None]:
    """Resolve the inputs required by a stage from completed predecessors."""
    completed = state["pipeline"]["completed_stages"]

    research = _locate_latest_artifact(
        state, completed, ("research_output", "research"),
    )
    screenplay = _locate_latest_artifact(
        state, completed, ("screenplay_blueprint", "screenplay"),
    )
    timeline = _locate_latest_artifact(
        state, completed, ("audio_timeline", "timeline"),
    )
    visuals = _locate_latest_artifact(
        state, completed, ("visual_manifest", "visual"),
    )

    return {
        "research": research,
        "screenplay": screenplay,
        "timeline": timeline,
        "visual_manifest": visuals,
    }


def _run_production_stage(
    state: dict[str, Any],
    stage: str,
    output_dir: Path,
) -> list[Path]:
    """Invoke a production slot with its supported arguments."""
    module_name, entrypoint_name = STAGE_MODULES[stage]
    module = importlib.import_module(module_name)
    function = getattr(module, entrypoint_name, None)

    if not callable(function):
        raise StageExecutionError(
            f"Entry point {entrypoint_name!r} is missing in {module_name!r}."
        )

    project = state["project"]
    profile_settings = project["profile_settings"]
    inputs = _resolve_stage_inputs(state, stage)
    workspace = _project_workspace(state)

    common = {
        "topic": project["topic"],
        "language": project["language"],
        "profile": project["profile"],
        "output_root": str(workspace),
        "output_dir": str(output_dir),
        "test_mode": False,
        "project_id": state["project_id"],
        "target_duration_seconds": profile_settings["duration_seconds"],
    }

    if stage == "slot1_research":
        values = {
            **common,
            "output_path": str(output_dir / "research_output.json"),
            "max_sources": int(os.getenv("CINEMA_MAX_RESEARCH_SOURCES", "12")),
            "timeout_seconds": int(os.getenv("CINEMA_RESEARCH_REQUEST_TIMEOUT", "12")),
        }

    elif stage == "slot2_screenplay":
        if inputs["research"] is None:
            raise StageExecutionError("Research output is required for screenplay.")
        values = {
            **common,
            "research_path": str(inputs["research"]),
            "research_output_path": str(inputs["research"]),
            "output_path": str(output_dir / "screenplay_blueprint.json"),
            "profile": profile_settings["screenplay_profile"],
        }

    elif stage == "slot4_audio":
        if inputs["screenplay"] is None:
            raise StageExecutionError("Screenplay output is required for audio.")
        values = {
            **common,
            "screenplay_path": str(inputs["screenplay"]),
            "output_dir": str(output_dir),
            "timeline_path": str(output_dir / "audio_timeline.json"),
            "voice": os.getenv("CINEMA_DEFAULT_VOICE", "default"),
        }

    elif stage == "slot3_visuals":
        if inputs["screenplay"] is None or inputs["timeline"] is None:
            raise StageExecutionError(
                "Screenplay and audio timeline are required for visuals."
            )
        values = {
            **common,
            "screenplay_path": str(inputs["screenplay"]),
            "timeline_path": str(inputs["timeline"]),
            "output_dir": str(output_dir),
            "device": os.getenv("CINEMA_DEVICE", "auto"),
            "model_name": os.getenv("CINEMA_VISUAL_MODEL") or None,
            "reference_image_path": os.getenv("CINEMA_REFERENCE_IMAGE") or None,
            "batch_size": max(1, int(os.getenv("CINEMA_VISUAL_BATCH_SIZE", "1"))),
            "seed": int(os.getenv("CINEMA_RANDOM_SEED", "42")),
        }

    elif stage == "slot5_editor":
        if (
            inputs["screenplay"] is None
            or inputs["timeline"] is None
            or inputs["visual_manifest"] is None
        ):
            raise StageExecutionError(
                "Screenplay, audio timeline and visual manifest are required."
            )
        values = {
            **common,
            "screenplay_path": str(inputs["screenplay"]),
            "timeline_path": str(inputs["timeline"]),
            "visual_manifest_path": str(inputs["visual_manifest"]),
            "output_path": str(output_dir / "final_video.mp4"),
            "lut_path": os.getenv("CINEMA_LUT_PATH") or None,
        }

    else:
        raise StageExecutionError(f"Unsupported production stage: {stage}")

    try:
        result = _invoke_supported(function, values)
    except Exception as exc:
        raise StageExecutionError(
            f"{stage} failed: {type(exc).__name__}: {exc}"
        ) from exc

    candidates: list[Path] = []
    if isinstance(result, (str, Path)):
        candidates.append(resolve_path(result))
    elif isinstance(result, dict):
        for key in (
            "path", "output_path", "artifact_path", "video_path",
            "timeline_path", "manifest_path",
        ):
            value = result.get(key)
            if isinstance(value, (str, Path)):
                candidates.append(resolve_path(value))
        artifacts = result.get("artifacts")
        if isinstance(artifacts, list):
            candidates.extend(
                resolve_path(item)
                for item in artifacts
                if isinstance(item, (str, Path))
            )
    elif isinstance(result, (list, tuple, set)):
        candidates.extend(
            resolve_path(item)
            for item in result
            if isinstance(item, (str, Path))
        )

    # Include files written directly by the slot.
    candidates.extend(_find_artifacts(state, stage))

    unique: dict[str, Path] = {}
    for path in candidates:
        if path.exists() and path.is_file() and is_within(path, workspace):
            unique[str(path.resolve())] = path.resolve()

    if not unique:
        raise StageExecutionError(
            f"{stage} returned without creating a verifiable workspace artifact."
        )

    return sorted(unique.values(), key=lambda item: item.stat().st_mtime)


# ---------------------------------------------------------------------------
# Stage execution and recovery
# ---------------------------------------------------------------------------

def execute_stage(
    job_id: str,
    stage_name: str,
    stage_input: dict[str, Any] | None = None,
    test_mode: bool | None = None,
) -> dict[str, Any]:
    """Execute one stage, validate outputs and persist its checkpoint."""
    del stage_input  # Stage inputs are resolved from verified project artifacts.

    with _ACTIVE_JOBS_LOCK:
        job = _ACTIVE_JOBS.get(job_id)

    if job is None or not job.get("state_path"):
        raise PipelineError(f"Pipeline job is not registered: {job_id}")

    state_path = resolve_path(job["state_path"])
    state = load_project_state(state_path)

    if stage_name not in state["stages"]:
        raise PipelineError(f"Stage is not configured for this project: {stage_name}")

    stage_record = state["stages"][stage_name]
    stage_order = state["pipeline"]["stage_order"]
    stage_index = stage_order.index(stage_name)

    if stage_index > 0:
        previous = stage_order[stage_index - 1]
        if state["stages"][previous]["status"] != "completed":
            raise PipelineError(
                f"Cannot run {stage_name} before {previous} completes."
            )

    if stage_record["status"] == "completed":
        records = state.get("artifacts", {}).get(stage_name, [])
        workspace = _project_workspace(state)
        if records and all(_verify_artifact_record(item, workspace) for item in records):
            return state
        stage_record["status"] = "pending"

    _check_cancelled(job_id)
    selected_test_mode = (
        bool(state["project"]["test_mode"])
        if test_mode is None
        else bool(test_mode)
    )

    output_dir = _stage_workspace(state, stage_name)
    start = time.monotonic()

    stage_record.update({
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "duration_seconds": None,
        "attempts": int(stage_record.get("attempts", 0)) + 1,
        "error": None,
    })
    state["status"] = "running"
    state["pipeline"]["current_stage"] = stage_name
    state["pipeline"]["started_at"] = (
        state["pipeline"].get("started_at") or utc_now()
    )
    save_project_state(state)

    logger.info("Starting %s for %s", stage_name, job_id)

    try:
        timeout = max(
            1,
            int(os.getenv(
                f"CINEMA_TIMEOUT_{stage_name.upper()}",
                str(STAGE_TIMEOUT_SECONDS.get(stage_name, 600)),
            )),
        )

        # Python cannot safely terminate a running slot thread. The timeout
        # reports the overrun; slots should also implement their own I/O limits.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            _run_test_stage if selected_test_mode else _run_production_stage,
            state,
            stage_name,
            output_dir,
        )
        try:
            artifacts = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise StageExecutionError(
                f"{stage_name} exceeded its {timeout}-second timeout. "
                "The running slot may need to reach its own cancellation point."
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        _check_cancelled(job_id)

        workspace = _project_workspace(state)
        records = [
            _artifact_record(path, workspace)
            for path in artifacts
            if path.exists() and path.is_file()
        ]
        if not records:
            raise StageExecutionError(f"{stage_name} produced no valid artifacts.")

        stage_record.update({
            "status": "completed",
            "completed_at": utc_now(),
            "duration_seconds": round(time.monotonic() - start, 3),
            "artifacts": records,
            "error": None,
        })
        state.setdefault("artifacts", {})[stage_name] = records

        if stage_name not in state["pipeline"]["completed_stages"]:
            state["pipeline"]["completed_stages"].append(stage_name)

        state["pipeline"]["failed_stage"] = None
        state["pipeline"]["last_error"] = None
        save_project_state(state)

        logger.info("Completed %s for %s", stage_name, job_id)
        return state

    except PipelineCancelled:
        stage_record["status"] = "cancelled"
        stage_record["completed_at"] = utc_now()
        stage_record["duration_seconds"] = round(time.monotonic() - start, 3)
        state["status"] = "cancelled"
        state["pipeline"]["cancel_reason"] = "Cancellation requested."
        save_project_state(state)
        raise

    except Exception as exc:
        stage_record.update({
            "status": "failed",
            "completed_at": utc_now(),
            "duration_seconds": round(time.monotonic() - start, 3),
            "error": f"{type(exc).__name__}: {exc}",
        })
        state["status"] = "failed"
        state["pipeline"]["failed_stage"] = stage_name
        state["pipeline"]["last_error"] = stage_record["error"]
        save_project_state(state)
        raise


def recover_project(
    state_path: str | Path,
    test_mode: bool | None = None,
) -> dict[str, Any]:
    """Invalidate corrupted checkpoints and preserve valid completed stages."""
    del test_mode
    state = load_project_state(state_path)
    workspace = _project_workspace(state)
    invalidated: list[str] = []
    invalidate_following = False

    for stage in state["pipeline"]["stage_order"]:
        record = state["stages"][stage]
        artifact_records = state.get("artifacts", {}).get(stage, [])
        valid = (
            record.get("status") == "completed"
            and bool(artifact_records)
            and all(_verify_artifact_record(item, workspace) for item in artifact_records)
        )

        if invalidate_following or (record.get("status") == "completed" and not valid):
            invalidate_following = True
            record.update(_default_stage_record())
            state.get("artifacts", {}).pop(stage, None)
            invalidated.append(stage)

    completed = []
    for stage in state["pipeline"]["stage_order"]:
        if state["stages"][stage]["status"] != "completed":
            break
        completed.append(stage)

    state["pipeline"]["completed_stages"] = completed
    state["pipeline"]["current_stage"] = None
    state["pipeline"]["failed_stage"] = None
    state["pipeline"]["last_error"] = None
    state["status"] = "created" if not completed else "running"
    state["recovery"] = {
        "recovered_at": utc_now(),
        "invalidated_stages": invalidated,
    }
    save_project_state(state)
    _register_job(state["project_id"], resolve_path(state["state_path"]))
    return state


# ---------------------------------------------------------------------------
# Pipeline lifecycle and status API
# ---------------------------------------------------------------------------

def run_pipeline(job_id: str, test_mode: bool = False) -> dict[str, Any]:
    """Run all configured stages in their declared dependency order."""
    with _ACTIVE_JOBS_LOCK:
        job = _ACTIVE_JOBS.get(job_id)

    if job is None or not job.get("state_path"):
        raise PipelineError(f"Pipeline job is not registered: {job_id}")

    state_path = resolve_path(job["state_path"])
    state = load_project_state(state_path)
    _register_job(job_id, state_path)

    event = _cancel_event(job_id)
    if event.is_set():
        raise PipelineCancelled(f"Pipeline was cancelled before starting: {job_id}")

    started = time.monotonic()
    state["project"]["test_mode"] = bool(test_mode)
    state["status"] = "running"
    state["pipeline"]["started_at"] = state["pipeline"].get("started_at") or utc_now()
    state["pipeline"]["completed_at"] = None
    state["pipeline"]["last_error"] = None
    save_project_state(state)

    try:
        state = recover_project(state_path, test_mode)
        state["status"] = "running"
        state["project"]["test_mode"] = bool(test_mode)
        save_project_state(state)

        for stage in state["pipeline"]["stage_order"]:
            _check_cancelled(job_id)
            state = load_project_state(state_path)
            if state["stages"][stage]["status"] == "completed":
                continue
            execute_stage(job_id, stage, test_mode=test_mode)

        _check_cancelled(job_id)
        state = load_project_state(state_path)

        if test_mode:
            video_path = _create_test_video(state)
            record = _artifact_record(video_path, _project_workspace(state))
            state.setdefault("artifacts", {}).setdefault("slot5_editor", []).append(record)
            state["stages"]["slot5_editor"]["artifacts"].append(record)

        state["status"] = "completed"
        state["pipeline"]["current_stage"] = None
        state["pipeline"]["completed_at"] = utc_now()
        state["pipeline"]["duration_seconds"] = round(time.monotonic() - started, 3)
        state["pipeline"]["last_error"] = None
        save_project_state(state)

        return {
            "job_id": job_id,
            "project_id": state["project_id"],
            "status": "completed",
            "progress": 100.0,
            "state_path": str(state_path),
            "artifacts": [
                record["path"]
                for records in state.get("artifacts", {}).values()
                for record in records
            ],
        }

    except PipelineCancelled:
        state = load_project_state(state_path)
        state["status"] = "cancelled"
        state["pipeline"]["current_stage"] = None
        state["pipeline"]["completed_at"] = utc_now()
        state["pipeline"]["duration_seconds"] = round(time.monotonic() - started, 3)
        state["pipeline"]["cancel_reason"] = state["pipeline"].get(
            "cancel_reason", "Cancellation requested."
        )
        save_project_state(state)
        return {
            "job_id": job_id,
            "project_id": state["project_id"],
            "status": "cancelled",
            "progress": _progress(state),
            "state_path": str(state_path),
        }

    except Exception as exc:
        logger.exception("Pipeline failed for %s", job_id)
        state = load_project_state(state_path)
        state["status"] = "failed"
        state["pipeline"]["completed_at"] = utc_now()
        state["pipeline"]["duration_seconds"] = round(time.monotonic() - started, 3)
        state["pipeline"]["last_error"] = f"{type(exc).__name__}: {exc}"
        save_project_state(state)
        return {
            "job_id": job_id,
            "project_id": state["project_id"],
            "status": "failed",
            "progress": _progress(state),
            "error": state["pipeline"]["last_error"],
            "state_path": str(state_path),
        }


def _progress(state: dict[str, Any]) -> float:
    """Calculate progress from the configured stage count."""
    stages = state["pipeline"]["stage_order"]
    if not stages:
        return 0.0
    completed = sum(
        1 for stage in stages
        if state["stages"].get(stage, {}).get("status") == "completed"
    )
    return round(100.0 * completed / len(stages), 2)


def get_pipeline_status(job_id: str) -> dict[str, Any]:
    """Return the API-compatible pipeline status for a registered job."""
    with _ACTIVE_JOBS_LOCK:
        job = _ACTIVE_JOBS.get(job_id)

    if job is None or not job.get("state_path"):
        return {
            "job_id": job_id,
            "status": "failed",
            "progress": 0.0,
            "error": "Pipeline job is not registered in this process.",
        }

    try:
        state = load_project_state(job["state_path"])
    except Exception as exc:
        return {
            "job_id": job_id,
            "status": "failed",
            "progress": 0.0,
            "error": f"Unable to load project state: {exc}",
        }

    stage_order = state["pipeline"]["stage_order"]
    current_stage = state["pipeline"].get("current_stage")
    if not current_stage:
        current_stage = next(
            (
                stage for stage in stage_order
                if state["stages"].get(stage, {}).get("status") in {"running", "failed"}
            ),
            None,
        )

    statuses = {
        name: record.get("status", "pending")
        for name, record in state["stages"].items()
    }

    return {
        "job_id": job_id,
        "project_id": state["project_id"],
        "status": state["status"],
        "current_stage": current_stage,
        "stage": current_stage,
        "progress": 100.0 if state["status"] == "completed" else _progress(state),
        "completed_stages": list(state["pipeline"].get("completed_stages", [])),
        "stage_statuses": statuses,
        "failed_stage": state["pipeline"].get("failed_stage"),
        "cancel_reason": state["pipeline"].get("cancel_reason"),
        "error": state["pipeline"].get("last_error"),
        "message": state["pipeline"].get("last_error"),
        "state_path": str(state["state_path"]),
        "workspace_dir": state["project"]["workspace_dir"],
        "updated_at": state.get("updated_at"),
        "test_mode": bool(state["project"].get("test_mode", False)),
        "artifacts": state.get("artifacts", {}),
    }


def cancel_pipeline(
    job_id: str,
    reason: str = "Cancellation requested.",
) -> dict[str, Any]:
    """Request cooperative cancellation for a pipeline job."""
    with _ACTIVE_JOBS_LOCK:
        job = _ACTIVE_JOBS.get(job_id)

    if job is None or not job.get("state_path"):
        raise PipelineError(f"Pipeline job is not registered: {job_id}")

    state_path = resolve_path(job["state_path"])
    state = load_project_state(state_path)

    if state["status"] in TERMINAL_STATES:
        return get_pipeline_status(job_id)

    _cancel_event(job_id).set()
    state["status"] = "cancelling"
    state["pipeline"]["cancel_reason"] = str(reason)
    save_project_state(state)

    return get_pipeline_status(job_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous AI Cinema Studio Engine")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--language", default="Hindi")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="reel")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    state = create_project(
        topic=args.topic,
        language=args.language,
        profile=args.profile,
        output_root=args.output_root,
        test_mode=args.test_mode,
    )
    result = run_pipeline(state["project_id"], args.test_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
