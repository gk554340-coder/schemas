# master_pipeline.py
# Autonomous AI Cinema Studio Engine
# Central orchestrator and sole owner of project_state.json.
# Python 3.10+

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = ROOT_DIR / "workspace"
STATE_DIR = WORKSPACE_DIR / "05_master_theater" / "state"
DEFAULT_OUTPUT_ROOT = WORKSPACE_DIR / "05_master_theater"
STATE_FILENAME = "project_state.json"
STATE_VERSION = "1.0.0"

STAGE_ORDER = (
    "slot1_research",
    "slot2_screenplay",
    "slot4_audio",
    "slot3_visuals",
    "slot5_editor",
)

STAGE_TIMEOUT_SECONDS = {
    "slot1_research": 300,
    "slot2_screenplay": 300,
    "slot4_audio": 600,
    "slot3_visuals": 900,
    "slot5_editor": 600,
}

PROFILE_DEFAULTS = {
    "reel": {
        "duration_seconds": 15,
        "aspect_ratio": "9:16",
        "resolution": "720x1280",
        "description": "Short-form vertical cinematic video",
    },
    "documentary": {
        "duration_seconds": 60,
        "aspect_ratio": "16:9",
        "resolution": "1280x720",
        "description": "Narrative documentary video",
    },
}

LOGGER = logging.getLogger("cinema.master_pipeline")

_ACTIVE_JOBS: dict[str, dict[str, Any]] = {}
_ACTIVE_JOBS_LOCK = threading.RLock()
_CANCEL_EVENTS: dict[str, threading.Event] = {}
_CANCEL_EVENTS_LOCK = threading.RLock()


class PipelineError(RuntimeError):
    """Raised when a pipeline stage or project operation fails."""


class PipelineCancelled(PipelineError):
    """Raised when a pipeline is cancelled by the caller."""


class StateIntegrityError(PipelineError):
    """Raised when project state or a stage artifact fails integrity checks."""


class StageExecutionError(PipelineError):
    """Raised when a stage cannot complete successfully."""


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert supported objects into JSON-compatible values."""
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="json")

    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Unable to read JSON file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise PipelineError(f"Expected a JSON object in {path}")

    return data


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically using a temporary file in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(
                _json_safe(data),
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)

        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            LOGGER.debug("Directory fsync is unavailable for %s", path.parent)

    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("Could not remove temporary state file %s", temporary_path)


@contextmanager
def _project_lock(state_path: Path):
    """Acquire a process-local lock for operations targeting the same state file."""
    lock_key = str(state_path.resolve())
    lock_name = "_lock_" + hashlib.sha256(lock_key.encode("utf-8")).hexdigest()

    with _STATE_LOCKS_GUARD:
        lock = _STATE_LOCKS.setdefault(lock_name, threading.RLock())

    lock.acquire()
    try:
        yield
    finally:
        lock.release()


_STATE_LOCKS: dict[str, threading.RLock] = {}
_STATE_LOCKS_GUARD = threading.RLock()


def sha256_file(path: str | Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    file_path = Path(path)

    if not file_path.is_file():
        raise StateIntegrityError(f"Checksum target is not a file: {file_path}")

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a path, interpreting relative paths against a supplied base."""
    candidate = Path(path).expanduser()

    if not candidate.is_absolute():
        candidate = (base_dir or ROOT_DIR) / candidate

    return candidate.resolve()


def _assert_within(path: Path, root: Path, description: str) -> Path:
    """Ensure a resolved path is inside its designated workspace root."""
    resolved_path = path.resolve()
    resolved_root = root.resolve()

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise PipelineError(
            f"{description} must remain inside {resolved_root}: {resolved_path}"
        ) from exc

    return resolved_path


def _project_workspace(state: dict[str, Any]) -> Path:
    """Get the project workspace recorded in project state."""
    project = state.get("project", {})
    raw_workspace = project.get("workspace_dir")

    if raw_workspace:
        return _resolve_path(raw_workspace)

    project_id = state.get("project_id", "unknown")
    return DEFAULT_OUTPUT_ROOT / "projects" / str(project_id)


def _get_state_path(state: dict[str, Any]) -> Path:
    """Get the canonical project-state path recorded in the state object."""
    raw_path = state.get("state_path")

    if raw_path:
        return _resolve_path(raw_path)

    return _project_workspace(state) / STATE_FILENAME


def _normalize_stage_record(stage: str) -> dict[str, Any]:
    """Create the default record for a pipeline stage."""
    return {
        "name": stage,
        "status": "pending",
        "attempts": 0,
        "started_at": None,
        "completed_at": None,
        "input": {},
        "outputs": {},
        "artifacts": [],
        "error": None,
        "duration_seconds": None,
    }


def _default_project_state(
    project_id: str,
    topic: str,
    language: str,
    profile: str,
    workspace_dir: Path,
    state_path: Path,
    test_mode: bool,
) -> dict[str, Any]:
    """Create the canonical project state structure."""
    return {
        "schema_version": STATE_VERSION,
        "project_id": project_id,
        "state_path": str(state_path.resolve()),
        "status": "created",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "test_mode": bool(test_mode),
        "project": {
            "topic": topic,
            "language": language,
            "profile": profile,
            "profile_settings": dict(PROFILE_DEFAULTS[profile]),
            "workspace_dir": str(workspace_dir.resolve()),
            "output_root": str(workspace_dir.resolve()),
        },
        "pipeline": {
            "stage_order": list(STAGE_ORDER),
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
            stage: _normalize_stage_record(stage)
            for stage in STAGE_ORDER
        },
        "artifacts": {},
        "integrity": {
            "algorithm": "sha256",
            "state_revision": 0,
            "last_verified_at": None,
        },
        "recovery": {
            "recovered_at": None,
            "actions": [],
        },
    }


def _validate_state_shape(state: dict[str, Any]) -> None:
    """Validate the minimum required structure for a project state."""
    if not isinstance(state, dict):
        raise StateIntegrityError("Project state must be a JSON object")

    if not state.get("project_id"):
        raise StateIntegrityError("Project state is missing project_id")

    project = state.get("project")
    if not isinstance(project, dict):
        raise StateIntegrityError("Project state is missing its project object")

    pipeline = state.get("pipeline")
    if not isinstance(pipeline, dict):
        raise StateIntegrityError("Project state is missing its pipeline object")

    stages = state.get("stages")
    if not isinstance(stages, dict):
        raise StateIntegrityError("Project state is missing its stages object")

    for stage in STAGE_ORDER:
        if stage not in stages or not isinstance(stages[stage], dict):
            raise StateIntegrityError(f"Project state is missing stage record: {stage}")


def _artifact_record(path: Path, kind: str | None = None) -> dict[str, Any]:
    """Create a manifest record for an existing artifact."""
    if not path.is_file():
        raise StateIntegrityError(f"Expected artifact does not exist: {path}")

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "kind": kind or path.suffix.lstrip(".") or "file",
        "verified_at": utc_now(),
    }


def _verify_artifact_record(
    artifact: dict[str, Any],
    *,
    required: bool = True,
) -> bool:
    """Verify an artifact manifest record against the file on disk."""
    raw_path = artifact.get("path")
    expected_checksum = artifact.get("sha256")

    if not raw_path:
        if required:
            raise StateIntegrityError("Artifact record is missing its path")
        return False

    path = _resolve_path(raw_path)

    if not path.is_file():
        if required:
            raise StateIntegrityError(f"Artifact file is missing: {path}")
        return False

    if expected_checksum:
        actual_checksum = sha256_file(path)
        if actual_checksum.lower() != str(expected_checksum).lower():
            raise StateIntegrityError(
                f"SHA-256 mismatch for {path}: expected {expected_checksum}, "
                f"got {actual_checksum}"
            )

    expected_size = artifact.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise StateIntegrityError(f"Artifact size mismatch for {path}")

    return True


def create_project(
    topic: str,
    language: str,
    profile: str,
    output_root: str | Path,
    test_mode: bool,
) -> dict[str, Any]:
    """Create a project workspace and its canonical project_state.json."""
    topic = str(topic).strip()
    language = str(language).strip()
    profile = str(profile).strip().lower()

    if not topic:
        raise ValueError("topic must not be empty")

    if not language:
        raise ValueError("language must not be empty")

    if profile not in PROFILE_DEFAULTS:
        raise ValueError(
            f"Unsupported profile '{profile}'. Choose one of: "
            f"{', '.join(PROFILE_DEFAULTS)}"
        )

    output_path = _resolve_path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    project_id = f"cinema-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    workspace_dir = output_path / "projects" / project_id
    workspace_dir.mkdir(parents=True, exist_ok=False)

    for relative_dir in (
        "01_research_vault",
        "02_screenplay_desk",
        "03_audio_studio",
        "04_visual_lab",
        "05_master_theater",
        "state",
    ):
        (workspace_dir / relative_dir).mkdir(parents=True, exist_ok=True)

    state_path = workspace_dir / STATE_FILENAME
    state = _default_project_state(
        project_id=project_id,
        topic=topic,
        language=language,
        profile=profile,
        workspace_dir=workspace_dir,
        state_path=state_path,
        test_mode=test_mode,
    )

    state["project"]["output_root"] = str(output_path.resolve())
    save_project_state(state_path, state)

    LOGGER.info("Created project %s", project_id)
    return state


def load_project_state(state_path: str | Path) -> dict[str, Any]:
    """Load and validate project state from its canonical JSON file."""
    path = _resolve_path(state_path)

    if not path.is_file():
        raise FileNotFoundError(f"Project state file not found: {path}")

    state = _read_json(path)
    _validate_state_shape(state)

    recorded_path = state.get("state_path")
    if recorded_path:
        expected_path = _resolve_path(recorded_path)
        if expected_path != path:
            raise StateIntegrityError(
                f"State path mismatch: file is {path}, but state records {expected_path}"
            )

    state.setdefault("integrity", {})
    state["integrity"].setdefault("algorithm", "sha256")
    state["integrity"].setdefault("state_revision", 0)

    for stage in STAGE_ORDER:
        state["stages"][stage].setdefault("artifacts", [])
        state["stages"][stage].setdefault("outputs", {})
        state["stages"][stage].setdefault("attempts", 0)

    return state


def save_project_state(
    state_path: str | Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Atomically persist project state; this module is its sole writer."""
    path = _resolve_path(state_path)
    _validate_state_shape(state)

    state["state_path"] = str(path)
    state["updated_at"] = utc_now()

    integrity = state.setdefault("integrity", {})
    integrity["algorithm"] = "sha256"
    integrity["state_revision"] = int(integrity.get("state_revision", 0)) + 1

    _atomic_write_json(path, state)
    return state


def _cancel_event(job_id: str) -> threading.Event:
    """Get or create a cancellation event for a job."""
    with _CANCEL_EVENTS_LOCK:
        return _CANCEL_EVENTS.setdefault(job_id, threading.Event())


def _check_cancelled(job_id: str) -> None:
    """Raise an exception if the job has been cancelled."""
    if _cancel_event(job_id).is_set():
        with _ACTIVE_JOBS_LOCK:
            record = _ACTIVE_JOBS.get(job_id, {})
            reason = record.get("cancel_reason") or "Cancelled by caller"

        raise PipelineCancelled(f"Pipeline {job_id} cancelled: {reason}")


def _write_json_artifact(path: Path, data: dict[str, Any]) -> Path:
    """Write a stage JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, data)
    return path


def _extract_scenes(blueprint: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract scene records from a screenplay blueprint."""
    scenes = blueprint.get("scenes")

    if not isinstance(scenes, list):
        screenplay = blueprint.get("screenplay")
        if isinstance(screenplay, dict):
            scenes = screenplay.get("scenes")

    if not isinstance(scenes, list):
        scenes = []

    normalized = []
    for index, scene in enumerate(scenes):
        if isinstance(scene, dict):
            normalized.append(scene)
        else:
            normalized.append({"scene_number": index + 1, "text": str(scene)})

    return normalized


def _load_stage_module(module_name: str):
    """Import a stage module from the project root."""
    import importlib

    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise StageExecutionError(
            f"Could not import stage module '{module_name}': {exc}"
        ) from exc


def _call_supported(function: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Call a stage function with only the keyword arguments it accepts."""
    import inspect

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(**kwargs)

    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return function(**filtered)


def _invoke_stage_entrypoint(
    module: Any,
    entrypoint_names: tuple[str, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Invoke the first available stage entrypoint."""
    for name in entrypoint_names:
        function = getattr(module, name, None)
        if callable(function):
            return _call_supported(function, kwargs)

    raise StageExecutionError(
        f"Stage module {module.__name__} has none of these entrypoints: "
        f"{', '.join(entrypoint_names)}"
    )


def _find_artifacts(
    result: Any,
    candidate_roots: list[Path],
) -> list[Path]:
    """Find artifact paths returned by a stage or written in its output directories."""
    found: dict[str, Path] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Path):
            if value.is_file():
                found[str(value.resolve())] = value.resolve()
            return

        if isinstance(value, str):
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = ROOT_DIR / candidate
            try:
                resolved = candidate.resolve()
                if resolved.is_file():
                    found[str(resolved)] = resolved
            except OSError:
                pass
            return

        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"path", "file", "file_path", "output_path"}:
                    visit(item)
                elif isinstance(item, (dict, list, tuple, Path)):
                    visit(item)
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(result)

    for root in candidate_roots:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and not path.name.startswith("."):
                    found[str(path.resolve())] = path.resolve()

    return sorted(found.values(), key=lambda item: str(item))


def _stage_workspace(state: dict[str, Any], stage_name: str) -> Path:
    """Return the expected output workspace for a stage."""
    workspace = _project_workspace(state)

    mapping = {
        "slot1_research": "01_research_vault",
        "slot2_screenplay": "02_screenplay_desk",
        "slot4_audio": "03_audio_studio",
        "slot3_visuals": "04_visual_lab",
        "slot5_editor": "05_master_theater",
    }

    return workspace / mapping[stage_name]


def _locate_latest_artifact(
    roots: list[Path],
    suffixes: tuple[str, ...] = (".json",),
    exclude: set[str] | None = None,
) -> Path | None:
    """Locate the most recently modified artifact with a matching extension."""
    excluded = exclude or set()
    candidates: list[Path] = []

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in suffixes
                and path.name not in excluded
                and not path.name.startswith(".")
            ):
                candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda item: item.stat().st_mtime)


def _run_test_stage(
    state: dict[str, Any],
    stage_name: str,
    stage_input: dict[str, Any],
) -> dict[str, Any]:
    """Produce deterministic lightweight artifacts for offline CI test mode."""
    output_dir = _stage_workspace(state, stage_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    project = state["project"]
    topic = project["topic"]
    language = project["language"]
    profile = project["profile"]
    profile_settings = project.get("profile_settings", PROFILE_DEFAULTS[profile])

    if stage_name == "slot1_research":
        output_path = output_dir / "research_output.json"
        payload = {
            "schema_version": "1.0.0",
            "project_id": state["project_id"],
            "topic": topic,
            "language": language,
            "profile": profile,
            "summary": f"Offline research brief for: {topic}",
            "key_points": [
                f"Introduce the central idea: {topic}",
                "Provide concise supporting context",
                "End with a clear narrative takeaway",
            ],
            "sources": [],
            "test_mode": True,
            "created_at": utc_now(),
        }

    elif stage_name == "slot2_screenplay":
        research_path = stage_input.get("research_path")
        research = _read_json(Path(research_path)) if research_path else {}

        output_path = output_dir / "screenplay_blueprint.json"
        target_duration = int(profile_settings.get("duration_seconds", 15))
        scene_count = 3 if target_duration >= 10 else 1
        scene_duration = max(1, target_duration // scene_count)
        scenes = []

        for index in range(scene_count):
            scenes.append(
                {
                    "scene_number": index + 1,
                    "title": f"Scene {index + 1}",
                    "duration_seconds": scene_duration,
                    "narration": (
                        f"{topic}. "
                        f"{research.get('key_points', ['A cinematic introduction'])[index % 3]}."
                    ),
                    "visual_prompt": (
                        f"Cinematic shot about {topic}, scene {index + 1}, "
                        "coherent lighting, clear subject, no on-screen text"
                    ),
                    "shot_type": ("wide", "medium", "close-up")[index % 3],
                }
            )

        actual_duration = sum(
            int(scene["duration_seconds"]) for scene in scenes
        )

        payload = {
            "schema_version": "1.0.0",
            "project_id": state["project_id"],
            "topic": topic,
            "language": language,
            "profile": profile,
            "title": topic,
            "duration_seconds": actual_duration,
            "aspect_ratio": profile_settings.get("aspect_ratio", "9:16"),
            "resolution": profile_settings.get("resolution", "720x1280"),
            "scenes": scenes,
            "test_mode": True,
            "created_at": utc_now(),
        }

    elif stage_name == "slot4_audio":
        screenplay_path = stage_input.get("screenplay_path")
        blueprint = _read_json(Path(screenplay_path)) if screenplay_path else {}
        scenes = _extract_scenes(blueprint)

        timeline_segments = []
        cursor = 0.0

        for index, scene in enumerate(scenes):
            duration = max(1.0, float(scene.get("duration_seconds", 1)))
            narration = str(
                scene.get("narration")
                or scene.get("dialogue")
                or scene.get("text")
                or topic
            )

            timeline_segments.append(
                {
                    "segment_id": f"narration-{index + 1:03d}",
                    "scene_number": index + 1,
                    "start_seconds": round(cursor, 3),
                    "end_seconds": round(cursor + duration, 3),
                    "duration_seconds": duration,
                    "text": narration,
                    "audio_path": None,
                    "test_mode": True,
                }
            )
            cursor += duration

        output_path = output_dir / "audio_timeline.json"
        payload = {
            "schema_version": "1.0.0",
            "project_id": state["project_id"],
            "duration_seconds": cursor,
            "language": language,
            "segments": timeline_segments,
            "events": timeline_segments,
            "music": [],
            "sound_effects": [],
            "test_mode": True,
            "created_at": utc_now(),
        }

    elif stage_name == "slot3_visuals":
        screenplay_path = stage_input.get("screenplay_path")
        timeline_path = stage_input.get("timeline_path")

        blueprint = _read_json(Path(screenplay_path)) if screenplay_path else {}
        timeline = _read_json(Path(timeline_path)) if timeline_path else {}
        scenes = _extract_scenes(blueprint)

        duration = float(
            timeline.get("duration_seconds")
            or blueprint.get("duration_seconds")
            or sum(float(scene.get("duration_seconds", 1)) for scene in scenes)
        )

        output_path = output_dir / "visual_manifest.json"
        frame_records = []
        cursor = 0.0

        for index, scene in enumerate(scenes):
            scene_duration = max(1.0, float(scene.get("duration_seconds", 1)))
            frame_path = output_dir / f"scene_{index + 1:03d}.ppm"

            # PPM is a valid, dependency-free image format for lightweight CI.
            width, height = 32, 32
            color = (
                (index * 67 + 40) % 256,
                (index * 97 + 80) % 256,
                (index * 127 + 120) % 256,
            )
            header = f"P6\n{width} {height}\n255\n".encode("ascii")
            pixel = bytes(color)
            image_data = header + pixel * (width * height)

            temporary = frame_path.with_suffix(".ppm.tmp")
            temporary.write_bytes(image_data)
            os.replace(temporary, frame_path)

            frame_records.append(
                {
                    "scene_number": index + 1,
                    "start_seconds": round(cursor, 3),
                    "end_seconds": round(min(cursor + scene_duration, duration), 3),
                    "duration_seconds": min(scene_duration, max(0.0, duration - cursor)),
                    "frame_path": str(frame_path.resolve()),
                    "sha256": sha256_file(frame_path),
                    "visual_prompt": scene.get("visual_prompt", topic),
                }
            )
            cursor += scene_duration

        payload = {
            "schema_version": "1.0.0",
            "project_id": state["project_id"],
            "duration_seconds": duration,
            "frames": frame_records,
            "scenes": frame_records,
            "motion": [],
            "test_mode": True,
            "created_at": utc_now(),
        }

    elif stage_name == "slot5_editor":
        output_path = output_dir / "test_export_spec.json"
        screenplay_path = stage_input.get("screenplay_path")
        timeline_path = stage_input.get("timeline_path")
        visual_manifest_path = stage_input.get("visual_manifest_path")

        blueprint = _read_json(Path(screenplay_path)) if screenplay_path else {}
        timeline = _read_json(Path(timeline_path)) if timeline_path else {}
        visual_manifest = (
            _read_json(Path(visual_manifest_path))
            if visual_manifest_path
            else {}
        )

        duration = float(
            timeline.get("duration_seconds")
            or blueprint.get("duration_seconds")
            or profile_settings.get("duration_seconds", 15)
        )

        payload = {
            "schema_version": "1.0.0",
            "project_id": state["project_id"],
            "title": topic,
            "language": language,
            "profile": profile,
            "duration_seconds": duration,
            "expected_export_path": str(
                (output_dir / "final_video.mp4").resolve()
            ),
            "screenplay_path": screenplay_path,
            "timeline_path": timeline_path,
            "visual_manifest_path": visual_manifest_path,
            "scene_count": len(_extract_scenes(blueprint)),
            "visual_count": len(
                visual_manifest.get("frames", visual_manifest.get("scenes", []))
            ),
            "test_mode": True,
            "created_at": utc_now(),
        }

    else:
        raise StageExecutionError(f"No offline test handler for {stage_name}")

    _write_json_artifact(output_path, payload)

    return {
        "status": "completed",
        "outputs": {
            "primary": str(output_path.resolve()),
            "duration_seconds": payload.get("duration_seconds"),
        },
        "artifacts": [str(output_path.resolve())],
        "test_mode": True,
    }


def _run_production_stage(
    state: dict[str, Any],
    stage_name: str,
    stage_input: dict[str, Any],
) -> dict[str, Any]:
    """Execute the actual stage module using its public entrypoint."""
    project = state["project"]
    output_dir = _stage_workspace(state, stage_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "topic": project["topic"],
        "language": project["language"],
        "profile": project["profile"],
        "output_root": str(project.get("output_root", output_dir)),
        "output_dir": str(output_dir),
        "test_mode": False,
        "project_id": state["project_id"],
    }

    if stage_name == "slot1_research":
        module = _load_stage_module("slot1_research")
        kwargs = {
            **common,
            "topic": project["topic"],
            "language": project["language"],
            "profile": project["profile"],
            "output_dir": str(output_dir),
        }
        result = _invoke_stage_entrypoint(
            module,
            ("run_research", "run", "research_topic", "execute"),
            kwargs,
        )

    elif stage_name == "slot2_screenplay":
        module = _load_stage_module("slot2_screenplay")
        research_path = stage_input.get("research_path")

        if not research_path:
            research_path = str(
                _locate_latest_artifact(
                    [_stage_workspace(state, "slot1_research")]
                )
                or ""
            )

        if not research_path:
            raise StageExecutionError("No research artifact is available")

        kwargs = {
            **common,
            "research_path": research_path,
            "research_output_path": research_path,
            "output_dir": str(output_dir),
        }
        result = _invoke_stage_entrypoint(
            module,
            ("run_screenplay", "run", "create_screenplay", "execute"),
            kwargs,
        )

    elif stage_name == "slot4_audio":
        module = _load_stage_module("slot4_audio")
        screenplay_path = stage_input.get("screenplay_path")

        if not screenplay_path:
            screenplay_path = str(
                _locate_latest_artifact(
                    [_stage_workspace(state, "slot2_screenplay")]
                )
                or ""
            )

        if not screenplay_path:
            raise StageExecutionError("No screenplay artifact is available")

        kwargs = {
            **common,
            "screenplay_path": screenplay_path,
            "blueprint_path": screenplay_path,
            "output_dir": str(output_dir),
        }
        result = _invoke_stage_entrypoint(
            module,
            ("run_audio", "run", "generate_audio", "execute"),
            kwargs,
        )

    elif stage_name == "slot3_visuals":
        module = _load_stage_module("slot3_visuals")
        screenplay_path = stage_input.get("screenplay_path")
        timeline_path = stage_input.get("timeline_path")

        if not screenplay_path:
            screenplay_path = str(
                _locate_latest_artifact(
                    [_stage_workspace(state, "slot2_screenplay")]
                )
                or ""
            )

        if not timeline_path:
            timeline_path = str(
                _locate_latest_artifact(
                    [_stage_workspace(state, "slot4_audio")]
                )
                or ""
            )

        if not screenplay_path or not timeline_path:
            raise StageExecutionError(
                "Screenplay and audio timeline are required for visual generation"
            )

        kwargs = {
            **common,
            "screenplay_path": screenplay_path,
            "timeline_path": timeline_path,
            "output_dir": str(output_dir),
            "device": "cpu",
            "model_name": None,
            "reference_image_path": stage_input.get("reference_image_path"),
            "batch_size": 1,
            "seed": int(stage_input.get("seed", 42)),
        }
        result = _invoke_stage_entrypoint(
            module,
            ("run_visuals", "run", "generate_visuals", "execute"),
            kwargs,
        )

    elif stage_name == "slot5_editor":
        module = _load_stage_module("slot5_editor")
        screenplay_path = stage_input.get("screenplay_path")
        timeline_path = stage_input.get("timeline_path")
        visual_manifest_path = stage_input.get("visual_manifest_path")

        if not screenplay_path:
            screenplay_path = str(
                _locate_latest_artifact(
                    [_stage_workspace(state, "slot2_screenplay")]
                )
                or ""
            )

        if not timeline_path:
            timeline_path = str(
                _locate_latest_artifact(
                    [_stage_workspace(state, "slot4_audio")]
                )
                or ""
            )

        if not visual_manifest_path:
            visual_manifest_path = str(
                _locate_latest_artifact(
                    [_stage_workspace(state, "slot3_visuals")]
                )
                or ""
            )

        if not screenplay_path or not timeline_path or not visual_manifest_path:
            raise StageExecutionError(
                "Screenplay, audio timeline, and visual manifest are required for editing"
            )

        final_output = output_dir / "final_video.mp4"
        kwargs = {
            **common,
            "screenplay_path": screenplay_path,
            "timeline_path": timeline_path,
            "visual_manifest_path": visual_manifest_path,
            "output_path": str(final_output),
            "lut_path": stage_input.get("lut_path"),
        }
        result = _invoke_stage_entrypoint(
            module,
            ("run_editor", "run", "render_video", "execute"),
            kwargs,
        )

        if not final_output.is_file():
            returned_paths = _find_artifacts(result, [output_dir])
            videos = [path for path in returned_paths if path.suffix.lower() in {".mp4", ".mov", ".mkv"}]
            if videos:
                final_output = videos[0]

    else:
        raise StageExecutionError(f"Unknown stage: {stage_name}")

    artifacts = _find_artifacts(result, [output_dir])

    if stage_name == "slot5_editor":
        videos = [
            path for path in artifacts
            if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
        ]
        if not videos:
            raise StageExecutionError(
                "Editor returned without producing a video artifact"
            )

    return {
        "status": "completed",
        "outputs": _json_safe(result) if isinstance(result, dict) else {"result": _json_safe(result)},
        "artifacts": [str(path.resolve()) for path in artifacts],
        "test_mode": False,
    }


def _resolve_stage_inputs(
    state: dict[str, Any],
    stage_name: str,
    stage_input: dict[str, Any],
) -> dict[str, Any]:
    """Resolve known predecessor artifacts and preserve explicit caller inputs."""
    resolved = dict(stage_input)
    stage_workspace_roots = {
        stage: _stage_workspace(state, stage)
        for stage in STAGE_ORDER
    }

    if stage_name == "slot2_screenplay":
        resolved.setdefault(
            "research_path",
            str(_locate_latest_artifact([stage_workspace_roots["slot1_research"]]) or ""),
        )

    elif stage_name == "slot4_audio":
        resolved.setdefault(
            "screenplay_path",
            str(_locate_latest_artifact([stage_workspace_roots["slot2_screenplay"]]) or ""),
        )

    elif stage_name == "slot3_visuals":
        resolved.setdefault(
            "screenplay_path",
            str(_locate_latest_artifact([stage_workspace_roots["slot2_screenplay"]]) or ""),
        )
        resolved.setdefault(
            "timeline_path",
            str(_locate_latest_artifact([stage_workspace_roots["slot4_audio"]]) or ""),
        )

    elif stage_name == "slot5_editor":
        resolved.setdefault(
            "screenplay_path",
            str(_locate_latest_artifact([stage_workspace_roots["slot2_screenplay"]]) or ""),
        )
        resolved.setdefault(
            "timeline_path",
            str(_locate_latest_artifact([stage_workspace_roots["slot4_audio"]]) or ""),
        )
        resolved.setdefault(
            "visual_manifest_path",
            str(_locate_latest_artifact([stage_workspace_roots["slot3_visuals"]]) or ""),
        )

    return resolved


def execute_stage(
    job_id: str,
    stage_name: str,
    stage_input: dict[str, Any],
    test_mode: bool,
) -> dict[str, Any]:
    """Execute one pipeline stage, verify output checksums, and persist its state."""
    if stage_name not in STAGE_ORDER:
        raise ValueError(
            f"Unknown stage '{stage_name}'. Valid stages: {', '.join(STAGE_ORDER)}"
        )

    with _ACTIVE_JOBS_LOCK:
        job = _ACTIVE_JOBS.get(job_id)

    if job is None:
        raise PipelineError(f"Unknown or inactive job: {job_id}")

    state_path = Path(job["state_path"])
    state = load_project_state(state_path)

    if state["status"] in {"cancelled", "failed"}:
        raise PipelineError(
            f"Cannot execute stage for a {state['status']} project"
        )

    stage_index = STAGE_ORDER.index(stage_name)
    completed = state["pipeline"].get("completed_stages", [])

    if stage_index > 0 and STAGE_ORDER[stage_index - 1] not in completed:
        raise PipelineError(
            f"Stage order violation: {stage_name} requires "
            f"{STAGE_ORDER[stage_index - 1]} to complete first"
        )

    _check_cancelled(job_id)
    stage_input = _resolve_stage_inputs(state, stage_name, stage_input or {})
    stage_record = state["stages"][stage_name]
    stage_record["status"] = "running"
    stage_record["attempts"] = int(stage_record.get("attempts", 0)) + 1
    stage_record["started_at"] = utc_now()
    stage_record["completed_at"] = None
    stage_record["error"] = None
    stage_record["input"] = _json_safe(stage_input)
    state["status"] = "running"
    state["pipeline"]["current_stage"] = stage_name
    state["pipeline"]["failed_stage"] = None
    save_project_state(state_path, state)

    start_time = time.monotonic()

    try:
        for key in (
            "research_path",
            "screenplay_path",
            "timeline_path",
            "visual_manifest_path",
        ):
            raw = stage_input.get(key)
            if raw:
                path = _resolve_path(raw)
                if not path.is_file():
                    raise StageExecutionError(
                        f"Required input artifact does not exist: {key}={path}"
                    )

        if test_mode:
            result = _run_test_stage(state, stage_name, stage_input)
        else:
            result = _run_production_stage(state, stage_name, stage_input)

        _check_cancelled(job_id)

        artifact_paths = []
        for raw_path in result.get("artifacts", []):
            path = _resolve_path(raw_path)
            if not path.is_file():
                raise StateIntegrityError(
                    f"Stage reported a missing artifact: {path}"
                )
            artifact_paths.append(path)

        if not artifact_paths:
            raise StageExecutionError(
                f"Stage {stage_name} completed without producing verifiable artifacts"
            )

        artifact_records = []
        for artifact_path in artifact_paths:
            record = _artifact_record(artifact_path)
            _verify_artifact_record(record)
            artifact_records.append(record)

        stage_record["status"] = "completed"
        stage_record["completed_at"] = utc_now()
        stage_record["duration_seconds"] = round(time.monotonic() - start_time, 3)
        stage_record["outputs"] = _json_safe(result.get("outputs", {}))
        stage_record["artifacts"] = artifact_records
        stage_record["error"] = None

        for index, artifact in enumerate(artifact_records):
            artifact_key = f"{stage_name}:{index + 1}"
            state["artifacts"][artifact_key] = artifact

        completed = state["pipeline"].setdefault("completed_stages", [])
        if stage_name not in completed:
            completed.append(stage_name)

        state["pipeline"]["completed_stages"] = [
            stage for stage in STAGE_ORDER if stage in completed
        ]
        state["pipeline"]["current_stage"] = None
        state["pipeline"]["last_error"] = None

        save_project_state(state_path, state)

        LOGGER.info(
            "Job %s: %s completed in %.3fs (%d artifacts verified)",
            job_id,
            stage_name,
            stage_record["duration_seconds"],
            len(artifact_records),
        )

        return {
            "job_id": job_id,
            "stage": stage_name,
            "status": "completed",
            "duration_seconds": stage_record["duration_seconds"],
            "artifacts": artifact_records,
            "outputs": stage_record["outputs"],
        }

    except PipelineCancelled:
        state = load_project_state(state_path)
        stage_record = state["stages"][stage_name]
        stage_record["status"] = "cancelled"
        stage_record["completed_at"] = utc_now()
        stage_record["duration_seconds"] = round(time.monotonic() - start_time, 3)
        state["status"] = "cancelled"
        state["pipeline"]["current_stage"] = None
        save_project_state(state_path, state)
        raise

    except Exception as exc:
        LOGGER.error(
            "Job %s stage %s failed: %s",
            job_id,
            stage_name,
            exc,
        )
        LOGGER.debug("%s", traceback.format_exc())

        state = load_project_state(state_path)
        stage_record = state["stages"][stage_name]
        stage_record["status"] = "failed"
        stage_record["completed_at"] = utc_now()
        stage_record["duration_seconds"] = round(time.monotonic() - start_time, 3)
        stage_record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "timestamp": utc_now(),
        }
        state["status"] = "failed"
        state["pipeline"]["current_stage"] = None
        state["pipeline"]["failed_stage"] = stage_name
        state["pipeline"]["last_error"] = stage_record["error"]
        save_project_state(state_path, state)

        if isinstance(exc, PipelineError):
            raise

        raise StageExecutionError(
            f"Stage {stage_name} failed: {exc}"
        ) from exc


def _create_test_video(
    state: dict[str, Any],
    output_path: Path,
    duration_seconds: float = 2.0,
) -> dict[str, Any]:
    """Create and validate a short FFmpeg test video without GPU dependencies."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = _assert_within(
        output_path,
        _project_workspace(state),
        "Test video output",
    )

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if not ffmpeg or not ffprobe:
        raise PipelineError(
            "Test-mode video validation requires both ffmpeg and ffprobe "
            "to be installed and available on PATH"
        )

    duration_seconds = max(1.0, min(float(duration_seconds), 2.0))

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=size=320x240:rate=24",
        "-f", "lavfi",
        "-i", "sine=frequency=440:sample_rate=44100",
        "-t", f"{duration_seconds:.3f}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "64k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("FFmpeg test-video generation timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(
            "FFmpeg test-video generation failed: "
            f"{(exc.stderr or exc.stdout or '').strip()}"
        ) from exc

    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise StateIntegrityError("FFmpeg did not produce a non-empty test video")

    probe_command = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration,size:stream=codec_type",
        "-of", "json",
        str(output_path),
    ]

    try:
        probe = subprocess.run(
            probe_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
        )
        metadata = json.loads(probe.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise StateIntegrityError(f"ffprobe could not validate test video: {exc}") from exc

    format_info = metadata.get("format", {})
    streams = metadata.get("streams", [])
    actual_duration = float(format_info.get("duration", 0))

    if actual_duration <= 0:
        raise StateIntegrityError("Test video has invalid duration")

    stream_types = {
        stream.get("codec_type")
        for stream in streams
        if isinstance(stream, dict)
    }

    if "video" not in stream_types:
        raise StateIntegrityError("Test export contains no video stream")

    if "audio" not in stream_types:
        raise StateIntegrityError("Test export contains no audio stream")

    if abs(actual_duration - duration_seconds) > 0.75:
        raise StateIntegrityError(
            f"Test video duration mismatch: expected about {duration_seconds}s, "
            f"got {actual_duration:.3f}s"
        )

    artifact = _artifact_record(output_path, "video")
    _verify_artifact_record(artifact)

    return {
        "path": str(output_path.resolve()),
        "duration_seconds": actual_duration,
        "size_bytes": output_path.stat().st_size,
        "sha256": artifact["sha256"],
        "streams": sorted(stream_types),
        "validated_at": utc_now(),
    }


def _finalize_test_export(
    state: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Create the final lightweight end-to-end test video after all stages pass."""
    workspace = _project_workspace(state)
    editor_workspace = _stage_workspace(state, "slot5_editor")
    editor_workspace.mkdir(parents=True, exist_ok=True)

    output_path = editor_workspace / "final_video.mp4"
    result = _create_test_video(state, output_path, duration_seconds=2.0)

    state = load_project_state(_get_state_path(state))
    state["artifacts"]["final_video"] = {
        "path": result["path"],
        "sha256": result["sha256"],
        "size_bytes": result["size_bytes"],
        "kind": "video",
        "duration_seconds": result["duration_seconds"],
        "verified_at": result["validated_at"],
    }

    editor_record = state["stages"]["slot5_editor"]
    editor_record["outputs"]["final_video"] = result["path"]
    editor_record["outputs"]["video_validation"] = result
    editor_record["artifacts"] = [
        artifact
        for artifact in editor_record.get("artifacts", [])
        if artifact.get("path") != result["path"]
    ]
    editor_record["artifacts"].append(
        _artifact_record(output_path, "video")
    )

    save_project_state(_get_state_path(state), state)
    return result


def recover_project(
    state_path: str | Path,
    test_mode: bool,
) -> dict[str, Any]:
    """Recover a project from persisted stage state and verify completed artifacts."""
    path = _resolve_path(state_path)
    state = load_project_state(path)

    actions: list[dict[str, Any]] = []
    completed = state["pipeline"].get("completed_stages", [])

    verified_stages = []
    invalid_stages = []

    for stage_name in STAGE_ORDER:
        record = state["stages"][stage_name]
        status = record.get("status")

        if status == "completed":
            artifacts = record.get("artifacts", [])
            try:
                if not artifacts:
                    raise StateIntegrityError(
                        f"Completed stage {stage_name} has no artifact records"
                    )

                for artifact in artifacts:
                    _verify_artifact_record(artifact)

                verified_stages.append(stage_name)
                actions.append({
                    "action": "verified",
                    "stage": stage_name,
                    "timestamp": utc_now(),
                })

            except StateIntegrityError as exc:
                invalid_stages.append(stage_name)
                actions.append({
                    "action": "invalidated",
                    "stage": stage_name,
                    "reason": str(exc),
                    "timestamp": utc_now(),
                })

        elif status == "running":
            record["status"] = "interrupted"
            record["error"] = {
                "type": "InterruptedExecution",
                "message": "The process stopped before this stage was committed.",
                "timestamp": utc_now(),
            }
            actions.append({
                "action": "marked_interrupted",
                "stage": stage_name,
                "timestamp": utc_now(),
            })

    if invalid_stages:
        first_invalid_index = min(STAGE_ORDER.index(stage) for stage in invalid_stages)
        for stage_name in STAGE_ORDER[first_invalid_index:]:
            record = state["stages"][stage_name]
            if record.get("status") == "completed":
                record["status"] = "pending"
                record["completed_at"] = None
                record["artifacts"] = []
                record["outputs"] = {}
                actions.append({
                    "action": "reset_stage",
                    "stage": stage_name,
                    "timestamp": utc_now(),
                })

        verified_stages = [
            stage for stage in verified_stages
            if STAGE_ORDER.index(stage) < first_invalid_index
        ]

    if state["status"] == "running":
        state["status"] = "recoverable"

    state["pipeline"]["completed_stages"] = [
        stage for stage in STAGE_ORDER if stage in verified_stages
    ]
    state["pipeline"]["current_stage"] = None

    state["recovery"] = {
        "recovered_at": utc_now(),
        "test_mode": bool(test_mode),
        "actions": actions,
    }

    if test_mode:
        state["test_mode"] = True

    save_project_state(path, state)

    return {
        "project_id": state["project_id"],
        "state_path": str(path),
        "status": state["status"],
        "verified_stages": verified_stages,
        "invalid_stages": invalid_stages,
        "actions": actions,
        "next_stage": next(
            (
                stage for stage in STAGE_ORDER
                if stage not in state["pipeline"]["completed_stages"]
            ),
            None,
        ),
    }


def run_pipeline(
    job_id: str,
    test_mode: bool,
) -> dict[str, Any]:
    """Run all pipeline stages in strict order and persist every transition."""
    with _ACTIVE_JOBS_LOCK:
        job = _ACTIVE_JOBS.get(job_id)

    if job is None:
        raise PipelineError(f"Unknown job: {job_id}")

    state_path = Path(job["state_path"])
    state = load_project_state(state_path)

    with _ACTIVE_JOBS_LOCK:
        job["status"] = "running"
        job["started_at"] = utc_now()
        job["test_mode"] = bool(test_mode)

    state["test_mode"] = bool(test_mode)
    state["status"] = "running"
    state["pipeline"]["started_at"] = utc_now()
    state["pipeline"]["completed_at"] = None
    state["pipeline"]["cancel_reason"] = None
    save_project_state(state_path, state)

    pipeline_start = time.monotonic()

    try:
        if test_mode:
            # Keep the CI path fully offline: no model downloads, GPU calls, or API requests.
            # FFmpeg is used only for the final short test export.
            for stage_name in STAGE_ORDER:
                _check_cancelled(job_id)
                state = load_project_state(state_path)

                if stage_name in state["pipeline"].get("completed_stages", []):
                    for artifact in state["stages"][stage_name].get("artifacts", []):
                        _verify_artifact_record(artifact)
                    continue

                stage_input = _resolve_stage_inputs(state, stage_name, {})
                execute_stage(job_id, stage_name, stage_input, test_mode=True)

            _check_cancelled(job_id)
            final_video = _finalize_test_export(state, job_id)

        else:
            for stage_name in STAGE_ORDER:
                _check_cancelled(job_id)
                state = load_project_state(state_path)

                if stage_name in state["pipeline"].get("completed_stages", []):
                    for artifact in state["stages"][stage_name].get("artifacts", []):
                        _verify_artifact_record(artifact)
                    continue

                stage_input = _resolve_stage_inputs(state, stage_name, {})
                execute_stage(job_id, stage_name, stage_input, test_mode=False)

            state = load_project_state(state_path)
            final_artifact = state.get("artifacts", {}).get("final_video")

            if final_artifact:
                _verify_artifact_record(final_artifact)
                final_video = {
                    "path": final_artifact["path"],
                    "duration_seconds": final_artifact.get("duration_seconds"),
                    "size_bytes": final_artifact.get("size_bytes"),
                    "sha256": final_artifact.get("sha256"),
                }
            else:
                editor_record = state["stages"]["slot5_editor"]
                candidates = [
                    artifact for artifact in editor_record.get("artifacts", [])
                    if Path(artifact.get("path", "")).suffix.lower()
                    in {".mp4", ".mov", ".mkv", ".webm"}
                ]
                if not candidates:
                    raise StateIntegrityError(
                        "Pipeline stages completed, but the final video artifact "
                        "is missing from the editor manifest"
                    )

                final_artifact = candidates[0]
                _verify_artifact_record(final_artifact)
                state["artifacts"]["final_video"] = final_artifact
                save_project_state(state_path, state)
                final_video = {
                    "path": final_artifact["path"],
                    "duration_seconds": final_artifact.get("duration_seconds"),
                    "size_bytes": final_artifact.get("size_bytes"),
                    "sha256": final_artifact.get("sha256"),
                }

        _check_cancelled(job_id)

        state = load_project_state(state_path)
        state["status"] = "completed"
        state["pipeline"]["current_stage"] = None
        state["pipeline"]["failed_stage"] = None
        state["pipeline"]["last_error"] = None
        state["pipeline"]["completed_at"] = utc_now()
        state["pipeline"]["duration_seconds"] = round(
            time.monotonic() - pipeline_start, 3
        )
        state["integrity"]["last_verified_at"] = utc_now()
        save_project_state(state_path, state)

        with _ACTIVE_JOBS_LOCK:
            job["status"] = "completed"
            job["completed_at"] = utc_now()
            job["duration_seconds"] = state["pipeline"]["duration_seconds"]
            job["final_video"] = final_video

        LOGGER.info(
            "Pipeline %s completed in %.3fs",
            job_id,
            state["pipeline"]["duration_seconds"],
        )

        return {
            "job_id": job_id,
            "project_id": state["project_id"],
            "status": "completed",
            "test_mode": bool(test_mode),
            "duration_seconds": state["pipeline"]["duration_seconds"],
            "completed_stages": list(STAGE_ORDER),
            "final_video": final_video,
            "state_path": str(state_path.resolve()),
        }

    except PipelineCancelled as exc:
        state = load_project_state(state_path)
        state["status"] = "cancelled"
        state["pipeline"]["current_stage"] = None
        state["pipeline"]["cancel_reason"] = str(exc)
        state["pipeline"]["completed_at"] = utc_now()
        state["pipeline"]["duration_seconds"] = round(
            time.monotonic() - pipeline_start, 3
        )
        save_project_state(state_path, state)

        with _ACTIVE_JOBS_LOCK:
            job["status"] = "cancelled"
            job["error"] = str(exc)
            job["completed_at"] = utc_now()

        raise

    except Exception as exc:
        LOGGER.error("Pipeline %s failed: %s", job_id, exc)
        LOGGER.debug("%s", traceback.format_exc())

        state = load_project_state(state_path)
        state["status"] = "failed"
        state["pipeline"]["current_stage"] = None
        state["pipeline"]["completed_at"] = utc_now()
        state["pipeline"]["duration_seconds"] = round(
            time.monotonic() - pipeline_start, 3
        )
        state["pipeline"]["last_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "timestamp": utc_now(),
        }
        save_project_state(state_path, state)

        with _ACTIVE_JOBS_LOCK:
            job["status"] = "failed"
            job["error"] = str(exc)
            job["completed_at"] = utc_now()

        raise


def cancel_pipeline(job_id: str, reason: str) -> dict[str, Any]:
    """Request cooperative cancellation of an active pipeline."""
    reason = str(reason or "Cancelled by caller").strip()

    with _ACTIVE_JOBS_LOCK:
        job = _ACTIVE_JOBS.get(job_id)

    if job is None:
        raise PipelineError(f"Unknown job: {job_id}")

    if job.get("status") in {"completed", "failed", "cancelled"}:
        return get_pipeline_status(job_id)

    event = _cancel_event(job_id)
    event.set()

    with _ACTIVE_JOBS_LOCK:
        job["cancel_reason"] = reason
        job["cancel_requested_at"] = utc_now()
        job["status"] = "cancellation_requested"

    state_path = Path(job["state_path"])
    state = load_project_state(state_path)
    state["pipeline"]["cancel_reason"] = reason
    state["pipeline"]["cancel_requested_at"] = utc_now()
    save_project_state(state_path, state)

    return get_pipeline_status(job_id)


def get_pipeline_status(job_id: str) -> dict[str, Any]:
    """Return the current in-memory job status and persisted project progress."""
    with _ACTIVE_JOBS_LOCK:
        job = _ACTIVE_JOBS.get(job_id)

    if job is None:
        raise PipelineError(f"Unknown job: {job_id}")

    state_path = Path(job["state_path"])
    state = load_project_state(state_path)

    return {
        "job_id": job_id,
        "project_id": state["project_id"],
        "status": job.get("status", state.get("status")),
        "current_stage": state["pipeline"].get("current_stage"),
        "completed_stages": list(state["pipeline"].get("completed_stages", [])),
        "stage_statuses": {
            stage: state["stages"][stage].get("status", "pending")
            for stage in STAGE_ORDER
        },
        "failed_stage": state["pipeline"].get("failed_stage"),
        "cancel_reason": job.get("cancel_reason")
        or state["pipeline"].get("cancel_reason"),
        "error": job.get("error")
        or state["pipeline"].get("last_error"),
        "state_path": str(state_path.resolve()),
        "updated_at": state.get("updated_at"),
        "test_mode": state.get("test_mode", False),
    }


def _setup_logging(verbose: bool = False) -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _register_signal_handlers(job_id: str) -> None:
    """Connect supported process signals to cooperative pipeline cancellation."""
    def handler(signum, _frame):
        try:
            cancel_pipeline(job_id, f"Received signal {signum}")
        except Exception:
            LOGGER.exception("Unable to request cancellation after signal %s", signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):
            # Signal handlers can only be installed from the main thread.
            pass


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the Autonomous AI Cinema Studio Engine pipeline."
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Topic or subject for the cinematic project.",
    )
    parser.add_argument(
        "--language",
        default="Hindi",
        help="Narration and screenplay language (default: Hindi).",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=sorted(PROFILE_DEFAULTS.keys()),
        help="Video profile: reel or documentary.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for project outputs.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run offline CI pipeline and generate a short FFmpeg test video.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    try:
        state = create_project(
            topic=args.topic,
            language=args.language,
            profile=args.profile,
            output_root=args.output_root,
            test_mode=args.test_mode,
        )

        job_id = state["project_id"]
        state_path = state["state_path"]

        with _ACTIVE_JOBS_LOCK:
            _ACTIVE_JOBS[job_id] = {
                "job_id": job_id,
                "state_path": state_path,
                "status": "created",
                "created_at": utc_now(),
                "test_mode": bool(args.test_mode),
            }

        _cancel_event(job_id)
        _register_signal_handlers(job_id)

        LOGGER.info("Starting project %s", job_id)
        LOGGER.info("State file: %s", state_path)

        result = run_pipeline(job_id, test_mode=args.test_mode)

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user")
        return 130

    except Exception as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
