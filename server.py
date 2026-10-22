# File: server.py
# Type: Python
# Purpose: FastAPI gateway for the Autonomous AI Cinema Studio Engine.

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import os
import re
import threading
import time
import uuid

from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "Autonomous AI Cinema Studio Engine API"
APP_VERSION = "1.0.0"

ROOT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = ROOT_DIR / "workspace"
OUTPUT_ROOT = WORKSPACE_DIR / "05_master_theater"
API_OUTPUT_DIR = OUTPUT_ROOT / "api_jobs"

PIPELINE_MODULE_NAME = os.getenv(
    "CINEMA_PIPELINE_MODULE",
    "master_pipeline",
)

POLL_INTERVAL_SECONDS = max(
    0.25,
    float(os.getenv("CINEMA_EVENT_POLL_INTERVAL", "0.75")),
)

MAX_EVENT_HISTORY = max(
    20,
    int(os.getenv("CINEMA_MAX_EVENT_HISTORY", "500")),
)

MAX_TOPIC_LENGTH = 500
MAX_LANGUAGE_LENGTH = 40

ALLOWED_PROFILES = {"reel", "documentary"}

ARTIFACT_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".wav",
    ".mp3",
    ".aac",
    ".flac",
    ".srt",
    ".vtt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".json",
    ".txt",
    ".pdf",
    ".ass",
}

logger = logging.getLogger("cinema_api")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Pydantic compatibility helpers
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return utc_now().isoformat()


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model across supported Pydantic versions."""
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def normalize_value(value: Any) -> Any:
    """Convert common pipeline values into JSON-compatible structures."""
    if isinstance(value, BaseModel):
        return model_to_dict(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): normalize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [normalize_value(item) for item in value]

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return str(value)


def first_value(data: Any, *keys: str, default: Any = None) -> Any:
    """Read the first available named value from a dictionary or object."""
    if isinstance(data, dict):
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return default

    for key in keys:
        value = getattr(data, key, None)
        if value is not None:
            return value

    return default


def safe_job_id(value: str) -> str:
    """Validate an externally supplied job identifier."""
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job_id format.",
        )
    return value


def ensure_output_directories() -> None:
    """Create required API output directories."""
    API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class CreateJobRequest(BaseModel):
    """Request body for creating a cinema production job."""

    topic: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TOPIC_LENGTH,
        description="Topic or brief for the requested video.",
    )

    language: str = Field(
        default="Hindi",
        min_length=2,
        max_length=MAX_LANGUAGE_LENGTH,
        description="Primary language for the generated video.",
    )

    profile: str = Field(
        default="reel",
        description="Production profile: reel or documentary.",
    )

    test_mode: bool = Field(
        default=False,
        description=(
            "Run the offline test pipeline without requiring a GPU."
        ),
    )

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        """Reject topics containing only whitespace."""
        value = value.strip()
        if not value:
            raise ValueError("topic must not be empty.")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        """Normalize the requested language."""
        value = value.strip()
        if not value:
            raise ValueError("language must not be empty.")
        return value

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        """Validate the production profile."""
        value = value.strip().lower()
        if value not in ALLOWED_PROFILES:
            raise ValueError(
                "profile must be either 'reel' or 'documentary'."
            )
        return value


class ArtifactResponse(BaseModel):
    """Description of a generated artifact."""

    name: str
    path: str
    size_bytes: int
    media_type: str
    modified_at: Optional[str] = None


class JobResponse(BaseModel):
    """Response returned when a job is accepted."""

    job_id: str
    status: str
    topic: str
    language: str
    profile: str
    test_mode: bool
    created_at: str
    status_url: str
    events_url: str


class JobStatusResponse(BaseModel):
    """Current job status and pipeline progress."""

    job_id: str
    status: str
    stage: Optional[str] = None
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    topic: Optional[str] = None
    language: Optional[str] = None
    profile: Optional[str] = None
    test_mode: bool = False
    message: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    artifacts: List[ArtifactResponse] = Field(default_factory=list)


class CancelJobResponse(BaseModel):
    """Response returned after requesting job cancellation."""

    job_id: str
    status: str
    message: str
    updated_at: str


class JobArtifactsResponse(BaseModel):
    """Artifact listing for a completed or running job."""

    job_id: str
    status: str
    artifacts: List[ArtifactResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Basic liveness response."""

    status: str
    service: str
    version: str
    timestamp: str


class ReadyResponse(BaseModel):
    """Readiness response describing required components."""

    status: str
    service: str
    pipeline_available: bool
    output_directory_writable: bool
    timestamp: str
    details: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# In-memory job registry and event stream
# ---------------------------------------------------------------------------

REGISTRY_LOCK = threading.RLock()

JOBS: Dict[str, Dict[str, Any]] = {}
EVENT_HISTORY: Dict[str, deque] = {}
JOB_CANCEL_EVENTS: Dict[str, threading.Event] = {}
PIPELINE_LOCKS: Dict[str, threading.Lock] = {}

PIPELINE_MODULE: Any = None
PIPELINE_IMPORT_ERROR: Optional[str] = None

TERMINAL_STATES = {
    "completed",
    "failed",
    "cancelled",
}

ACTIVE_STATES = {
    "queued",
    "starting",
    "running",
    "cancelling",
}


# ---------------------------------------------------------------------------
# Pipeline module loading
# ---------------------------------------------------------------------------

def load_pipeline_module() -> Any:
    """Load the configured master pipeline module."""
    global PIPELINE_MODULE
    global PIPELINE_IMPORT_ERROR

    if PIPELINE_MODULE is not None:
        return PIPELINE_MODULE

    try:
        PIPELINE_MODULE = importlib.import_module(PIPELINE_MODULE_NAME)
        PIPELINE_IMPORT_ERROR = None
        logger.info(
            "Loaded pipeline module: %s",
            PIPELINE_MODULE_NAME,
        )
        return PIPELINE_MODULE

    except Exception as exc:
        PIPELINE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed to load the master pipeline.")
        raise RuntimeError(PIPELINE_IMPORT_ERROR) from exc


def pipeline_function(name: str) -> Any:
    """Return a required callable from the master pipeline."""
    module = load_pipeline_module()
    function = getattr(module, name, None)

    if not callable(function):
        raise RuntimeError(
            f"Pipeline function '{name}' is missing or not callable."
        )

    return function


async def invoke_pipeline(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute a synchronous pipeline function away from the event loop."""
    if inspect.iscoroutinefunction(function):
        return await function(*args, **kwargs)

    return await asyncio.to_thread(
        lambda: function(*args, **kwargs)
    )


# ---------------------------------------------------------------------------
# Job state and event helpers
# ---------------------------------------------------------------------------

def append_event(
    job_id: str,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append a structured telemetry event to the job history."""
    event = {
        "event": event_type,
        "job_id": job_id,
        "timestamp": utc_iso(),
        "data": normalize_value(payload or {}),
    }

    with REGISTRY_LOCK:
        history = EVENT_HISTORY.setdefault(
            job_id,
            deque(maxlen=MAX_EVENT_HISTORY),
        )
        history.append(event)

    return event


def update_job(
    job_id: str,
    **updates: Any,
) -> Optional[Dict[str, Any]]:
    """Update a registered job and emit a state-change event."""
    with REGISTRY_LOCK:
        job = JOBS.get(job_id)

        if job is None:
            return None

        job.update(updates)
        job["updated_at"] = utc_iso()
        snapshot = dict(job)

    append_event(
        job_id,
        "job_update",
        {
            key: value
            for key, value in snapshot.items()
            if key not in {"cancel_event"}
        },
    )

    return snapshot


def get_job_record(job_id: str) -> Optional[Dict[str, Any]]:
    """Return a shallow copy of a registered job."""
    with REGISTRY_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        return dict(job)


def get_event_snapshot(job_id: str) -> List[Dict[str, Any]]:
    """Return a stable snapshot of recent job events."""
    with REGISTRY_LOCK:
        history = EVENT_HISTORY.get(job_id)
        if history is None:
            return []
        return list(history)


def set_terminal_state(
    job_id: str,
    state: str,
    message: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Mark a job as terminal and publish the final event."""
    timestamp = utc_iso()

    updates: Dict[str, Any] = {
        "status": state,
        "completed_at": timestamp,
        "updated_at": timestamp,
    }

    if message is not None:
        updates["message"] = message

    if error is not None:
        updates["error"] = error

    update_job(job_id, **updates)

    append_event(
        job_id,
        "job_finished",
        {
            "status": state,
            "message": message,
            "error": error,
        },
    )


# ---------------------------------------------------------------------------
# Pipeline result normalization
# ---------------------------------------------------------------------------

def normalize_status(value: Any) -> str:
    """Map pipeline status values to the API's status vocabulary."""
    if value is None:
        return "running"

    raw = str(value).strip().lower()

    if raw in {"success", "successful", "done", "complete"}:
        return "completed"

    if raw in {"error", "exception"}:
        return "failed"

    if raw in {"canceled", "aborted"}:
        return "cancelled"

    if raw in {
        "queued",
        "starting",
        "running",
        "cancelling",
        "completed",
        "failed",
        "cancelled",
    }:
        return raw

    return "running"


def extract_progress(data: Any) -> float:
    """Extract a numeric progress percentage from pipeline data."""
    value = first_value(
        data,
        "progress",
        "progress_percent",
        "percentage",
        "percent",
        default=0.0,
    )

    try:
        progress = float(value)
    except (TypeError, ValueError):
        progress = 0.0

    if 0.0 <= progress <= 1.0:
        progress *= 100.0

    return max(0.0, min(100.0, progress))


def extract_artifact_paths(data: Any) -> List[str]:
    """Extract artifact paths from common pipeline result shapes."""
    paths: List[str] = []

    if isinstance(data, (str, Path)):
        candidate = str(data)
        if Path(candidate).suffix.lower() in ARTIFACT_EXTENSIONS:
            paths.append(candidate)
        return paths

    if isinstance(data, (list, tuple, set)):
        for item in data:
            paths.extend(extract_artifact_paths(item))
        return paths

    if not isinstance(data, dict):
        if isinstance(data, BaseModel):
            data = model_to_dict(data)
        else:
            return paths

    path_keys = {
        "path",
        "file",
        "file_path",
        "output_path",
        "artifact_path",
        "video_path",
    }

    for key, value in data.items():
        if key.lower() in path_keys and isinstance(
            value,
            (str, Path),
        ):
            paths.append(str(value))
        elif isinstance(value, (dict, list, tuple, set)):
            paths.extend(extract_artifact_paths(value))

    return paths


def resolve_project_root(create_result: Any) -> Optional[Path]:
    """Discover the project directory from create_project output."""
    value = first_value(
        create_result,
        "project_dir",
        "project_path",
        "workspace_path",
        "output_dir",
        "directory",
        "root_dir",
    )

    if value is None:
        state_path = first_value(
            create_result,
            "state_path",
            "project_state_path",
        )
        if state_path is not None:
            value = Path(str(state_path)).parent

    if value is None:
        return None

    try:
        path = Path(str(value)).expanduser().resolve()
    except (OSError, ValueError):
        return None

    return path if path.exists() else None


def resolve_pipeline_job_id(create_result: Any) -> Optional[str]:
    """Read the pipeline's own identifier from project creation output."""
    value = first_value(
        create_result,
        "job_id",
        "project_id",
        "id",
        "project_key",
    )

    if value is None:
        return None

    identifier = str(value).strip()

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
        return None

    return identifier


# ---------------------------------------------------------------------------
# Artifact discovery and safe path handling
# ---------------------------------------------------------------------------

def is_within_directory(path: Path, root: Path) -> bool:
    """Check whether a resolved path is inside a trusted root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def media_type_for(path: Path) -> str:
    """Return a conservative media type for an artifact."""
    mapping = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".srt": "application/x-subrip",
        ".vtt": "text/vtt",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".json": "application/json",
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".ass": "text/plain",
    }
    return mapping.get(
        path.suffix.lower(),
        "application/octet-stream",
    )


def artifact_record(path: Path, project_root: Path) -> Optional[ArtifactResponse]:
    """Build an artifact response for an allowed file."""
    try:
        resolved = path.resolve(strict=True)

        if not resolved.is_file():
            return None

        if not is_within_directory(resolved, project_root):
            return None

        if resolved.suffix.lower() not in ARTIFACT_EXTENSIONS:
            return None

        stat_result = resolved.stat()

        return ArtifactResponse(
            name=resolved.name,
            path=str(resolved),
            size_bytes=stat_result.st_size,
            media_type=media_type_for(resolved),
            modified_at=datetime.fromtimestamp(
                stat_result.st_mtime,
                timezone.utc,
            ).isoformat(),
        )

    except (OSError, ValueError):
        return None


def discover_artifacts(job_id: str) -> List[ArtifactResponse]:
    """Discover artifacts for a job without leaving its trusted directory."""
    job = get_job_record(job_id)

    if job is None:
        return []

    project_root_value = job.get("project_root")
    paths: List[Path] = []

    if project_root_value:
        project_root = Path(project_root_value).resolve()

        if project_root.exists():
            for candidate in project_root.rglob("*"):
                if not candidate.is_file():
                    continue

                if candidate.name == "project_state.json":
                    continue

                paths.append(candidate)
    else:
        project_root = OUTPUT_ROOT.resolve()

    for raw_path in job.get("artifact_paths", []):
        try:
            candidate = Path(raw_path).expanduser().resolve()
        except (OSError, ValueError):
            continue

        if is_within_directory(candidate, project_root):
            paths.append(candidate)

    unique: Dict[str, ArtifactResponse] = {}

    for candidate in paths:
        record = artifact_record(candidate, project_root)
        if record is not None:
            unique[record.path] = record

    return sorted(
        unique.values(),
        key=lambda item: (
            item.modified_at or "",
            item.name.lower(),
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Pipeline status synchronization
# ---------------------------------------------------------------------------

async def refresh_pipeline_status(job_id: str) -> Dict[str, Any]:
    """Refresh the API registry using master_pipeline status data."""
    job = get_job_record(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    if job["status"] in TERMINAL_STATES:
        return job

    pipeline_id = job.get("pipeline_job_id") or job_id

    try:
        status_function = pipeline_function("get_pipeline_status")
        raw_status = await invoke_pipeline(
            status_function,
            pipeline_id,
        )

        if raw_status is None:
            return job

        raw_status = normalize_value(raw_status)

        status_value = normalize_status(
            first_value(
                raw_status,
                "status",
                "state",
                default=job["status"],
            )
        )

        stage = first_value(
            raw_status,
            "stage",
            "current_stage",
            "stage_name",
        )

        progress = extract_progress(raw_status)

        message = first_value(
            raw_status,
            "message",
            "detail",
            "description",
        )

        error = first_value(
            raw_status,
            "error",
            "error_message",
            "failure_reason",
        )

        artifact_paths = extract_artifact_paths(raw_status)

        updates: Dict[str, Any] = {
            "status": status_value,
            "stage": str(stage) if stage is not None else job.get("stage"),
            "progress": progress,
        }

        if message is not None:
            updates["message"] = str(message)

        if error is not None:
            updates["error"] = str(error)

        if artifact_paths:
            updates["artifact_paths"] = list(
                dict.fromkeys(
                    job.get("artifact_paths", []) + artifact_paths
                )
            )

        if status_value == "running" and not job.get("started_at"):
            updates["started_at"] = utc_iso()

        if status_value in TERMINAL_STATES:
            updates["completed_at"] = utc_iso()

        updated = update_job(job_id, **updates) or job

        if status_value in TERMINAL_STATES:
            append_event(
                job_id,
                "pipeline_status",
                {
                    "status": status_value,
                    "stage": updates.get("stage"),
                    "progress": progress,
                    "error": error,
                },
            )

        return updated

    except HTTPException:
        raise

    except Exception as exc:
        logger.warning(
            "Unable to refresh status for job %s: %s",
            job_id,
            exc,
        )
        return job


# ---------------------------------------------------------------------------
# Background job execution
# ---------------------------------------------------------------------------

async def execute_job(job_id: str) -> None:
    """Create the project and run its pipeline in the background."""
    job = get_job_record(job_id)

    if job is None:
        return

    cancel_event = JOB_CANCEL_EVENTS[job_id]

    update_job(
        job_id,
        status="starting",
        message="Preparing the production workspace.",
        started_at=utc_iso(),
    )

    append_event(
        job_id,
        "job_started",
        {
            "test_mode": job["test_mode"],
            "profile": job["profile"],
        },
    )

    try:
        create_function = pipeline_function("create_project")

        create_result = await invoke_pipeline(
            create_function,
            topic=job["topic"],
            language=job["language"],
            profile=job["profile"],
            output_root=str(OUTPUT_ROOT),
            test_mode=job["test_mode"],
        )

        if create_result is None:
            raise RuntimeError(
                "create_project returned no project information."
            )

        normalized_result = normalize_value(create_result)
        pipeline_job_id = resolve_pipeline_job_id(normalized_result)
        project_root = resolve_project_root(normalized_result)

        if pipeline_job_id is None:
            raise RuntimeError(
                "Cannot determine the pipeline job identifier from "
                "create_project output. Ensure it returns a job_id or "
                "project_id that get_pipeline_status and run_pipeline "
                "accept."
            )

        if project_root is None:
            logger.warning(
                "Project root was not discoverable for job %s. "
                "Artifact discovery will use the configured output root.",
                job_id,
            )

        update_job(
            job_id,
            pipeline_job_id=pipeline_job_id,
            project_root=str(project_root) if project_root else None,
            project_info=normalized_result,
            status="running",
            stage="slot1_research",
            progress=0.0,
            message="Production pipeline started.",
        )

        append_event(
            job_id,
            "project_created",
            {
                "pipeline_job_id": pipeline_job_id,
                "project_root": str(project_root) if project_root else None,
            },
        )

        if cancel_event.is_set():
            await cancel_pipeline_job(
                job_id,
                "Cancellation requested before pipeline execution.",
            )
            return

        run_function = pipeline_function("run_pipeline")

        append_event(
            job_id,
            "pipeline_execution_started",
            {
                "pipeline_job_id": pipeline_job_id,
                "test_mode": job["test_mode"],
            },
        )

        run_result = await invoke_pipeline(
            run_function,
            pipeline_job_id,
            job["test_mode"],
        )

        normalized_run_result = normalize_value(run_result)
        result_paths = extract_artifact_paths(normalized_run_result)

        if result_paths:
            current_job = get_job_record(job_id) or {}
            merged_paths = list(
                dict.fromkeys(
                    current_job.get("artifact_paths", []) + result_paths
                )
            )
            update_job(job_id, artifact_paths=merged_paths)

        if cancel_event.is_set():
            await cancel_pipeline_job(
                job_id,
                "Cancellation requested during pipeline execution.",
            )
            return

        latest = await refresh_pipeline_status(job_id)

        if latest["status"] == "failed":
            set_terminal_state(
                job_id,
                "failed",
                message=latest.get("message") or "Pipeline failed.",
                error=latest.get("error") or "Pipeline reported failure.",
            )
            return

        if latest["status"] == "cancelled":
            set_terminal_state(
                job_id,
                "cancelled",
                message=latest.get("message") or "Pipeline cancelled.",
            )
            return

        if latest["status"] != "completed":
            result_status = normalize_status(
                first_value(
                    normalized_run_result,
                    "status",
                    "state",
                    default="completed",
                )
            )

            if result_status == "failed":
                set_terminal_state(
                    job_id,
                    "failed",
                    message="Pipeline execution failed.",
                    error=str(
                        first_value(
                            normalized_run_result,
                            "error",
                            "error_message",
                            default="Unknown pipeline error.",
                        )
                    ),
                )
                return

            if result_status == "cancelled":
                set_terminal_state(
                    job_id,
                    "cancelled",
                    message="Pipeline cancelled.",
                )
                return

        update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100.0,
            message="Production pipeline completed successfully.",
            pipeline_result=normalized_run_result,
        )

        set_terminal_state(
            job_id,
            "completed",
            message="Production pipeline completed successfully.",
        )

    except asyncio.CancelledError:
        logger.info("Async task cancelled for job %s.", job_id)

        try:
            await cancel_pipeline_job(
                job_id,
                "API background task was cancelled.",
            )
        except Exception:
            logger.exception(
                "Failed to propagate cancellation for job %s.",
                job_id,
            )

        raise

    except Exception as exc:
        logger.exception("Job %s failed.", job_id)

        set_terminal_state(
            job_id,
            "failed",
            message="Production pipeline failed.",
            error=f"{type(exc).__name__}: {exc}",
        )


async def cancel_pipeline_job(job_id: str, reason: str) -> None:
    """Request cancellation from master_pipeline and update API state."""
    job = get_job_record(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    if job["status"] in TERMINAL_STATES:
        return

    cancel_event = JOB_CANCEL_EVENTS.get(job_id)
    if cancel_event is not None:
        cancel_event.set()

    update_job(
        job_id,
        status="cancelling",
        message=reason,
        cancel_reason=reason,
    )

    append_event(
        job_id,
        "cancellation_requested",
        {"reason": reason},
    )

    pipeline_id = job.get("pipeline_job_id") or job_id

    try:
        cancel_function = pipeline_function("cancel_pipeline")

        await invoke_pipeline(
            cancel_function,
            pipeline_id,
            reason,
        )

    except Exception as exc:
        logger.exception(
            "Pipeline cancellation call failed for job %s.",
            job_id,
        )

        update_job(
            job_id,
            cancellation_error=f"{type(exc).__name__}: {exc}",
        )

        # The API does not claim that the pipeline stopped if the
        # pipeline's cancellation function failed.
        return

    latest = await refresh_pipeline_status(job_id)

    if latest["status"] in TERMINAL_STATES:
        if latest["status"] == "completed":
            set_terminal_state(
                job_id,
                "completed",
                message="The pipeline completed before cancellation took effect.",
            )
        elif latest["status"] == "failed":
            set_terminal_state(
                job_id,
                "failed",
                message=latest.get("message") or "Pipeline failed.",
                error=latest.get("error"),
            )
        else:
            set_terminal_state(
                job_id,
                "cancelled",
                message=reason,
            )


async def monitor_job(job_id: str) -> None:
    """Poll pipeline status while a job is active."""
    while True:
        job = get_job_record(job_id)

        if job is None or job["status"] in TERMINAL_STATES:
            return

        try:
            await refresh_pipeline_status(job_id)
        except Exception:
            logger.exception(
                "Status monitor failed for job %s.",
                job_id,
            )

        job = get_job_record(job_id)

        if job is None or job["status"] in TERMINAL_STATES:
            return

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize required directories and validate pipeline availability."""
    ensure_output_directories()

    try:
        load_pipeline_module()
    except Exception:
        logger.exception(
            "Pipeline module is unavailable during startup. "
            "Liveness will remain available; readiness will report failure."
        )

    application.state.job_tasks = {}
    application.state.monitor_tasks = {}

    yield

    tasks = []

    for task in application.state.job_tasks.values():
        if not task.done():
            task.cancel()
            tasks.append(task)

    for task in application.state.monitor_tasks.values():
        if not task.done():
            task.cancel()
            tasks.append(task)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "API gateway for creating, monitoring, cancelling, and streaming "
        "Autonomous AI Cinema Studio Engine jobs."
    ),
    lifespan=lifespan,
)

# Configure explicit origins in production through CINEMA_CORS_ORIGINS.
# Wildcard origins are retained as a development default.
cors_origins = [
    item.strip()
    for item in os.getenv(
        "CINEMA_CORS_ORIGINS",
        "*",
    ).split(",")
    if item.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request,
    exc: RequestValidationError,
):
    """Return a consistent response for invalid request bodies."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "The request did not pass validation.",
            "details": normalize_value(exc.errors()),
        },
    )


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
)
async def health() -> HealthResponse:
    """Return liveness information without requiring pipeline readiness."""
    return HealthResponse(
        status="ok",
        service=APP_NAME,
        version=APP_VERSION,
        timestamp=utc_iso(),
    )


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["system"],
)
async def ready() -> ReadyResponse:
    """Report whether the gateway can accept pipeline jobs."""
    pipeline_available = False
    output_writable = False
    details: Dict[str, Any] = {}

    try:
        module = load_pipeline_module()

        required_functions = [
            "create_project",
            "run_pipeline",
            "get_pipeline_status",
            "cancel_pipeline",
        ]

        missing_functions = [
            name
            for name in required_functions
            if not callable(getattr(module, name, None))
        ]

        pipeline_available = not missing_functions

        details["pipeline_module"] = PIPELINE_MODULE_NAME
        details["missing_functions"] = missing_functions

    except Exception as exc:
        details["pipeline_error"] = (
            PIPELINE_IMPORT_ERROR or f"{type(exc).__name__}: {exc}"
        )

    try:
        ensure_output_directories()
        probe = API_OUTPUT_DIR / f".ready-{uuid.uuid4().hex}.tmp"
        probe.write_text("ready", encoding="utf-8")
        probe.unlink(missing_ok=True)
        output_writable = True

    except OSError as exc:
        details["output_directory_error"] = str(exc)

    is_ready = pipeline_available and output_writable

    return ReadyResponse(
        status="ready" if is_ready else "not_ready",
        service=APP_NAME,
        pipeline_available=pipeline_available,
        output_directory_writable=output_writable,
        timestamp=utc_iso(),
        details=details,
    )


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------

@app.post(
    "/v1/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["jobs"],
)
async def create_job(
    request: CreateJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Create a production project and schedule its pipeline."""
    del background_tasks  # Execution is managed by tracked asyncio tasks.

    try:
        load_pipeline_module()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pipeline is unavailable: {exc}",
        ) from exc

    ensure_output_directories()

    external_job_id = uuid.uuid4().hex

    record = {
        "job_id": external_job_id,
        "pipeline_job_id": None,
        "status": "queued",
        "topic": request.topic,
        "language": request.language,
        "profile": request.profile,
        "test_mode": request.test_mode,
        "progress": 0.0,
        "stage": None,
        "message": "Job accepted and queued.",
        "error": None,
        "created_at": utc_iso(),
        "started_at": None,
        "updated_at": utc_iso(),
        "completed_at": None,
        "project_root": None,
        "project_info": None,
        "pipeline_result": None,
        "artifact_paths": [],
        "cancel_reason": None,
        "cancel_event": None,
    }

    with REGISTRY_LOCK:
        JOBS[external_job_id] = record
        EVENT_HISTORY[external_job_id] = deque(
            maxlen=MAX_EVENT_HISTORY
        )
        JOB_CANCEL_EVENTS[external_job_id] = threading.Event()
        PIPELINE_LOCKS[external_job_id] = threading.Lock()

    append_event(
        external_job_id,
        "job_queued",
        {
            "topic": request.topic,
            "language": request.language,
            "profile": request.profile,
            "test_mode": request.test_mode,
        },
    )

    task = asyncio.create_task(execute_job(external_job_id))
    app.state.job_tasks[external_job_id] = task

    monitor_task = asyncio.create_task(monitor_job(external_job_id))
    app.state.monitor_tasks[external_job_id] = monitor_task

    return JobResponse(
        job_id=external_job_id,
        status="queued",
        topic=request.topic,
        language=request.language,
        profile=request.profile,
        test_mode=request.test_mode,
        created_at=record["created_at"],
        status_url=f"/v1/jobs/{external_job_id}",
        events_url=f"/v1/jobs/{external_job_id}/events",
    )


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["jobs"],
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Return current status, stage progress, and discovered artifacts."""
    job_id = safe_job_id(job_id)

    job = get_job_record(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    if job["status"] not in TERMINAL_STATES:
        job = await refresh_pipeline_status(job_id)

    artifacts = discover_artifacts(job_id)

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        stage=job.get("stage"),
        progress=extract_progress(job),
        topic=job.get("topic"),
        language=job.get("language"),
        profile=job.get("profile"),
        test_mode=bool(job.get("test_mode", False)),
        message=job.get("message"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        updated_at=job.get("updated_at"),
        completed_at=job.get("completed_at"),
        error=job.get("error"),
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Job cancellation
# ---------------------------------------------------------------------------

@app.post(
    "/v1/jobs/{job_id}/cancel",
    response_model=CancelJobResponse,
    tags=["jobs"],
)
async def cancel_job(job_id: str) -> CancelJobResponse:
    """Request cooperative cancellation through master_pipeline."""
    job_id = safe_job_id(job_id)

    job = get_job_record(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    if job["status"] in TERMINAL_STATES:
        return CancelJobResponse(
            job_id=job_id,
            status=job["status"],
            message="Job is already in a terminal state.",
            updated_at=job.get("updated_at") or utc_iso(),
        )

    reason = "Cancellation requested through the API."

    await cancel_pipeline_job(job_id, reason)

    latest = get_job_record(job_id) or job

    if latest["status"] == "cancelling":
        message = (
            "Cancellation was requested. The pipeline may take time "
            "to reach a safe cancellation point."
        )
    elif latest["status"] == "cancelled":
        message = "Job cancelled."
    elif latest["status"] == "completed":
        message = "Job completed before cancellation took effect."
    elif latest["status"] == "failed":
        message = "The pipeline failed while cancellation was requested."
    else:
        message = "Cancellation request processed."

    return CancelJobResponse(
        job_id=job_id,
        status=latest["status"],
        message=message,
        updated_at=latest.get("updated_at") or utc_iso(),
    )


# ---------------------------------------------------------------------------
# Job artifacts
# ---------------------------------------------------------------------------

@app.get(
    "/v1/jobs/{job_id}/artifacts",
    response_model=JobArtifactsResponse,
    tags=["jobs"],
)
async def get_job_artifacts(job_id: str) -> JobArtifactsResponse:
    """List files produced by a job."""
    job_id = safe_job_id(job_id)

    job = get_job_record(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    if job["status"] not in TERMINAL_STATES:
        await refresh_pipeline_status(job_id)

    return JobArtifactsResponse(
        job_id=job_id,
        status=(get_job_record(job_id) or job)["status"],
        artifacts=discover_artifacts(job_id),
    )


# ---------------------------------------------------------------------------
# WebSocket live telemetry
# ---------------------------------------------------------------------------

@app.websocket("/v1/jobs/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    """Stream job state changes and progress telemetry over WebSocket."""
    try:
        job_id = safe_job_id(job_id)
    except HTTPException:
        await websocket.close(code=1008, reason="Invalid job_id.")
        return

    job = get_job_record(job_id)

    if job is None:
        await websocket.close(code=1008, reason="Job not found.")
        return

    await websocket.accept()

    await websocket.send_json(
        {
            "event": "connection_established",
            "job_id": job_id,
            "timestamp": utc_iso(),
            "data": {
                "status": job["status"],
                "progress": job.get("progress", 0.0),
            },
        }
    )

    sent_signatures = set()
    last_snapshot_signature = None

    try:
        while True:
            current_job = get_job_record(job_id)

            if current_job is None:
                await websocket.send_json(
                    {
                        "event": "job_unavailable",
                        "job_id": job_id,
                        "timestamp": utc_iso(),
                        "data": {},
                    }
                )
                break

            # Send new events from the retained event history.
            for event in get_event_snapshot(job_id):
                signature = (
                    event.get("timestamp"),
                    event.get("event"),
                    json.dumps(
                        event.get("data", {}),
                        sort_keys=True,
                        ensure_ascii=False,
                        default=str,
                    ),
                )

                if signature in sent_signatures:
                    continue

                sent_signatures.add(signature)

                await websocket.send_json(event)

                if len(sent_signatures) > MAX_EVENT_HISTORY * 2:
                    sent_signatures.clear()

            # Send a periodic snapshot so reconnecting clients can
            # recover the latest state even if an event was missed.
            snapshot = {
                "event": "job_snapshot",
                "job_id": job_id,
                "timestamp": utc_iso(),
                "data": {
                    "status": current_job["status"],
                    "stage": current_job.get("stage"),
                    "progress": current_job.get("progress", 0.0),
                    "message": current_job.get("message"),
                    "error": current_job.get("error"),
                    "test_mode": current_job.get("test_mode", False),
                },
            }

            signature = json.dumps(
                snapshot["data"],
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )

            if signature != last_snapshot_signature:
                await websocket.send_json(snapshot)
                last_snapshot_signature = signature

            if current_job["status"] in TERMINAL_STATES:
                await websocket.send_json(
                    {
                        "event": "stream_finished",
                        "job_id": job_id,
                        "timestamp": utc_iso(),
                        "data": {
                            "status": current_job["status"],
                            "progress": current_job.get("progress", 0.0),
                        },
                    }
                )
                break

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    except WebSocketDisconnect:
        logger.info(
            "WebSocket client disconnected for job %s.",
            job_id,
        )

    except Exception:
        logger.exception(
            "WebSocket stream failed for job %s.",
            job_id,
        )

        try:
            await websocket.close(code=1011, reason="Telemetry stream error.")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get("/", tags=["system"])
async def root() -> Dict[str, Any]:
    """Expose basic service metadata and API documentation locations."""
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "health": "/health",
        "readiness": "/ready",
        "openapi": "/openapi.json",
        "documentation": "/docs",
    }


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("CINEMA_API_HOST", "127.0.0.1")
    port = int(os.getenv("CINEMA_API_PORT", "8000"))

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
