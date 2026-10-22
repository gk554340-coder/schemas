# File: slot5_editor.py
# Code type: Python
# Autonomous AI Cinema Studio Engine — Slot 5: Master Theater
# Safe FFmpeg orchestration, audio mixing, cinematic grading, subtitles,
# export integrity checks, NVENC support, and CPU fallback.

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import wave

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
MASTER_ROOT = (PROJECT_ROOT / "workspace" / "05_master_theater").resolve()
RESEARCH_ROOT = (PROJECT_ROOT / "workspace" / "01_research_vault").resolve()
SCREENPLAY_ROOT = (PROJECT_ROOT / "workspace" / "02_screenplay_desk").resolve()
AUDIO_ROOT = (PROJECT_ROOT / "workspace" / "03_audio_studio").resolve()
VISUAL_ROOT = (PROJECT_ROOT / "workspace" / "04_visual_lab").resolve()

MANIFEST_VERSION = "1.0.0"
DEFAULT_FPS = 24
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_AUDIO_BITRATE = "192k"
DEFAULT_VIDEO_BITRATE = "6000k"
DEFAULT_TOLERANCE_SECONDS = 0.75
MIN_TEST_DURATION_SECONDS = 1.0
MAX_TEST_DURATION_SECONDS = 2.0

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
SUPPORTED_SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass"}
SUPPORTED_LUT_EXTENSIONS = {".cube", ".3dl", ".dat", ".png", ".jpg", ".jpeg"}

class EditorPipelineError(RuntimeError):
    """Raised when editing, mixing, grading, or export operations fail."""


class EditorSandboxError(EditorPipelineError):
    """Raised when a path violates a workspace boundary."""


class EditorValidationError(EditorPipelineError):
    """Raised when media inputs or final exports fail validation."""


@dataclass
class MediaAsset:
    """Normalized metadata for one media asset."""

    asset_id: str
    asset_type: str
    path: str
    relative_path: str
    sha256: str
    size_bytes: int
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ExportValidation:
    """Integrity and technical validation results for a rendered file."""

    valid: bool
    output_path: str
    file_exists: bool
    size_bytes: int
    sha256: str | None
    expected_duration_seconds: float
    actual_duration_seconds: float | None
    duration_difference_seconds: float | None
    tolerance_seconds: float
    has_video_stream: bool
    has_audio_stream: bool
    errors: list[str]
    warnings: list[str]
    checked_at_unix: int


def _is_within(path: Path, root: Path) -> bool:
    """Return whether the resolved path is inside the given root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _safe_path(
    path: str | Path,
    root: Path,
    *,
    must_exist: bool = False,
    allow_root: bool = False,
) -> Path:
    """Resolve a path and reject traversal outside its permitted sandbox."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()

    if not _is_within(candidate, root):
        raise EditorSandboxError(
            f"Path escapes permitted sandbox '{root}': {candidate}"
        )
    if not allow_root and candidate == root.resolve():
        raise EditorSandboxError("The sandbox root itself is not a file path.")
    if must_exist and not candidate.exists():
        raise FileNotFoundError(str(candidate))
    return candidate


def _ensure_master_dirs() -> None:
    """Create managed folders within the master theater sandbox."""
    MASTER_ROOT.mkdir(parents=True, exist_ok=True)
    for folder in (
        "exports",
        "audio_mix",
        "subtitles",
        "graded",
        "manifests",
        "temporary",
        "logs",
    ):
        (MASTER_ROOT / folder).mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    """Convert common Python model objects into JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="json")
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes to a file in the master theater sandbox."""
    path = _safe_path(path, MASTER_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically serialize a JSON manifest."""
    data = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")
    _atomic_write_bytes(path, data)


def _sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 digest in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_dump(value: Any) -> dict[str, Any]:
    """Normalize dictionaries, model objects, and dataclasses."""
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="json")
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    raise EditorValidationError(
        f"Unsupported input object: {type(value).__name__}"
    )


def _first_value(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first non-null value from the provided field names."""
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return default


def _as_float(value: Any, field_name: str, minimum: float = 0.0) -> float:
    """Parse and validate a finite numeric value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EditorValidationError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise EditorValidationError(
            f"{field_name} must be finite and >= {minimum}."
        )
    return parsed


def _as_int(value: Any, field_name: str, minimum: int = 0) -> int:
    """Parse and validate an integer value."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EditorValidationError(f"{field_name} must be an integer.") from exc
    if parsed < minimum:
        raise EditorValidationError(f"{field_name} must be >= {minimum}.")
    return parsed


def _resolve_input_path(
    raw_path: str | Path,
    allowed_roots: Sequence[Path],
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve an input file and require it to belong to an allowed root."""
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        possible = [(root / candidate).resolve() for root in allowed_roots]
        existing = next((path for path in possible if path.exists()), None)
        resolved = existing if existing is not None else possible[0]

    if not any(_is_within(resolved, root) for root in allowed_roots):
        raise EditorSandboxError(
            f"Input path is outside permitted project workspaces: {resolved}"
        )
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


def _resolve_screenplay_path(path: str | Path) -> Path:
    """Resolve the read-only screenplay input."""
    return _resolve_input_path(path, (SCREENPLAY_ROOT, PROJECT_ROOT))


def _resolve_timeline_path(path: str | Path) -> Path:
    """Resolve the read-only audio timeline input."""
    return _resolve_input_path(path, (AUDIO_ROOT, PROJECT_ROOT))


def _resolve_visual_manifest_path(path: str | Path) -> Path:
    """Resolve the read-only visual manifest input."""
    return _resolve_input_path(path, (VISUAL_ROOT, PROJECT_ROOT))


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject malformed or non-object JSON."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EditorValidationError(f"Invalid JSON file '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise EditorValidationError(f"Expected a JSON object in '{path}'.")
    return data


def _run_process(
    args: Sequence[str],
    *,
    timeout_seconds: int = 3600,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a process using a structured argument list, never a shell string."""
    if not args or not all(isinstance(arg, str) for arg in args):
        raise EditorValidationError("Process arguments must be a non-empty string list.")
    try:
        result = subprocess.run(
            list(args),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            cwd=str(cwd) if cwd else None,
        )
    except FileNotFoundError as exc:
        raise EditorPipelineError(
            f"Required executable was not found: {args[0]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise EditorPipelineError(
            f"Process timed out after {timeout_seconds} seconds: {args[0]}"
        ) from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr[-5000:] if stderr else stdout[-5000:]
        raise EditorPipelineError(
            f"Process failed with exit code {result.returncode}: {args[0]}\n{detail}"
        )
    return result


def _find_executable(name: str, environment_variable: str | None = None) -> str:
    """Locate an executable using PATH or an explicit environment override."""
    override = os.environ.get(environment_variable, "").strip() if environment_variable else ""
    if override:
        resolved_override = shutil.which(override) or override
        if Path(resolved_override).is_file():
            return str(Path(resolved_override).resolve())
        raise EditorPipelineError(
            f"Executable override {environment_variable} is invalid: {override}"
        )
    executable = shutil.which(name)
    if not executable:
        raise EditorPipelineError(
            f"Required executable '{name}' is not installed or not on PATH."
        )
    return executable


def _ffmpeg_path() -> str:
    """Find FFmpeg."""
    return _find_executable("ffmpeg", "FFMPEG_PATH")


def _ffprobe_path() -> str:
    """Find FFprobe."""
    return _find_executable("ffprobe", "FFPROBE_PATH")


def _probe_media(path: Path) -> dict[str, Any]:
    """Read media stream metadata through FFprobe's JSON output."""
    path = path.resolve()
    probe = _ffprobe_path()
    result = _run_process(
        [
            probe,
            "-v", "error",
            "-show_entries",
            "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,duration,sample_rate,channels",
            "-of", "json",
            str(path),
        ],
        timeout_seconds=120,
    )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise EditorPipelineError(
            f"FFprobe returned invalid JSON for '{path}'."
        ) from exc
    if not isinstance(data, dict):
        raise EditorPipelineError(f"FFprobe returned an invalid result for '{path}'.")
    return data


def _parse_fraction(value: Any) -> float:
    """Convert an FFprobe rational value such as 24/1 into a float."""
    if not value or value == "0/0":
        return 0.0
    try:
        if isinstance(value, str) and "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _probe_duration(probe_data: Mapping[str, Any]) -> float | None:
    """Extract the best available media duration from FFprobe data."""
    format_info = probe_data.get("format", {})
    if isinstance(format_info, Mapping):
        raw_duration = format_info.get("duration")
        if raw_duration is not None:
            try:
                duration = float(raw_duration)
                if math.isfinite(duration) and duration >= 0:
                    return duration
            except (TypeError, ValueError):
                pass

    durations: list[float] = []
    for stream in probe_data.get("streams", []):
        if not isinstance(stream, Mapping):
            continue
        raw_duration = stream.get("duration")
        if raw_duration is not None:
            try:
                duration = float(raw_duration)
                if math.isfinite(duration) and duration >= 0:
                    durations.append(duration)
            except (TypeError, ValueError):
                pass
    return max(durations) if durations else None


def _stream_flags(probe_data: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return whether the probed file contains video and audio streams."""
    has_video = False
    has_audio = False
    for stream in probe_data.get("streams", []):
        if not isinstance(stream, Mapping):
            continue
        kind = str(stream.get("codec_type", "")).lower()
        has_video |= kind == "video"
        has_audio |= kind == "audio"
    return has_video, has_audio


def _normalize_media_record(
    raw: Mapping[str, Any],
    asset_type: str,
    allowed_roots: Sequence[Path],
    index: int,
) -> dict[str, Any]:
    """Resolve one media asset and calculate its trusted metadata."""
    raw_path = _first_value(raw, ("path", "file_path", "relative_path", "uri"), None)
    if not raw_path:
        raise EditorValidationError(f"{asset_type} asset {index} has no path.")

    path = _resolve_input_path(str(raw_path), allowed_roots)
    if path.suffix.lower() not in (
        SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS
    ):
        raise EditorValidationError(
            f"Unsupported {asset_type} file extension: {path.suffix}"
        )

    expected_checksum = _first_value(raw, ("sha256", "checksum", "file_sha256"), None)
    actual_checksum = _sha256_file(path)
    if expected_checksum and str(expected_checksum).lower() != actual_checksum:
        raise EditorValidationError(f"Checksum mismatch for input asset: {path}")

    probe_data: dict[str, Any] = {}
    duration = 0.0
    width = 0
    height = 0
    fps = 0.0
    codec = None

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        probe_data = _probe_media(path)
        duration = _probe_duration(probe_data) or 0.0
        for stream in probe_data.get("streams", []):
            if not isinstance(stream, Mapping):
                continue
            if stream.get("codec_type") == "video":
                width = int(stream.get("width") or 0)
                height = int(stream.get("height") or 0)
                fps = _parse_fraction(
                    stream.get("avg_frame_rate") or stream.get("r_frame_rate")
                )
                codec = str(stream.get("codec_name") or "") or codec
                break
            if stream.get("codec_type") == "audio":
                codec = str(stream.get("codec_name") or "") or codec

    return {
        "asset_id": str(
            _first_value(raw, ("asset_id", "id"), f"{asset_type}_{index:04d}")
        ),
        "asset_type": asset_type,
        "path": str(path),
        "relative_path": str(path),
        "sha256": actual_checksum,
        "size_bytes": path.stat().st_size,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "codec": codec,
        "probe": probe_data,
        "source_record": dict(raw),
    }


def _flatten_asset_records(
    manifest: Mapping[str, Any],
    candidate_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Collect asset records from common manifest structures."""
    records: list[dict[str, Any]] = []
    for key in candidate_keys:
        value = manifest.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, Mapping))
        elif isinstance(value, Mapping):
            if value.get("path") or value.get("file_path"):
                records.append(dict(value))
            else:
                for nested in value.values():
                    if isinstance(nested, Mapping) and (
                        nested.get("path") or nested.get("file_path")
                    ):
                        records.append(dict(nested))
                    elif isinstance(nested, list):
                        records.extend(
                            item for item in nested if isinstance(item, Mapping)
                        )

    scenes = manifest.get("scenes")
    if isinstance(scenes, list):
        for scene in scenes:
            if not isinstance(scene, Mapping):
                continue
            for key in candidate_keys:
                value = scene.get(key)
                if isinstance(value, Mapping):
                    records.append(dict(value))
                elif isinstance(value, list):
                    records.extend(item for item in value if isinstance(item, Mapping))

            for key in ("frame_assets", "frames", "motion_asset", "motion"):
                value = scene.get(key)
                if isinstance(value, Mapping):
                    records.append(dict(value))
                elif isinstance(value, list):
                    records.extend(item for item in value if isinstance(item, Mapping))

    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        raw_path = _first_value(record, ("path", "file_path", "relative_path"), None)
        if raw_path:
            unique[str(raw_path)] = dict(record)
    return list(unique.values())


def _timeline_events(timeline: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize timeline events for audio assembly and subtitles."""
    raw = _first_value(
        timeline,
        ("events", "segments", "audio_events", "timeline"),
        [],
    )
    if isinstance(raw, Mapping):
        raw = raw.get("events", [])
    if not isinstance(raw, list):
        raise EditorValidationError("Audio timeline events must be a list.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        event = dict(item)
        start = _first_value(
            event,
            ("start_seconds", "start", "start_time", "start_s", "offset_seconds"),
            0.0,
        )
        end = _first_value(
            event,
            ("end_seconds", "end", "end_time", "end_s"),
            None,
        )
        start_value = _as_float(start, f"timeline event {index} start")
        if end is None:
            duration = _as_float(
                _first_value(event, ("duration_seconds", "duration", "length_seconds"), 0.0),
                f"timeline event {index} duration",
            )
            end_value = start_value + duration
        else:
            end_value = _as_float(end, f"timeline event {index} end")
        if end_value < start_value:
            raise EditorValidationError(
                f"Timeline event {index} ends before it starts."
            )
        event["start_seconds"] = start_value
        event["end_seconds"] = end_value
        event["event_type"] = str(
            _first_value(event, ("event_type", "type", "kind", "category"), "audio")
        ).lower()
        normalized.append(event)

    normalized.sort(key=lambda event: (event["start_seconds"], event["end_seconds"]))
    return normalized


def _expected_duration(
    blueprint: Mapping[str, Any],
    timeline: Mapping[str, Any],
    visual_manifest: Mapping[str, Any],
) -> float:
    """Determine the master export duration from authoritative timeline data."""
    for source in (timeline, visual_manifest, blueprint):
        explicit = _first_value(
            source,
            (
                "total_duration_seconds",
                "duration_seconds",
                "total_duration",
                "duration",
            ),
            None,
        )
        if explicit is not None:
            parsed = _as_float(explicit, "project duration")
            if parsed > 0:
                return parsed

    events = _timeline_events(timeline)
    if events:
        return max(event["end_seconds"] for event in events)

    scenes = visual_manifest.get("scenes", [])
    ends: list[float] = []
    if isinstance(scenes, list):
        for scene in scenes:
            if isinstance(scene, Mapping) and scene.get("end_seconds") is not None:
                ends.append(_as_float(scene["end_seconds"], "visual scene end"))
    if ends:
        return max(ends)

    raise EditorValidationError(
        "Cannot determine project duration from screenplay, audio timeline, or visual manifest."
    )


def _validate_manifest_asset(
    asset: Mapping[str, Any],
    *,
    asset_type: str,
    allowed_roots: Sequence[Path],
) -> dict[str, Any]:
    """Validate existence and checksum for a manifest asset."""
    raw_path = _first_value(asset, ("path", "file_path", "relative_path"), None)
    if not raw_path:
        raise EditorValidationError(f"{asset_type} asset is missing its path.")
    path = _resolve_input_path(str(raw_path), allowed_roots)
    expected = _first_value(asset, ("sha256", "checksum"), None)
    actual = _sha256_file(path)
    if expected and str(expected).lower() != actual:
        raise EditorValidationError(f"{asset_type} checksum mismatch: {path}")
    return {
        "asset_id": str(_first_value(asset, ("asset_id", "id"), path.stem)),
        "asset_type": asset_type,
        "path": str(path),
        "relative_path": str(path),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
        "source_record": dict(asset),
    }


def validate_media_inputs(
    blueprint: Mapping[str, Any] | Any,
    timeline: Mapping[str, Any] | Any,
    visual_manifest: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """
    Validate media assets and assemble a normalized, checksum-verified manifest.

    Input manifests may be dictionaries or model objects. File references
    must resolve to the expected project workspaces.
    """
    blueprint_data = _model_dump(blueprint)
    timeline_data = _model_dump(timeline)
    visual_data = _model_dump(visual_manifest)

    events = _timeline_events(timeline_data)
    duration = _expected_duration(blueprint_data, timeline_data, visual_data)

    visual_records = _flatten_asset_records(
        visual_data,
        (
            "frame_assets",
            "frames",
            "motion_assets",
            "motion_asset",
            "video_assets",
            "assets",
        ),
    )
    if not visual_records:
        raise EditorValidationError(
            "Visual manifest contains no frame or motion assets."
        )

    normalized_visuals: list[dict[str, Any]] = []
    for index, raw in enumerate(visual_records, start=1):
        raw_type = str(
            _first_value(raw, ("asset_type", "type", "kind"), "")
        ).lower()
        if "motion" in raw_type or "video" in raw_type:
            asset_type = "motion"
        elif Path(str(_first_value(raw, ("path", "file_path", "relative_path"), ""))).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            asset_type = "motion"
        elif Path(str(_first_value(raw, ("path", "file_path", "relative_path"), ""))).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            asset_type = "frame"
        else:
            asset_type = "visual"
        allowed = (VISUAL_ROOT, PROJECT_ROOT)
        normalized_visuals.append(
            _normalize_media_record(raw, asset_type, allowed, index)
        )

    motion_assets = [
        asset for asset in normalized_visuals
        if asset["asset_type"] == "motion"
    ]
    frame_assets = [
        asset for asset in normalized_visuals
        if asset["asset_type"] == "frame"
    ]
    if not motion_assets and not frame_assets:
        raise EditorValidationError("No usable visual assets were found.")

    audio_records = _flatten_asset_records(
        timeline_data,
        ("audio_assets", "audio_tracks", "tracks", "segments", "assets"),
    )
    normalized_audio: list[dict[str, Any]] = []
    for index, raw in enumerate(audio_records, start=1):
        raw_type = str(
            _first_value(raw, ("asset_type", "type", "kind", "event_type"), "")
        ).lower()
        raw_path = _first_value(raw, ("path", "file_path", "relative_path"), None)
        if not raw_path:
            continue
        suffix = Path(str(raw_path)).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        if any(
            existing["path"] == str(Path(str(raw_path)).resolve())
            for existing in normalized_audio
        ):
            continue
        normalized_audio.append(
            _normalize_media_record(
                raw,
                "audio",
                (AUDIO_ROOT, PROJECT_ROOT),
                index,
            )
        )

    if not events and not normalized_audio:
        raise EditorValidationError(
            "Audio timeline has neither timing events nor referenced audio assets."
        )

    for event in events:
        event_end = event["end_seconds"]
        if event_end > duration + 0.05:
            raise EditorValidationError(
                f"Audio event ends at {event_end:.3f}s, beyond project duration "
                f"{duration:.3f}s."
            )

    return {
        "manifest_version": MANIFEST_VERSION,
        "manifest_type": "validated_media_manifest",
        "created_at_unix": int(time.time()),
        "project_id": str(
            _first_value(blueprint_data, ("project_id", "project_name", "id"), "cinema_project")
        ),
        "expected_duration_seconds": duration,
        "fps": _as_float(
            _first_value(
                timeline_data,
                ("fps", "frame_rate", "frames_per_second"),
                _first_value(visual_data, ("fps", "frame_rate", "frames_per_second"), DEFAULT_FPS),
            ),
            "project fps",
            1.0,
        ),
        "blueprint": blueprint_data,
        "timeline": timeline_data,
        "timeline_events": events,
        "visual_assets": normalized_visuals,
        "motion_assets": motion_assets,
        "frame_assets": frame_assets,
        "audio_assets": normalized_audio,
        "input_checksums": {
            "visual_manifest_sha256": hashlib.sha256(
                json.dumps(visual_data, sort_keys=True, default=_json_default).encode()
            ).hexdigest(),
            "timeline_sha256": hashlib.sha256(
                json.dumps(timeline_data, sort_keys=True, default=_json_default).encode()
            ).hexdigest(),
            "blueprint_sha256": hashlib.sha256(
                json.dumps(blueprint_data, sort_keys=True, default=_json_default).encode()
            ).hexdigest(),
        },
    }


def _select_motion_asset(
    media_manifest: Mapping[str, Any],
    scene_id: str | None = None,
    scene_index: int = 0,
) -> dict[str, Any] | None:
    """Select a motion asset, preferring a matching scene identifier."""
    motions = media_manifest.get("motion_assets", [])
    if not isinstance(motions, list) or not motions:
        return None

    if scene_id:
        for asset in motions:
            if not isinstance(asset, Mapping):
                continue
            source = asset.get("source_record", {})
            candidate_scene = _first_value(
                source,
                ("scene_id", "scene", "scene_slug"),
                None,
            ) if isinstance(source, Mapping) else None
            if candidate_scene is not None and str(candidate_scene) == str(scene_id):
                return dict(asset)

    return dict(motions[min(scene_index, len(motions) - 1)])


def _ffmpeg_has_encoder(encoder: str) -> bool:
    """Check whether the installed FFmpeg advertises a named encoder."""
    try:
        result = _run_process(
            [_ffmpeg_path(), "-hide_banner", "-encoders"],
            timeout_seconds=60,
        )
    except EditorPipelineError:
        return False
    return encoder in result.stdout


def _choose_video_encoder() -> tuple[list[str], str]:
    """Prefer NVIDIA NVENC when available; otherwise select CPU H.264."""
    requested = os.environ.get("VIDEO_ENCODER", "").strip().lower()
    if requested in {"h264_nvenc", "hevc_nvenc"}:
        if _ffmpeg_has_encoder(requested):
            return [requested, "-preset", "p5", "-cq", "23"], requested
        return ["libx264", "-preset", "medium", "-crf", "20"], "libx264"

    if requested in {"libx264", "libx265"}:
        return (
            [requested, "-preset", "medium", "-crf", "20"],
            requested,
        )

    if _ffmpeg_has_encoder("h264_nvenc"):
        return ["h264_nvenc", "-preset", "p5", "-cq", "23"], "h264_nvenc"
    return ["libx264", "-preset", "medium", "-crf", "20"], "libx264"


def _quote_filter_path(path: str | Path) -> str:
    """Escape a filesystem path for FFmpeg filter syntax, not shell syntax."""
    value = str(Path(path).resolve())
    value = value.replace("\\", "\\\\")
    value = value.replace(":", "\\:")
    value = value.replace("'", "\\'")
    value = value.replace(",", "\\,")
    value = value.replace("[", "\\[")
    value = value.replace("]", "\\]")
    return value


def _scene_timing_records(media_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read scene intervals from the visual manifest for clip sequencing."""
    visual_data = media_manifest.get("visual_manifest", {})
    if not isinstance(visual_data, Mapping):
        visual_data = {}
    scenes = visual_data.get("scenes", [])
    if not isinstance(scenes, list):
        scenes = []

    normalized: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        if not isinstance(scene, Mapping):
            continue
        start = _first_value(scene, ("start_seconds", "start", "start_time"), None)
        end = _first_value(scene, ("end_seconds", "end", "end_time"), None)
        duration = _first_value(scene, ("duration_seconds", "duration"), None)
        if start is None and end is None and duration is None:
            continue
        start_value = _as_float(start if start is not None else 0.0, "scene start")
        if end is not None:
            end_value = _as_float(end, "scene end")
        elif duration is not None:
            end_value = start_value + _as_float(duration, "scene duration")
        else:
            continue
        if end_value <= start_value:
            continue
        normalized.append(
            {
                "scene_id": str(scene.get("scene_id", f"scene_{index + 1:03d}")),
                "start_seconds": start_value,
                "end_seconds": end_value,
                "duration_seconds": end_value - start_value,
                "scene_record": dict(scene),
            }
        )
    normalized.sort(key=lambda item: item["start_seconds"])
    return normalized


def _concat_escape_path(path: Path) -> str:
    """Escape a path for FFmpeg concat-demuxer list-file syntax."""
    value = str(path.resolve())
    return value.replace("\\", "\\\\").replace("'", "'\\''")


def _make_concat_file(paths: Sequence[Path], output_dir: Path) -> Path:
    """Write an FFmpeg concat demuxer list inside the master sandbox."""
    if not paths:
        raise EditorValidationError("Cannot create a concat list without input files.")
    list_path = _safe_path(
        output_dir / f"concat_{uuid.uuid4().hex}.txt",
        MASTER_ROOT,
    )
    lines = [f"file '{_concat_escape_path(path)}'" for path in paths]
    _atomic_write_bytes(list_path, ("\n".join(lines) + "\n").encode("utf-8"))
    return list_path


def _write_silence_wav(path: Path, duration_seconds: float, sample_rate: int = 48000) -> None:
    """Write a PCM silence track as a portable fallback audio source."""
    path = _safe_path(path, MASTER_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(math.ceil(duration_seconds * sample_rate)))
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp.wav",
        dir=str(path.parent),
    )
    os.close(fd)
    try:
        with wave.open(temp_name, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            block = b"\x00" * (min(frame_count, sample_rate) * 4)
            remaining = frame_count
            while remaining:
                current = min(remaining, sample_rate)
                handle.writeframesraw(block[: current * 4])
                remaining -= current
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _audio_event_path(event: Mapping[str, Any]) -> str | None:
    """Extract an optional audio file path from a timeline event."""
    value = _first_value(
        event,
        ("path", "audio_path", "file_path", "asset_path", "uri"),
        None,
    )
    return str(value) if value else None


def _resolve_audio_event_path(raw_path: str) -> Path:
    """Resolve a timeline audio path within permitted audio/master workspaces."""
    return _resolve_input_path(raw_path, (AUDIO_ROOT, MASTER_ROOT, PROJECT_ROOT))


def _collect_audio_clips(
    media_manifest: Mapping[str, Any],
    duration: float,
) -> list[dict[str, Any]]:
    """Build timed audio clip records from timeline events and audio assets."""
    clips: list[dict[str, Any]] = []
    events = media_manifest.get("timeline_events", [])
    for index, event in enumerate(events if isinstance(events, list) else []):
        if not isinstance(event, Mapping):
            continue
        event_type = str(event.get("event_type", "audio")).lower()
        raw_path = _audio_event_path(event)
        if not raw_path:
            continue
        start = _as_float(event.get("start_seconds", 0), f"audio clip {index} start")
        end = _as_float(event.get("end_seconds", start), f"audio clip {index} end")
        if end <= start or start >= duration:
            continue
        path = _resolve_audio_event_path(raw_path)
        clips.append(
            {
                "path": str(path),
                "start_seconds": start,
                "end_seconds": min(end, duration),
                "duration_seconds": min(end, duration) - start,
                "event_type": event_type,
                "volume": _as_float(
                    _first_value(event, ("volume", "gain", "volume_linear"), 1.0),
                    f"audio clip {index} volume",
                ),
                "ducking": bool(event.get("ducking", False)),
                "source": dict(event),
            }
        )

    if not clips:
        for asset in media_manifest.get("audio_assets", []):
            if not isinstance(asset, Mapping):
                continue
            path = str(asset.get("path", ""))
            if not path:
                continue
            clips.append(
                {
                    "path": path,
                    "start_seconds": 0.0,
                    "end_seconds": min(
                        duration,
                        float(asset.get("duration_seconds") or duration),
                    ),
                    "duration_seconds": min(
                        duration,
                        float(asset.get("duration_seconds") or duration),
                    ),
                    "event_type": str(asset.get("asset_type", "audio")),
                    "volume": 1.0,
                    "ducking": False,
                    "source": dict(asset),
                }
            )
    return clips


def _audio_filter_for_clip(
    input_index: int,
    clip: Mapping[str, Any],
    sample_rate: int = 48000,
) -> str:
    """Build an audio filter chain with timed placement and volume control."""
    start_ms = max(0, int(round(float(clip["start_seconds"]) * 1000)))
    volume = max(0.0, min(4.0, float(clip.get("volume", 1.0))))
    return (
        f"[{input_index}:a]"
        f"aresample={sample_rate},"
        f"aformat=sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts=stereo,"
        f"volume={volume:.6f},"
        f"adelay={start_ms}|{start_ms},"
        f"apad"
        f"[a{input_index}]"
    )


def mix_audio_tracks(
    timeline: Mapping[str, Any] | Any,
    media_manifest: Mapping[str, Any] | Any,
    output_path: str | Path,
    test_mode: bool = False,
) -> dict[str, Any]:
    """
    Mix dialogue, music, ambience, and sound effects to a single audio file.

    The master output is written only within workspace/05_master_theater/.
    The FFmpeg invocation uses a structured argument array and shell=False.
    """
    _ensure_master_dirs()
    timeline_data = _model_dump(timeline)
    manifest_data = _model_dump(media_manifest)
    duration = _as_float(
        _first_value(
            manifest_data,
            ("expected_duration_seconds", "total_duration_seconds"),
            _first_value(
                timeline_data,
                ("total_duration_seconds", "duration_seconds", "duration"),
                0.0,
            ),
        ),
        "audio mix duration",
    )
    if duration <= 0:
        events = _timeline_events(timeline_data)
        duration = max((event["end_seconds"] for event in events), default=0.0)
    if duration <= 0:
        raise EditorValidationError("Cannot mix audio with an unknown duration.")

    destination = _safe_path(output_path, MASTER_ROOT)
    if destination.suffix.lower() not in {".wav", ".m4a", ".aac", ".mp3", ".flac"}:
        destination = destination.with_suffix(".wav")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if test_mode:
        silence_path = destination.with_name(f".{destination.stem}.silence.wav")
        _write_silence_wav(silence_path, min(duration, 2.0))
        try:
            probe = _probe_media(silence_path)
            actual = _probe_duration(probe) or 0.0
            if actual <= 0:
                raise EditorValidationError("Test silence track failed duration validation.")
            os.replace(silence_path, destination)
        finally:
            try:
                silence_path.unlink(missing_ok=True)
            except OSError:
                pass
        return {
            "asset_type": "mixed_audio",
            "path": str(destination),
            "sha256": _sha256_file(destination),
            "size_bytes": destination.stat().st_size,
            "duration_seconds": actual,
            "test_mode": True,
            "mix_method": "deterministic_pcm_silence",
            "tracks": [],
        }

    clips = _collect_audio_clips(manifest_data, duration)
    if not clips:
        raise EditorValidationError(
            "No audio files were found in the timeline or validated media manifest."
        )

    ffmpeg = _ffmpeg_path()
    args: list[str] = [ffmpeg, "-hide_banner", "-y", "-nostdin"]
    for clip in clips:
        args.extend(["-i", str(clip["path"])])

    filter_parts: list[str] = []
    labels: list[str] = []
    for index, clip in enumerate(clips):
        filter_parts.append(_audio_filter_for_clip(index, clip))
        labels.append(f"[a{index}]")

    if len(labels) == 1:
        filter_parts.append(f"{labels[0]}atrim=duration={duration:.6f},asetpts=N/SR/TB[mixout]")
    else:
        filter_parts.append(
            "".join(labels)
            + f"amix=inputs={len(labels)}:duration=longest:dropout_transition=2,"
            + f"atrim=duration={duration:.6f},asetpts=N/SR/TB[mixout]"
        )

    args.extend(
        [
            "-filter_complex", ";".join(filter_parts),
            "-map", "[mixout]",
            "-t", f"{duration:.6f}",
            "-c:a", "pcm_s16le" if destination.suffix.lower() == ".wav" else "aac",
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart" if destination.suffix.lower() in {".m4a", ".aac"} else "0",
            str(destination),
        ]
    )
    _run_process(args, timeout_seconds=max(300, int(duration * 8)))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise EditorPipelineError("FFmpeg produced no audio mix.")
    probe = _probe_media(destination)
    actual_duration = _probe_duration(probe)
    if actual_duration is None or actual_duration <= 0:
        raise EditorValidationError("Mixed audio has no valid duration.")
    if abs(actual_duration - duration) > max(1.0, duration * 0.03):
        raise EditorValidationError(
            f"Mixed audio duration mismatch: expected {duration:.3f}s, "
            f"got {actual_duration:.3f}s."
        )

    return {
        "asset_type": "mixed_audio",
        "path": str(destination),
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "duration_seconds": actual_duration,
        "test_mode": False,
        "mix_method": "ffmpeg_filter_complex_amix",
        "tracks": clips,
        "ffmpeg_arguments": args[1:],
    }


def _build_subtitle_text(value: Any) -> str:
    """Normalize subtitle text and strip unsupported control characters."""
    if value is None:
        return ""
    text_value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text_value = "".join(
        character for character in text_value
        if character in "\n\t" or ord(character) >= 32
    )
    return text_value.strip()


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SubRip HH:MM:SS,mmm time notation."""
    milliseconds_total = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds_total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _subtitle_segments(
    blueprint: Mapping[str, Any],
    timeline: Mapping[str, Any],
    language: str,
) -> list[dict[str, Any]]:
    """Build timed subtitle segments, preferring word/line timing from audio."""
    segments: list[dict[str, Any]] = []
    events = _timeline_events(timeline)

    for index, event in enumerate(events):
        event_type = str(event.get("event_type", "")).lower()
        text_value = _first_value(
            event,
            ("text", "transcript", "spoken_text", "subtitle", "line", "content"),
            None,
        )
        if text_value is None:
            continue
        if any(token in event_type for token in ("music", "sfx", "sound_effect", "ambience")):
            continue
        text_value = _build_subtitle_text(text_value)
        if not text_value:
            continue
        segments.append(
            {
                "start_seconds": event["start_seconds"],
                "end_seconds": event["end_seconds"],
                "text": text_value,
                "language": str(event.get("language", language)),
                "scene_id": str(event.get("scene_id", "")),
                "source": "audio_timeline",
            }
        )

    if segments:
        return segments

    scenes = _first_value(
        blueprint,
        ("scenes", "scene_blueprints", "scene_list", "shots"),
        [],
    )
    if isinstance(scenes, list):
        for scene_index, scene in enumerate(scenes):
            if not isinstance(scene, Mapping):
                continue
            raw_lines = _first_value(
                scene,
                ("dialogue", "lines", "narration", "subtitle_text", "text"),
                None,
            )
            if isinstance(raw_lines, str):
                raw_lines = [raw_lines]
            if not isinstance(raw_lines, list) or not raw_lines:
                continue
            start = _as_float(
                _first_value(scene, ("start_seconds", "start_time", "start_s"), 0.0),
                f"scene {scene_index + 1} subtitle start",
            )
            end_value = _first_value(
                scene,
                ("end_seconds", "end_time", "end_s"),
                None,
            )
            duration_value = _first_value(
                scene,
                ("duration_seconds", "duration", "estimated_duration"),
                None,
            )
            if end_value is not None:
                end = _as_float(end_value, "subtitle scene end")
            elif duration_value is not None:
                end = start + _as_float(duration_value, "subtitle scene duration")
            else:
                continue
            cleaned_lines = [_build_subtitle_text(line) for line in raw_lines]
            cleaned_lines = [line for line in cleaned_lines if line]
            if not cleaned_lines or end <= start:
                continue
            per_line = (end - start) / len(cleaned_lines)
            for line_index, line in enumerate(cleaned_lines):
                segments.append(
                    {
                        "start_seconds": start + line_index * per_line,
                        "end_seconds": (
                            end if line_index == len(cleaned_lines) - 1
                            else start + (line_index + 1) * per_line
                        ),
                        "text": line,
                        "language": language,
                        "scene_id": str(scene.get("scene_id", f"scene_{scene_index + 1:03d}")),
                        "source": "screenplay_fallback",
                    }
                )
    return segments


def render_subtitles(
    blueprint: Mapping[str, Any] | Any,
    timeline: Mapping[str, Any] | Any,
    output_path: str | Path,
    language: str = "hi",
    test_mode: bool = False,
) -> dict[str, Any]:
    """
    Render subtitles to a UTF-8 SRT file.

    Subtitle timing uses the audio timeline whenever timed dialogue is
    available. Test mode creates a valid, minimal SRT without model access.
    """
    _ensure_master_dirs()
    blueprint_data = _model_dump(blueprint)
    timeline_data = _model_dump(timeline)
    destination = _safe_path(output_path, MASTER_ROOT)
    if destination.suffix.lower() not in {".srt", ".vtt"}:
        destination = destination.with_suffix(".srt")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if test_mode:
        segments = [
            {
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "text": "TEST EXPORT",
                "language": language,
                "scene_id": "test_scene",
                "source": "test_fixture",
            }
        ]
    else:
        segments = _subtitle_segments(blueprint_data, timeline_data, language)

    if destination.suffix.lower() == ".vtt":
        lines = ["WEBVTT", ""]
        for segment in segments:
            start = _format_srt_time(segment["start_seconds"]).replace(",", ".")
            end = _format_srt_time(segment["end_seconds"]).replace(",", ".")
            lines.extend([f"{start} --> {end}", segment["text"], ""])
    else:
        lines = []
        for index, segment in enumerate(segments, start=1):
            start = _format_srt_time(segment["start_seconds"])
            end = _format_srt_time(segment["end_seconds"])
            lines.extend([str(index), f"{start} --> {end}", segment["text"], ""])
    encoded = ("\n".join(lines).rstrip() + "\n").encode("utf-8")
    _atomic_write_bytes(destination, encoded)

    return {
        "asset_type": "subtitles",
        "path": str(destination),
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "language": language,
        "segment_count": len(segments),
        "segments": segments,
        "test_mode": bool(test_mode),
    }


def apply_cinematic_grade(
    input_path: str | Path,
    output_path: str | Path,
    lut_path: str | Path | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """
    Apply a cinematic grade using FFmpeg filters or a supplied LUT.

    Test mode validates and copies the source media rather than running
    expensive grading filters. A LUT, when supplied, must be readable and
    must be inside an approved project workspace.
    """
    _ensure_master_dirs()
    source = _resolve_input_path(
        input_path,
        (MASTER_ROOT, VISUAL_ROOT, PROJECT_ROOT),
        must_exist=True,
    )
    destination = _safe_path(output_path, MASTER_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise EditorValidationError(
            f"Cinematic grading requires a video input, got '{source.suffix}'."
        )

    lut_file: Path | None = None
    if lut_path is not None and str(lut_path).strip():
        lut_file = _resolve_input_path(
            lut_path,
            (MASTER_ROOT, PROJECT_ROOT),
            must_exist=True,
        )
        if lut_file.suffix.lower() not in SUPPORTED_LUT_EXTENSIONS:
            raise EditorValidationError(
                f"Unsupported LUT/image extension: {lut_file.suffix}"
            )

    if test_mode:
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return {
            "asset_type": "graded_video",
            "path": str(destination),
            "sha256": _sha256_file(destination),
            "size_bytes": destination.stat().st_size,
            "duration_seconds": _probe_duration(_probe_media(destination)) or 0.0,
            "grading": "test_mode_copy",
            "lut_path": str(lut_file) if lut_file else None,
            "test_mode": True,
        }

    ffmpeg = _ffmpeg_path()
    filters: list[str] = []
    if lut_file and lut_file.suffix.lower() == ".cube":
        filters.append(f"lut3d=file='{_quote_filter_path(lut_file)}'")
    elif lut_file and lut_file.suffix.lower() == ".3dl":
        filters.append(f"lut3d=file='{_quote_filter_path(lut_file)}'")
    else:
        filters.append(
            "eq=contrast=1.06:saturation=1.08:brightness=0.005:gamma=1.01"
        )
        filters.append("colorbalance=rs=0.02:gs=0.005:bs=-0.015")
        if lut_file and lut_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            raise EditorValidationError(
                "Image-based LUTs are not supported by this FFmpeg grading path; "
                "use a .cube or .3dl LUT."
            )

    encoder_args, encoder_name = _choose_video_encoder()
    temp_output = _safe_path(
        destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"),
        MASTER_ROOT,
    )
    args = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-nostdin",
        "-i", str(source),
        "-vf", ",".join(filters),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", encoder_args[0],
        *encoder_args[1:],
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(temp_output),
    ]
    try:
        _run_process(args, timeout_seconds=3600)
        if not temp_output.is_file() or temp_output.stat().st_size == 0:
            raise EditorPipelineError("Cinematic grading produced no output.")
        _probe_media(temp_output)
        os.replace(temp_output, destination)
    finally:
        try:
            temp_output.unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "asset_type": "graded_video",
        "path": str(destination),
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "duration_seconds": _probe_duration(_probe_media(destination)) or 0.0,
        "grading": "ffmpeg_lut" if lut_file else "cinematic_eq_colorbalance",
        "lut_path": str(lut_file) if lut_file else None,
        "encoder": encoder_name,
        "test_mode": False,
    }


def _build_test_video(
    output_path: Path,
    duration_seconds: float = 1.5,
) -> dict[str, Any]:
    """Create a small test video using FFmpeg lavfi test sources."""
    ffmpeg = _ffmpeg_path()
    duration = min(
        MAX_TEST_DURATION_SECONDS,
        max(MIN_TEST_DURATION_SECONDS, float(duration_seconds)),
    )
    encoder_args, encoder_name = _choose_video_encoder()

    args = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-nostdin",
        "-f", "lavfi",
        "-i", "testsrc2=size=320x240:rate=24",
        "-f", "lavfi",
        "-i", "sine=frequency=440:sample_rate=48000",
        "-t", f"{duration:.3f}",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", encoder_args[0],
        *encoder_args[1:],
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "96k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]
    _run_process(args, timeout_seconds=180)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise EditorPipelineError("FFmpeg lavfi test render created no output.")
    return {
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
        "duration_seconds": _probe_duration(_probe_media(output_path)),
        "encoder": encoder_name,
        "test_source": "lavfi:testsrc2+sine",
    }


def build_ffmpeg_command(
    media_manifest: Mapping[str, Any] | Any,
    output_path: str | Path,
    test_mode: bool = False,
) -> list[str]:
    """
    Build an FFmpeg command as a structured argument list.

    In test mode, the command uses FFmpeg lavfi sources and does not depend
    on project assets. Production mode prefers scene motion clips and uses
    a concat-demuxer list when multiple clips are present.
    """
    _ensure_master_dirs()
    manifest = _model_dump(media_manifest)
    destination = _safe_path(output_path, MASTER_ROOT)
    if destination.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        destination = destination.with_suffix(".mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _ffmpeg_path()
    encoder_args, _encoder_name = _choose_video_encoder()

    if test_mode:
        duration = min(
            MAX_TEST_DURATION_SECONDS,
            max(
                MIN_TEST_DURATION_SECONDS,
                _as_float(
                    _first_value(
                        manifest,
                        ("expected_duration_seconds", "total_duration_seconds"),
                        1.5,
                    ),
                    "test duration",
                ),
            ),
        )
        return [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-nostdin",
            "-f", "lavfi",
            "-i", "testsrc2=size=320x240:rate=24",
            "-f", "lavfi",
            "-i", "sine=frequency=440:sample_rate=48000",
            "-t", f"{duration:.3f}",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", encoder_args[0],
            *encoder_args[1:],
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "96k",
            "-shortest",
            "-movflags", "+faststart",
            str(destination),
        ]

    motions = manifest.get("motion_assets", [])
    if not isinstance(motions, list):
        motions = []
    if not motions:
        raise EditorValidationError(
            "No motion assets are available for production export."
        )

    motion_paths: list[Path] = []
    for asset in motions:
        if not isinstance(asset, Mapping):
            continue
        path = _resolve_input_path(
            str(asset.get("path", "")),
            (VISUAL_ROOT, MASTER_ROOT, PROJECT_ROOT),
        )
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        expected = asset.get("sha256")
        if expected and _sha256_file(path) != expected:
            raise EditorValidationError(f"Motion asset checksum mismatch: {path}")
        motion_paths.append(path)

    if not motion_paths:
        raise EditorValidationError("No playable motion files were found.")

    args: list[str] = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-nostdin",
    ]

    if len(motion_paths) == 1:
        args.extend(["-i", str(motion_paths[0])])
    else:
        concat_list = _make_concat_file(motion_paths, MASTER_ROOT / "temporary")
        args.extend(["-f", "concat", "-safe", "0", "-i", str(concat_list)])

    expected_duration = _as_float(
        _first_value(
            manifest,
            ("expected_duration_seconds", "total_duration_seconds"),
            sum(
                float(asset.get("duration_seconds") or 0)
                for asset in motions
                if isinstance(asset, Mapping)
            ),
        ),
        "expected export duration",
    )
    if expected_duration <= 0:
        raise EditorValidationError("Production export duration must be positive.")

    args.extend(
        [
            "-t", f"{expected_duration:.6f}",
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", encoder_args[0],
            *encoder_args[1:],
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", DEFAULT_AUDIO_BITRATE,
            "-ar", "48000",
            "-movflags", "+faststart",
            str(destination),
        ]
    )
    return args


def validate_export(
    output_path: str | Path,
    expected_duration_seconds: float,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    """
    Verify exported file integrity, streams, and duration tolerance.

    FFprobe validates the container and streams. A full SHA-256 pass checks
    file readability and supplies a stable integrity fingerprint.
    """
    _ensure_master_dirs()
    destination = _safe_path(output_path, MASTER_ROOT)
    expected = _as_float(expected_duration_seconds, "expected_duration_seconds")
    tolerance = _as_float(tolerance_seconds, "tolerance_seconds")

    errors: list[str] = []
    warnings_list: list[str] = []
    exists = destination.is_file()
    size = destination.stat().st_size if exists else 0
    checksum: str | None = None
    actual_duration: float | None = None
    has_video = False
    has_audio = False

    if not exists:
        errors.append("Export file does not exist.")
    elif size <= 0:
        errors.append("Export file is empty.")
    else:
        try:
            checksum = _sha256_file(destination)
            probe = _probe_media(destination)
            actual_duration = _probe_duration(probe)
            has_video, has_audio = _stream_flags(probe)
            if not has_video:
                errors.append("Export does not contain a video stream.")
            if actual_duration is None:
                errors.append("Export duration could not be determined.")
            elif actual_duration <= 0:
                errors.append("Export duration is not positive.")
            elif abs(actual_duration - expected) > tolerance:
                errors.append(
                    f"Duration mismatch: expected {expected:.3f}s, "
                    f"actual {actual_duration:.3f}s, tolerance {tolerance:.3f}s."
                )
            if not has_audio:
                warnings_list.append("Export has no audio stream.")
            if not checksum:
                errors.append("Could not compute export checksum.")
        except Exception as exc:
            errors.append(f"Export integrity probe failed: {type(exc).__name__}: {exc}")

    difference = (
        abs(actual_duration - expected)
        if actual_duration is not None
        else None
    )
    return asdict(
        ExportValidation(
            valid=not errors,
            output_path=str(destination),
            file_exists=exists,
            size_bytes=size,
            sha256=checksum,
            expected_duration_seconds=expected,
            actual_duration_seconds=actual_duration,
            duration_difference_seconds=difference,
            tolerance_seconds=tolerance,
            has_video_stream=has_video,
            has_audio_stream=has_audio,
            errors=errors,
            warnings=warnings_list,
            checked_at_unix=int(time.time()),
        )
    )


def _prepare_export_duration(
    blueprint: Mapping[str, Any],
    timeline: Mapping[str, Any],
    visual_manifest: Mapping[str, Any],
    test_mode: bool,
) -> float:
    """Get the target duration, restricting CI test exports to 1–2 seconds."""
    if test_mode:
        return 1.5
    return _expected_duration(blueprint, timeline, visual_manifest)


def _copy_or_remux_video(
    source: Path,
    destination: Path,
    duration: float,
    subtitles_path: Path | None,
    test_mode: bool,
) -> dict[str, Any]:
    """Render a single video source, optionally embedding subtitles."""
    ffmpeg = _ffmpeg_path()
    encoder_args, encoder_name = _choose_video_encoder()
    args = [ffmpeg, "-hide_banner", "-y", "-nostdin", "-i", str(source)]

    filters: list[str] = []
    if subtitles_path is not None:
        filters.append(f"subtitles='{_quote_filter_path(subtitles_path)}'")

    args.extend(["-t", f"{duration:.6f}", "-map", "0:v:0", "-map", "0:a?"])
    if filters:
        args.extend(["-vf", ",".join(filters)])
        args.extend(["-c:v", encoder_args[0], *encoder_args[1:]])
    else:
        args.extend(["-c:v", "copy"])
    args.extend(["-c:a", "aac", "-b:a", DEFAULT_AUDIO_BITRATE, "-movflags", "+faststart", str(destination)])
    _run_process(args, timeout_seconds=max(180, int(duration * 20)))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise EditorPipelineError("FFmpeg render produced no export.")
    return {
        "path": str(destination),
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "duration_seconds": _probe_duration(_probe_media(destination)),
        "encoder": encoder_name if filters else "stream-copy-video",
    }


def run_editor(
    screenplay_path: str | Path,
    timeline_path: str | Path,
    visual_manifest_path: str | Path,
    output_path: str | Path,
    lut_path: str | Path | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """
    Execute the Master Theater export pipeline.

    Production flow validates inputs, mixes audio, generates subtitles,
    composes scene motion, optionally grades the video, renders the final
    export, and validates file integrity and duration. Test mode uses FFmpeg
    lavfi test sources and validates a 1.5-second export without requiring
    project media or a GPU.
    """
    _ensure_master_dirs()
    screenplay_file = _resolve_screenplay_path(screenplay_path)
    timeline_file = _resolve_timeline_path(timeline_path)
    visual_file = _resolve_visual_manifest_path(visual_manifest_path)
    destination = _safe_path(output_path, MASTER_ROOT)

    if destination.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        destination = destination.with_suffix(".mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)

    blueprint = _read_json_file(screenplay_file)
    timeline = _read_json_file(timeline_file)
    visual_manifest = _read_json_file(visual_file)

    if test_mode:
        expected_duration = 1.5
        test_temp = _safe_path(
            MASTER_ROOT / "temporary" / f"ci_lavfi_{uuid.uuid4().hex}.mp4",
            MASTER_ROOT,
        )
        try:
            test_render = _build_test_video(test_temp, expected_duration)
            if not test_temp.is_file() or test_temp.stat().st_size == 0:
                raise EditorPipelineError("Test export failed to create a file.")

            validation = validate_export(
                test_temp,
                expected_duration_seconds=expected_duration,
                tolerance_seconds=0.75,
            )
            if not validation["valid"]:
                raise EditorValidationError(
                    "CI test export failed validation: "
                    + "; ".join(validation["errors"])
                )

            if test_temp.resolve() != destination.resolve():
                os.replace(test_temp, destination)

            final_validation = validate_export(
                destination,
                expected_duration_seconds=expected_duration,
                tolerance_seconds=0.75,
            )
            if not final_validation["valid"]:
                raise EditorValidationError(
                    "Final CI export failed validation: "
                    + "; ".join(final_validation["errors"])
                )

            manifest_path = _safe_path(
                MASTER_ROOT / "manifests" / f"{destination.stem}_export_manifest.json",
                MASTER_ROOT,
            )
            export_manifest = {
                "manifest_version": MANIFEST_VERSION,
                "manifest_type": "master_export_manifest",
                "status": "success",
                "test_mode": True,
                "created_at_unix": int(time.time()),
                "output_path": str(destination),
                "output_sha256": _sha256_file(destination),
                "output_size_bytes": destination.stat().st_size,
                "expected_duration_seconds": expected_duration,
                "validation": final_validation,
                "test_render": test_render,
                "inputs": {
                    "screenplay_path": str(screenplay_file),
                    "screenplay_sha256": _sha256_file(screenplay_file),
                    "timeline_path": str(timeline_file),
                    "timeline_sha256": _sha256_file(timeline_file),
                    "visual_manifest_path": str(visual_file),
                    "visual_manifest_sha256": _sha256_file(visual_file),
                },
                "manifest_path": str(manifest_path),
            }
            _atomic_write_json(manifest_path, export_manifest)
            export_manifest["manifest_sha256"] = _sha256_file(manifest_path)
            _atomic_write_json(manifest_path, export_manifest)

            return {
                "status": "success",
                "test_mode": True,
                "output_path": str(destination),
                "output_sha256": _sha256_file(destination),
                "output_size_bytes": destination.stat().st_size,
                "duration_seconds": final_validation["actual_duration_seconds"],
                "validation": final_validation,
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "export_manifest": export_manifest,
            }
        finally:
            try:
                test_temp.unlink(missing_ok=True)
            except OSError:
                pass
            gc.collect()

    media_manifest = validate_media_inputs(
        blueprint=blueprint,
        timeline=timeline,
        visual_manifest=visual_manifest,
    )
    media_manifest["visual_manifest"] = visual_manifest
    expected_duration = _prepare_export_duration(
        blueprint,
        timeline,
        visual_manifest,
        test_mode=False,
    )

    media_manifest_path = _safe_path(
        MASTER_ROOT / "manifests" / f"{destination.stem}_validated_media.json",
        MASTER_ROOT,
    )
    _atomic_write_json(media_manifest_path, media_manifest)

    audio_mix_path = _safe_path(
        MASTER_ROOT / "audio_mix" / f"{destination.stem}_mix.wav",
        MASTER_ROOT,
    )
    audio_manifest = mix_audio_tracks(
        timeline=timeline,
        media_manifest=media_manifest,
        output_path=audio_mix_path,
        test_mode=False,
    )

    subtitle_path = _safe_path(
        MASTER_ROOT / "subtitles" / f"{destination.stem}.srt",
        MASTER_ROOT,
    )
    subtitle_manifest = render_subtitles(
        blueprint=blueprint,
        timeline=timeline,
        output_path=subtitle_path,
        language=str(
            _first_value(
                timeline,
                ("language", "subtitle_language", "narration_language"),
                "hi",
            )
        ),
        test_mode=False,
    )

    ffmpeg_command = build_ffmpeg_command(
        media_manifest=media_manifest,
        output_path=destination,
        test_mode=False,
    )

    # Replace motion-audio tracks with the finalized master mix so the
    # soundtrack follows the audio timeline rather than source clip audio.
    if "-i" in ffmpeg_command:
        first_input_index = ffmpeg_command.index("-i")
        first_input_path = Path(ffmpeg_command[first_input_index + 1]).resolve()
        if first_input_path.is_file():
            insert_index = first_input_index + 2
            ffmpeg_command[insert_index:insert_index] = ["-i", str(audio_mix_path)]

            # Explicitly map the mixed audio input while preserving the
            # first motion input as the visual source.
            try:
                map_index = ffmpeg_command.index("-map")
                if map_index + 1 < len(ffmpeg_command):
                    next_map = ffmpeg_command[map_index + 1]
                    if next_map.startswith("0:a"):
                        ffmpeg_command[map_index + 1] = "1:a:0"
                    else:
                        ffmpeg_command[map_index:map_index] = ["-map", "1:a:0"]
                else:
                    ffmpeg_command.extend(["-map", "1:a:0"])
            except ValueError:
                ffmpeg_command.extend(["-map", "1:a:0"])

    # Render to a temporary path so failed encodes cannot replace a previous
    # successful export.
    temporary_export = _safe_path(
        destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.render{destination.suffix}"
        ),
        MASTER_ROOT,
    )
    ffmpeg_command[-1] = str(temporary_export)

    try:
        _run_process(
            ffmpeg_command,
            timeout_seconds=max(600, int(expected_duration * 60)),
        )
        if not temporary_export.is_file() or temporary_export.stat().st_size == 0:
            raise EditorPipelineError("FFmpeg created an empty final render.")

        probe = _probe_media(temporary_export)
        has_video, _has_audio = _stream_flags(probe)
        if not has_video:
            raise EditorValidationError("Intermediate render has no video stream.")

        graded_path = temporary_export
        grading_manifest: dict[str, Any] | None = None
        if lut_path is not None and str(lut_path).strip():
            graded_path = _safe_path(
                MASTER_ROOT / "graded" / f"{destination.stem}_graded{destination.suffix}",
                MASTER_ROOT,
            )
            grading_manifest = apply_cinematic_grade(
                input_path=temporary_export,
                output_path=graded_path,
                lut_path=lut_path,
                test_mode=False,
            )

        # Subtitle burn-in is optional by default to preserve an accessible
        # sidecar. Set BURN_SUBTITLES=1 to burn subtitles into the video.
        burn_subtitles = os.environ.get("BURN_SUBTITLES", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if burn_subtitles and subtitle_manifest.get("segment_count", 0) > 0:
            subtitle_render = _safe_path(
                MASTER_ROOT / "temporary" / f"{destination.stem}_subtitled{destination.suffix}",
                MASTER_ROOT,
            )
            subtitle_args = [
                _ffmpeg_path(),
                "-hide_banner",
                "-y",
                "-nostdin",
                "-i", str(graded_path),
                "-vf", f"subtitles='{_quote_filter_path(subtitle_path)}'",
                "-map", "0:v:0",
                "-map", "0:a?",
                "-c:v", _choose_video_encoder()[0][0],
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(subtitle_render),
            ]
            _run_process(subtitle_args, timeout_seconds=max(300, int(expected_duration * 20)))
            if not subtitle_render.is_file() or subtitle_render.stat().st_size == 0:
                raise EditorPipelineError("Subtitle burn-in failed.")
            os.replace(subtitle_render, destination)
        else:
            final_source = graded_path
            if final_source.resolve() != destination.resolve():
                shutil.copy2(final_source, destination)

        validation = validate_export(
            destination,
            expected_duration_seconds=expected_duration,
            tolerance_seconds=max(DEFAULT_TOLERANCE_SECONDS, 1.0),
        )
        if not validation["valid"]:
            raise EditorValidationError(
                "Final export failed validation: " + "; ".join(validation["errors"])
            )

        export_manifest_path = _safe_path(
            MASTER_ROOT / "manifests" / f"{destination.stem}_export_manifest.json",
            MASTER_ROOT,
        )
        export_manifest = {
            "manifest_version": MANIFEST_VERSION,
            "manifest_type": "master_export_manifest",
            "status": "success",
            "test_mode": False,
            "created_at_unix": int(time.time()),
            "output_path": str(destination),
            "output_sha256": _sha256_file(destination),
            "output_size_bytes": destination.stat().st_size,
            "expected_duration_seconds": expected_duration,
            "actual_duration_seconds": validation["actual_duration_seconds"],
            "validation": validation,
            "inputs": {
                "screenplay_path": str(screenplay_file),
                "screenplay_sha256": _sha256_file(screenplay_file),
                "timeline_path": str(timeline_file),
                "timeline_sha256": _sha256_file(timeline_file),
                "visual_manifest_path": str(visual_file),
                "visual_manifest_sha256": _sha256_file(visual_file),
            },
            "media_manifest_path": str(media_manifest_path),
            "media_manifest_sha256": _sha256_file(media_manifest_path),
            "audio_mix": audio_manifest,
            "subtitle_manifest": subtitle_manifest,
            "grading_manifest": grading_manifest,
            "encoder": _choose_video_encoder()[1],
            "ffmpeg_arguments": ffmpeg_command[1:-1],
            "manifest_path": str(export_manifest_path),
        }
        _atomic_write_json(export_manifest_path, export_manifest)
        export_manifest["manifest_sha256"] = _sha256_file(export_manifest_path)
        _atomic_write_json(export_manifest_path, export_manifest)

        return {
            "status": "success",
            "test_mode": False,
            "output_path": str(destination),
            "output_sha256": _sha256_file(destination),
            "output_size_bytes": destination.stat().st_size,
            "duration_seconds": validation["actual_duration_seconds"],
            "validation": validation,
            "manifest_path": str(export_manifest_path),
            "manifest_sha256": _sha256_file(export_manifest_path),
            "audio_mix": audio_manifest,
            "subtitles": subtitle_manifest,
            "grading": grading_manifest,
            "export_manifest": export_manifest,
        }
    finally:
        try:
            temporary_export.unlink(missing_ok=True)
        except OSError:
            pass
        gc.collect()


__all__ = [
    "EditorPipelineError",
    "EditorSandboxError",
    "EditorValidationError",
    "MediaAsset",
    "ExportValidation",
    "validate_media_inputs",
    "build_ffmpeg_command",
    "mix_audio_tracks",
    "apply_cinematic_grade",
    "render_subtitles",
    "validate_export",
    "run_editor",
]
