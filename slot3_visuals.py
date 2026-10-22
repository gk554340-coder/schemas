# File: slot3_visuals.py
# Code type: Python
# Autonomous AI Cinema Studio Engine — Slot 3: Visual Lab
# All generated assets and manifests are confined to workspace/04_visual_lab/.
# Test mode is offline, deterministic, and suitable for GitHub Actions CI.

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
import uuid
import warnings

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
VISUAL_ROOT = (PROJECT_ROOT / "workspace" / "04_visual_lab").resolve()
RESEARCH_ROOT = (PROJECT_ROOT / "workspace" / "01_research_vault").resolve()
SCREENPLAY_ROOT = (PROJECT_ROOT / "workspace" / "02_screenplay_desk").resolve()
AUDIO_ROOT = (PROJECT_ROOT / "workspace" / "03_audio_studio").resolve()

DEFAULT_FRAME_WIDTH = 512
DEFAULT_FRAME_HEIGHT = 512
DEFAULT_FPS = 24
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEED = 20261021
MANIFEST_VERSION = "1.0.0"

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

_VISUAL_RUNTIME: dict[str, Any] = {
    "initialized": False,
    "test_mode": True,
    "device": "cpu",
    "model_name": None,
    "pipeline": None,
    "ip_adapter_loaded": False,
    "last_error": None,
}

class VisualPipelineError(RuntimeError):
    """Raised when visual generation fails or violates a pipeline contract."""


class VisualSandboxError(VisualPipelineError):
    """Raised when an input or output path escapes its permitted sandbox."""


class VisualValidationError(VisualPipelineError):
    """Raised when screenplay, audio timeline, or visual manifests are invalid."""


@dataclass
class VisualAsset:
    """Metadata for a generated image or motion asset."""

    asset_id: str
    scene_id: str
    asset_type: str
    path: str
    relative_path: str
    sha256: str
    size_bytes: int
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    seed: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class VisualSceneManifest:
    """Per-scene manifest linking images, motion, timing, and checksums."""

    scene_id: str
    scene_number: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    frame_count: int
    fps: float
    character_anchor_ids: list[str]
    frame_assets: list[dict[str, Any]]
    motion_asset: dict[str, Any] | None
    audio_cues: list[dict[str, Any]]
    seed: int
    status: str = "complete"


def _is_within(path: Path, root: Path) -> bool:
    """Return whether path resolves inside root."""
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
    """Resolve a path and reject traversal outside the permitted root."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()

    if not _is_within(candidate, root):
        raise VisualSandboxError(
            f"Path escapes permitted sandbox '{root}': {candidate}"
        )
    if not allow_root and candidate == root.resolve():
        raise VisualSandboxError("The sandbox root itself is not a file path.")
    if must_exist and not candidate.exists():
        raise FileNotFoundError(str(candidate))
    return candidate


def _ensure_visual_dirs() -> None:
    """Create the visual sandbox and its managed subdirectories."""
    VISUAL_ROOT.mkdir(parents=True, exist_ok=True)
    for name in (
        "anchors",
        "frames",
        "motion",
        "manifests",
        "runtime",
        "temporary",
    ):
        (VISUAL_ROOT / name).mkdir(parents=True, exist_ok=True)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes to a file inside the visual sandbox."""
    path = _safe_path(path, VISUAL_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write stable, human-readable JSON."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _json_default(value: Any) -> Any:
    """Convert common non-JSON objects to serializable representations."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
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


def _sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(
    path: Path,
    *,
    asset_id: str,
    scene_id: str,
    asset_type: str,
    duration_seconds: float = 0.0,
    width: int = 0,
    height: int = 0,
    fps: float = 0.0,
    frame_count: int = 0,
    seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build portable asset metadata with a checksum."""
    path = _safe_path(path, VISUAL_ROOT, must_exist=True)
    return asdict(
        VisualAsset(
            asset_id=asset_id,
            scene_id=scene_id,
            asset_type=asset_type,
            path=str(path),
            relative_path=str(path.relative_to(VISUAL_ROOT)),
            sha256=_sha256_file(path),
            size_bytes=path.stat().st_size,
            duration_seconds=max(0.0, float(duration_seconds)),
            width=int(width),
            height=int(height),
            fps=float(fps),
            frame_count=int(frame_count),
            seed=seed,
            metadata=dict(metadata or {}),
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and validate its top-level type."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise VisualValidationError(f"Expected a JSON object in {path}")
    return payload


def _model_dump(value: Any) -> dict[str, Any]:
    """Normalize dictionaries, Pydantic models, and dataclasses."""
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
    raise VisualValidationError(
        f"Unsupported object type: {type(value).__name__}"
    )


def _read_model_json(path: str | Path, root: Path) -> dict[str, Any]:
    """Read a JSON input from an explicitly permitted project sandbox."""
    resolved = _safe_path(path, root, must_exist=True)
    if not resolved.is_file():
        raise VisualValidationError(f"Expected a JSON file: {resolved}")
    try:
        return _read_json(resolved)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VisualValidationError(
            f"Invalid JSON input '{resolved}': {exc}"
        ) from exc


def _resolve_screenplay_path(path: str | Path) -> Path:
    """Resolve screenplay JSON from the Slot 2 sandbox or a project-relative path."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        local_candidate = (SCREENPLAY_ROOT / candidate).resolve()
        project_candidate = (PROJECT_ROOT / candidate).resolve()
        resolved = local_candidate if local_candidate.exists() else project_candidate

    if not (
        _is_within(resolved, SCREENPLAY_ROOT)
        or _is_within(resolved, PROJECT_ROOT)
    ):
        raise VisualSandboxError("Screenplay path is outside permitted project paths.")
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


def _resolve_audio_timeline_path(path: str | Path) -> Path:
    """Resolve the timeline JSON from Slot 4 or a project-relative path."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        local_candidate = (AUDIO_ROOT / candidate).resolve()
        project_candidate = (PROJECT_ROOT / candidate).resolve()
        resolved = local_candidate if local_candidate.exists() else project_candidate

    if not (
        _is_within(resolved, AUDIO_ROOT)
        or _is_within(resolved, PROJECT_ROOT)
    ):
        raise VisualSandboxError("Audio timeline path is outside permitted project paths.")
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


def _first_value(mapping: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    """Return the first non-null value found under candidate field names."""
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return default


def _as_float(value: Any, name: str, minimum: float = 0.0) -> float:
    """Parse a finite float and enforce a lower bound."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VisualValidationError(f"{name} must be numeric.") from exc
    if not math.isfinite(result) or result < minimum:
        raise VisualValidationError(f"{name} must be finite and >= {minimum}.")
    return result


def _as_int(value: Any, name: str, minimum: int = 0) -> int:
    """Parse an integer and enforce a lower bound."""
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise VisualValidationError(f"{name} must be an integer.") from exc
    if result < minimum:
        raise VisualValidationError(f"{name} must be >= {minimum}.")
    return result


def _scene_list(blueprint: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract normalized scene records from a screenplay blueprint."""
    raw_scenes = _first_value(
        blueprint,
        ("scenes", "scene_blueprints", "scene_list", "shots"),
        None,
    )
    if raw_scenes is None and isinstance(blueprint.get("screenplay"), Mapping):
        raw_scenes = _first_value(
            blueprint["screenplay"],
            ("scenes", "scene_blueprints", "scene_list", "shots"),
            None,
        )
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise VisualValidationError(
            "Screenplay blueprint must contain a non-empty scenes list."
        )

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_scenes, start=1):
        if not isinstance(raw, Mapping):
            raise VisualValidationError(f"Scene {index} must be a JSON object.")
        scene = dict(raw)
        scene_number = _as_int(
            _first_value(scene, ("scene_number", "scene_no", "number", "index"), index),
            f"scene {index} number",
            1,
        )
        scene_id = str(
            _first_value(scene, ("scene_id", "id", "slug"), f"scene_{scene_number:03d}")
        ).strip()
        if not scene_id:
            scene_id = f"scene_{scene_number:03d}"
        scene["scene_number"] = scene_number
        scene["scene_id"] = scene_id
        normalized.append(scene)
    return normalized


def _extract_timeline_events(timeline: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize audio timeline events to a common timing representation."""
    raw_events = _first_value(
        timeline,
        ("events", "segments", "audio_events", "timeline"),
        [],
    )
    if isinstance(raw_events, Mapping):
        raw_events = raw_events.get("events", [])
    if not isinstance(raw_events, list):
        raise VisualValidationError("Audio timeline events must be a list.")

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            continue
        event = dict(raw)
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
            duration = _first_value(
                event,
                ("duration_seconds", "duration", "length_seconds"),
                0.0,
            )
            end_value = start_value + _as_float(
                duration, f"timeline event {index} duration"
            )
        else:
            end_value = _as_float(end, f"timeline event {index} end")
        if end_value < start_value:
            raise VisualValidationError(
                f"Audio event {index} ends before it starts."
            )
        event["start_seconds"] = start_value
        event["end_seconds"] = end_value
        event["event_type"] = str(
            _first_value(event, ("event_type", "type", "kind", "category"), "audio")
        ).lower()
        event["scene_id"] = str(
            _first_value(event, ("scene_id", "scene", "scene_slug"), "")
        )
        normalized.append(event)

    normalized.sort(key=lambda item: (item["start_seconds"], item["end_seconds"]))
    return normalized


def _scene_timing(
    scene: Mapping[str, Any],
    scene_index: int,
    events: Sequence[Mapping[str, Any]],
    cursor_seconds: float,
) -> tuple[float, float]:
    """Determine scene timing using explicit screenplay values or audio cues."""
    scene_id = str(scene.get("scene_id", ""))
    explicit_start = _first_value(
        scene,
        ("start_seconds", "start_time", "start_s", "timeline_start"),
        None,
    )
    explicit_end = _first_value(
        scene,
        ("end_seconds", "end_time", "end_s", "timeline_end"),
        None,
    )
    explicit_duration = _first_value(
        scene,
        ("duration_seconds", "duration", "length_seconds", "estimated_duration"),
        None,
    )

    related_events = [
        event for event in events
        if event.get("scene_id") == scene_id
        or str(event.get("scene_id", "")) == str(scene.get("scene_number", ""))
    ]

    if explicit_start is not None:
        start = _as_float(explicit_start, f"{scene_id} start")
    elif related_events:
        start = min(event["start_seconds"] for event in related_events)
    else:
        start = cursor_seconds

    if explicit_end is not None:
        end = _as_float(explicit_end, f"{scene_id} end")
    elif related_events:
        end = max(event["end_seconds"] for event in related_events)
    elif explicit_duration is not None:
        end = start + _as_float(explicit_duration, f"{scene_id} duration")
    else:
        raise VisualValidationError(
            f"Scene '{scene_id}' has no duration or audio-timeline timing anchors."
        )

    if explicit_duration is not None and explicit_end is None and not related_events:
        end = start + _as_float(explicit_duration, f"{scene_id} duration")
    if end <= start:
        raise VisualValidationError(
            f"Scene '{scene_id}' must have positive duration; got {start}–{end}."
        )
    return start, end


def _events_for_scene(
    scene: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    """Return audio cues overlapping a scene's exact time interval."""
    scene_id = str(scene.get("scene_id", ""))
    scene_number = str(scene.get("scene_number", ""))
    cues: list[dict[str, Any]] = []
    for event in events:
        event_scene = str(event.get("scene_id", ""))
        overlaps = (
            event["start_seconds"] < end
            and event["end_seconds"] > start
        )
        scene_matches = not event_scene or event_scene in {scene_id, scene_number}
        if overlaps and scene_matches:
            cues.append(dict(event))
    return cues


def _anchor_ids_for_scene(
    scene: Mapping[str, Any],
    anchor_manifest: Mapping[str, Any],
) -> list[str]:
    """Find character anchor identifiers referenced by a scene."""
    anchors = anchor_manifest.get("anchors", [])
    if not isinstance(anchors, list):
        anchors = []
    known_ids = {
        str(item.get("character_id"))
        for item in anchors
        if isinstance(item, Mapping) and item.get("character_id") is not None
    }

    characters = _first_value(
        scene,
        ("characters", "character_ids", "cast", "character_names"),
        [],
    )
    if isinstance(characters, str):
        characters = [characters]
    if not isinstance(characters, list):
        characters = []

    result: list[str] = []
    for character in characters:
        if isinstance(character, Mapping):
            identifier = _first_value(
                character, ("character_id", "id", "name", "character_name"), None
            )
        else:
            identifier = character
        if identifier is not None:
            candidate = str(identifier)
            if candidate in known_ids and candidate not in result:
                result.append(candidate)

    if not result and len(known_ids) == 1:
        result = [next(iter(known_ids))]
    return result


def _deterministic_png(width: int, height: int, seed: int, label: str) -> bytes:
    """Create a valid, deterministic PNG without external imaging libraries."""
    import struct
    import zlib

    width = max(1, int(width))
    height = max(1, int(height))
    rng = random.Random(seed)
    label_digest = hashlib.sha256(label.encode("utf-8")).digest()
    base_r, base_g, base_b = label_digest[0], label_digest[1], label_digest[2]
    scanlines = bytearray()

    for y in range(height):
        scanlines.append(0)
        for x in range(width):
            variation = rng.randrange(0, 24)
            radial = int(18 * math.sin((x / max(1, width - 1)) * math.pi))
            scanlines.extend(
                (
                    max(0, min(255, base_r // 2 + variation + radial)),
                    max(0, min(255, base_g // 2 + variation)),
                    max(0, min(255, base_b // 2 + variation)),
                )
            )

    def chunk(chunk_type: bytes, payload: bytes) -> bytes:
        body = chunk_type + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"tEXt", b"Description\x00" + label.encode("utf-8", "replace")[:500])
        + chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6))
        + chunk(b"IEND", b"")
    )


def _write_dummy_mp4(path: Path, duration_seconds: float, seed: int) -> dict[str, Any]:
    """Write a small, valid placeholder motion asset using a generated black video."""
    try:
        import cv2
        import numpy as np

        fps = DEFAULT_FPS
        frame_count = max(1, int(math.ceil(duration_seconds * fps)))
        height = 64
        width = 64
        temporary_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.mp4")
        writer = cv2.VideoWriter(
            str(temporary_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise VisualPipelineError("OpenCV could not initialize the test video writer.")
        rng = random.Random(seed)
        try:
            for frame_index in range(frame_count):
                value = rng.randrange(0, 12)
                frame = np.full((height, width, 3), value, dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"TEST {frame_index:04d}",
                    (2, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.22,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(frame)
        finally:
            writer.release()
        if not temporary_path.exists() or temporary_path.stat().st_size == 0:
            raise VisualPipelineError("Test video writer produced no output.")
        os.replace(temporary_path, path)
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "container": "mp4",
            "test_placeholder": True,
        }
    except ImportError:
        payload = {
            "format": "visual-test-motion-placeholder",
            "version": 1,
            "duration_seconds": duration_seconds,
            "fps": DEFAULT_FPS,
            "frame_count": max(1, int(math.ceil(duration_seconds * DEFAULT_FPS))),
            "seed": seed,
            "note": "OpenCV and NumPy are unavailable; this is a JSON placeholder, not playable video.",
        }
        path = path.with_suffix(".motion.json")
        _atomic_write_json(path, payload)
        return {
            "width": 0,
            "height": 0,
            "fps": DEFAULT_FPS,
            "frame_count": payload["frame_count"],
            "container": "json-placeholder",
            "test_placeholder": True,
        }


def _validate_image_file(path: Path) -> tuple[int, int]:
    """Read image dimensions using Pillow, or validate PNG headers directly."""
    path = _safe_path(path, VISUAL_ROOT, must_exist=True)
    if path.suffix.lower() == ".png":
        with path.open("rb") as handle:
            header = handle.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
            import struct
            width, height = struct.unpack(">II", header[16:24])
            return width, height
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except ImportError as exc:
        raise VisualPipelineError(
            f"Cannot validate non-PNG image without Pillow: {path}"
        ) from exc
    except Exception as exc:
        raise VisualPipelineError(f"Invalid image asset '{path}': {exc}") from exc


def _runtime_device(device: str) -> str:
    """Normalize a requested compute device and validate CUDA availability."""
    requested = str(device or "auto").strip().lower()
    if requested in {"", "auto"}:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if requested.startswith("cuda"):
        try:
            import torch
        except ImportError as exc:
            raise VisualPipelineError(
                "CUDA was requested, but PyTorch is not installed."
            ) from exc
        if not torch.cuda.is_available():
            raise VisualPipelineError(
                "CUDA was requested, but torch.cuda.is_available() is false."
            )
    if requested not in {"cpu", "mps"} and not requested.startswith("cuda"):
        raise VisualValidationError(
            f"Unsupported device '{device}'. Use auto, cpu, mps, or cuda."
        )
    return requested


def initialize_visual_runtime(
    device: str = "auto",
    model_name: str = "black-forest-labs/FLUX.1-schnell",
    test_mode: bool = False,
) -> dict[str, Any]:
    """
    Initialize the visual generation runtime.

    Test mode deliberately imports no GPU framework or model library.
    Production mode loads FLUX.1-schnell; IP-Adapter support is configured
    when compatible pipeline classes and weights are available.
    """
    _ensure_visual_dirs()
    if test_mode:
        _VISUAL_RUNTIME.update(
            {
                "initialized": True,
                "test_mode": True,
                "device": "cpu",
                "model_name": "deterministic-test-runtime",
                "pipeline": None,
                "ip_adapter_loaded": False,
                "last_error": None,
            }
        )
        return {
            "initialized": True,
            "test_mode": True,
            "device": "cpu",
            "model_name": "deterministic-test-runtime",
            "backend": "synthetic",
            "ip_adapter_loaded": False,
        }

    resolved_device = _runtime_device(device)
    try:
        import torch
        from diffusers import FluxPipeline

        pipeline = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=(
                torch.bfloat16
                if resolved_device.startswith("cuda")
                and torch.cuda.is_bf16_supported()
                else torch.float16
                if resolved_device.startswith("cuda")
                else torch.float32
            ),
        )
        pipeline = pipeline.to(resolved_device)

        ip_adapter_loaded = False
        adapter_repo = os.environ.get("VISUAL_IP_ADAPTER_REPO", "").strip()
        adapter_subfolder = os.environ.get("VISUAL_IP_ADAPTER_SUBFOLDER", "").strip()
        adapter_weight = os.environ.get("VISUAL_IP_ADAPTER_WEIGHT", "").strip()

        if adapter_repo and hasattr(pipeline, "load_ip_adapter"):
            kwargs: dict[str, Any] = {}
            if adapter_subfolder:
                kwargs["subfolder"] = adapter_subfolder
            if adapter_weight:
                kwargs["weight_name"] = adapter_weight
            pipeline.load_ip_adapter(adapter_repo, **kwargs)
            if hasattr(pipeline, "set_ip_adapter_scale"):
                scale = float(os.environ.get("VISUAL_IP_ADAPTER_SCALE", "0.75"))
                pipeline.set_ip_adapter_scale(max(0.0, min(1.0, scale)))
            ip_adapter_loaded = True

        _VISUAL_RUNTIME.update(
            {
                "initialized": True,
                "test_mode": False,
                "device": resolved_device,
                "model_name": model_name,
                "pipeline": pipeline,
                "ip_adapter_loaded": ip_adapter_loaded,
                "last_error": None,
            }
        )
        return {
            "initialized": True,
            "test_mode": False,
            "device": resolved_device,
            "model_name": model_name,
            "backend": "diffusers.FluxPipeline",
            "ip_adapter_loaded": ip_adapter_loaded,
        }
    except Exception as exc:
        _VISUAL_RUNTIME["last_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        flush_visual_resources()
        raise VisualPipelineError(
            "Failed to initialize the FLUX visual runtime. Verify PyTorch, "
            "Diffusers, model access, device support, and available VRAM. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc


def _load_anchor_image(path: Path) -> Any:
    """Load an anchor image with Pillow for production IP-Adapter input."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise VisualPipelineError(
            "Pillow is required to load reference images in production mode."
        ) from exc
    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except Exception as exc:
        raise VisualValidationError(
            f"Cannot read character reference image '{path}': {exc}"
        ) from exc


def establish_character_anchor(
    reference_image_path: str | Path | None,
    output_dir: str | Path,
    character_id: str,
    test_mode: bool = False,
) -> dict[str, Any]:
    """
    Establish a reusable character reference manifest.

    In test mode a deterministic synthetic image is generated if no reference
    image is supplied. Production mode requires a valid reference image and
    records its checksum for reproducible identity conditioning.
    """
    _ensure_visual_dirs()
    character_id = str(character_id).strip()
    if not character_id:
        raise VisualValidationError("character_id must not be empty.")

    safe_character = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in character_id
    )[:100]
    target_dir = _safe_path(output_dir, VISUAL_ROOT, allow_root=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    source_path: Path | None = None
    if reference_image_path is not None and str(reference_image_path).strip():
        candidate = Path(reference_image_path).expanduser()
        if not candidate.is_absolute():
            possible_paths = [
                (VISUAL_ROOT / candidate).resolve(),
                (PROJECT_ROOT / candidate).resolve(),
            ]
            source_path = next((item for item in possible_paths if item.is_file()), None)
        else:
            source_path = candidate.resolve()
        if source_path is None or not source_path.is_file():
            if not test_mode:
                raise FileNotFoundError(str(reference_image_path))
            source_path = None

    if source_path is not None:
        if source_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise VisualValidationError(
                f"Unsupported reference image type: {source_path.suffix}"
            )
        source_checksum = _sha256_file(source_path)
        width = 0
        height = 0
        if test_mode:
            try:
                if source_path.suffix.lower() == ".png":
                    width, height = _validate_external_image_dimensions(source_path)
                else:
                    from PIL import Image
                    with Image.open(source_path) as image:
                        width, height = int(image.width), int(image.height)
            except Exception:
                width, height = 0, 0
        else:
            width, height = _validate_external_image_dimensions(source_path)
    else:
        if not test_mode:
            raise VisualValidationError(
                "Production character anchoring requires a reference image."
            )
        source_checksum = None
        width = DEFAULT_FRAME_WIDTH
        height = DEFAULT_FRAME_HEIGHT

    anchor_id = f"anchor_{safe_character}"
    anchor_image_path = _safe_path(
        target_dir / f"{safe_character}_anchor.png",
        VISUAL_ROOT,
    )
    if source_path is not None:
        if source_path.suffix.lower() == ".png" and source_path.resolve() == anchor_image_path.resolve():
            pass
        elif test_mode:
            try:
                from PIL import Image
                with Image.open(source_path) as image:
                    image.convert("RGB").save(anchor_image_path, format="PNG")
            except ImportError:
                _atomic_write_bytes(
                    anchor_image_path,
                    _deterministic_png(width or 64, height or 64, DEFAULT_SEED, character_id),
                )
        else:
            try:
                from PIL import Image
                with Image.open(source_path) as image:
                    image.convert("RGB").save(anchor_image_path, format="PNG")
            except Exception as exc:
                raise VisualValidationError(
                    f"Failed to normalize reference image: {exc}"
                ) from exc
    else:
        _atomic_write_bytes(
            anchor_image_path,
            _deterministic_png(
                DEFAULT_FRAME_WIDTH,
                DEFAULT_FRAME_HEIGHT,
                int(hashlib.sha256(character_id.encode()).hexdigest()[:8], 16),
                character_id,
            ),
        )

    actual_width, actual_height = _validate_image_file(anchor_image_path)
    anchor = {
        "anchor_id": anchor_id,
        "character_id": character_id,
        "image": _file_metadata(
            anchor_image_path,
            asset_id=anchor_id,
            scene_id="character_anchor",
            asset_type="character_anchor",
            width=actual_width,
            height=actual_height,
            metadata={
                "source_sha256": source_checksum,
                "test_mode": bool(test_mode),
            },
        ),
        "source_sha256": source_checksum,
        "created_at_unix": int(time.time()),
        "test_mode": bool(test_mode),
    }

    manifest_path = _safe_path(
        target_dir / f"{safe_character}_anchor_manifest.json",
        VISUAL_ROOT,
    )
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "manifest_type": "character_anchor",
        "character_id": character_id,
        "anchors": [anchor],
        "anchor_image_path": str(anchor_image_path),
        "anchor_image_sha256": _sha256_file(anchor_image_path),
        "manifest_path": str(manifest_path),
        "test_mode": bool(test_mode),
    }
    _atomic_write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


def _validate_external_image_dimensions(path: Path) -> tuple[int, int]:
    """Read dimensions for an external reference image without copying it."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except ImportError as exc:
        raise VisualPipelineError(
            "Pillow is required to validate reference images."
        ) from exc
    except Exception as exc:
        raise VisualValidationError(f"Invalid reference image '{path}': {exc}") from exc


def _read_anchor_manifest(anchor_manifest: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Load and validate an anchor manifest supplied as a dictionary or path."""
    if isinstance(anchor_manifest, Mapping):
        payload = dict(anchor_manifest)
    else:
        path = Path(anchor_manifest).expanduser()
        if not path.is_absolute():
            path = VISUAL_ROOT / path
        path = _safe_path(path, VISUAL_ROOT, must_exist=True)
        payload = _read_json(path)

    anchors = payload.get("anchors", [])
    if not isinstance(anchors, list):
        raise VisualValidationError("Anchor manifest 'anchors' must be a list.")
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            raise VisualValidationError("Each anchor must be an object.")
        image_record = anchor.get("image", {})
        if isinstance(image_record, Mapping) and image_record.get("path"):
            image_path = _safe_path(image_record["path"], VISUAL_ROOT, must_exist=True)
            expected = image_record.get("sha256")
            if expected and _sha256_file(image_path) != expected:
                raise VisualValidationError(
                    f"Character anchor checksum mismatch: {image_path}"
                )
    return payload


def _scene_prompt(scene: Mapping[str, Any]) -> str:
    """Build a descriptive generation prompt from scene blueprint fields."""
    prompt_fields = (
        "visual_prompt",
        "image_prompt",
        "prompt",
        "description",
        "visual_description",
        "setting",
        "location",
        "mood",
        "cinematography",
        "camera",
        "lighting",
    )
    parts: list[str] = []
    for key in prompt_fields:
        value = scene.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    if not parts:
        title = _first_value(scene, ("title", "scene_title", "slug"), scene["scene_id"])
        parts.append(f"Cinematic storyboard frame for {title}")
    return ", ".join(parts)


def _scene_seed(base_seed: int, scene_id: str, frame_index: int) -> int:
    """Derive a stable frame seed independent of Python hash randomization."""
    digest = hashlib.sha256(
        f"{base_seed}:{scene_id}:{frame_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _get_ip_adapter_image(
    anchor_manifest: Mapping[str, Any],
    scene: Mapping[str, Any],
) -> Any | None:
    """Load the first scene-matched anchor image for IP-Adapter conditioning."""
    if not _VISUAL_RUNTIME.get("ip_adapter_loaded"):
        return None
    anchors = anchor_manifest.get("anchors", [])
    wanted = set(_anchor_ids_for_scene(scene, anchor_manifest))
    selected = None
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            continue
        if wanted and str(anchor.get("character_id")) not in wanted:
            continue
        image_record = anchor.get("image", {})
        if isinstance(image_record, Mapping) and image_record.get("path"):
            selected = _safe_path(image_record["path"], VISUAL_ROOT, must_exist=True)
            break
    if selected is None:
        return None
    return _load_anchor_image(selected)


def _generate_flux_frame(
    prompt: str,
    seed: int,
    output_path: Path,
    scene: Mapping[str, Any],
    anchor_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate one production frame using the initialized FLUX pipeline."""
    pipeline = _VISUAL_RUNTIME.get("pipeline")
    if pipeline is None:
        raise VisualPipelineError("Visual runtime is not initialized.")
    try:
        import torch
        from PIL import Image

        generator_device = (
            _VISUAL_RUNTIME["device"]
            if str(_VISUAL_RUNTIME["device"]).startswith("cuda")
            else "cpu"
        )
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        width = _as_int(
            _first_value(scene, ("frame_width", "width"), DEFAULT_FRAME_WIDTH),
            "frame width",
            64,
        )
        height = _as_int(
            _first_value(scene, ("frame_height", "height"), DEFAULT_FRAME_HEIGHT),
            "frame height",
            64,
        )
        width = min(2048, max(64, width - width % 8))
        height = min(2048, max(64, height - height % 8))
        inference_steps = _as_int(
            _first_value(scene, ("num_inference_steps", "steps"), 4),
            "inference steps",
            1,
        )
        guidance_scale = float(
            _first_value(scene, ("guidance_scale",), 0.0)
        )

        call_args: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": min(12, inference_steps),
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        adapter_image = _get_ip_adapter_image(anchor_manifest, scene)
        if adapter_image is not None:
            call_args["ip_adapter_image"] = adapter_image

        with torch.inference_mode():
            result = pipeline(**call_args)
        images = getattr(result, "images", None)
        if not images:
            raise VisualPipelineError("FLUX returned no images.")
        image = images[0]
        if not isinstance(image, Image.Image):
            raise VisualPipelineError("FLUX returned an unsupported image object.")
        image.save(output_path, format="PNG")
        return {
            "width": image.width,
            "height": image.height,
            "backend": "flux-schnell",
            "steps": min(12, inference_steps),
            "ip_adapter_used": adapter_image is not None,
        }
    except Exception as exc:
        raise VisualPipelineError(
            f"FLUX frame generation failed for scene "
            f"'{scene.get('scene_id')}': {type(exc).__name__}: {exc}"
        ) from exc


def _normalize_batch_size(batch_size: int) -> int:
    """Validate and cap the number of concurrently requested frames."""
    value = _as_int(batch_size, "batch_size", 1)
    return min(value, 8)


def generate_scene_frames(
    blueprint: Mapping[str, Any] | Any,
    audio_timeline: Mapping[str, Any] | Any,
    anchor_manifest: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    test_mode: bool = False,
) -> dict[str, Any]:
    """
    Generate deterministic storyboard frames synchronized to the audio timeline.

    The audio timeline is the authoritative source for scene duration and
    frame counts. Test mode writes valid PNGs without loading a model.
    """
    _ensure_visual_dirs()
    batch_size = _normalize_batch_size(batch_size)
    base_seed = _as_int(seed, "seed", 0)
    blueprint_data = _model_dump(blueprint)
    timeline_data = _model_dump(audio_timeline)
    anchors_data = _read_anchor_manifest(anchor_manifest)
    events = _extract_timeline_events(timeline_data)
    scenes = _scene_list(blueprint_data)

    target_dir = _safe_path(output_dir, VISUAL_ROOT, allow_root=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    scene_manifests: list[dict[str, Any]] = []
    all_frame_assets: list[dict[str, Any]] = []
    cursor = 0.0
    total_duration = 0.0
    total_frames = 0

    for scene_index, scene in enumerate(scenes, start=1):
        scene_id = str(scene["scene_id"])
        scene_number = int(scene["scene_number"])
        start, end = _scene_timing(scene, scene_index, events, cursor)
        duration = end - start
        fps = _as_float(
            _first_value(
                scene,
                ("fps", "frame_rate", "frames_per_second"),
                _first_value(timeline_data, ("fps", "frame_rate", "frames_per_second"), DEFAULT_FPS),
            ),
            f"{scene_id} fps",
            1.0,
        )
        if fps > 120:
            raise VisualValidationError(f"{scene_id} fps cannot exceed 120.")
        frame_count = max(1, int(math.ceil(duration * fps)))
        cues = _events_for_scene(scene, events, start, end)
        anchor_ids = _anchor_ids_for_scene(scene, anchors_data)
        prompt = _scene_prompt(scene)
        safe_scene_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in scene_id
        )[:100]
        scene_frame_dir = _safe_path(
            target_dir / "frames" / safe_scene_id,
            VISUAL_ROOT,
            allow_root=True,
        )
        scene_frame_dir.mkdir(parents=True, exist_ok=True)
        frame_assets: list[dict[str, Any]] = []

        for frame_index in range(frame_count):
            frame_seed = _scene_seed(base_seed, scene_id, frame_index)
            frame_path = _safe_path(
                scene_frame_dir / f"frame_{frame_index:06d}.png",
                VISUAL_ROOT,
            )
            if test_mode:
                _atomic_write_bytes(
                    frame_path,
                    _deterministic_png(
                        DEFAULT_FRAME_WIDTH,
                        DEFAULT_FRAME_HEIGHT,
                        frame_seed,
                        f"{scene_id}|{frame_index}|{prompt}",
                    ),
                )
                width, height = _validate_image_file(frame_path)
                generation_metadata = {
                    "backend": "synthetic-test-frame",
                    "test_mode": True,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            else:
                generation_metadata = _generate_flux_frame(
                    prompt,
                    frame_seed,
                    frame_path,
                    scene,
                    anchors_data,
                )
                width = int(generation_metadata["width"])
                height = int(generation_metadata["height"])

            asset = _file_metadata(
                frame_path,
                asset_id=f"{safe_scene_id}_frame_{frame_index:06d}",
                scene_id=scene_id,
                asset_type="storyboard_frame",
                duration_seconds=1.0 / fps,
                width=width,
                height=height,
                fps=fps,
                frame_count=1,
                seed=frame_seed,
                metadata={
                    **generation_metadata,
                    "frame_index": frame_index,
                    "frame_time_seconds": start + frame_index / fps,
                    "scene_start_seconds": start,
                    "scene_end_seconds": end,
                    "prompt": prompt,
                },
            )
            frame_assets.append(asset)
            all_frame_assets.append(asset)

            if not test_mode and (frame_index + 1) % batch_size == 0:
                flush_visual_resources()

        scene_manifest = VisualSceneManifest(
            scene_id=scene_id,
            scene_number=scene_number,
            start_seconds=start,
            end_seconds=end,
            duration_seconds=duration,
            frame_count=frame_count,
            fps=fps,
            character_anchor_ids=anchor_ids,
            frame_assets=frame_assets,
            motion_asset=None,
            audio_cues=cues,
            seed=_scene_seed(base_seed, scene_id, 0),
        )
        scene_payload = asdict(scene_manifest)
        scene_payload["prompt"] = prompt
        scene_payload["test_mode"] = bool(test_mode)
        scene_payload["timeline_source"] = "slot4_audio_timeline"
        scene_manifest_path = _safe_path(
            target_dir / "manifests" / f"{safe_scene_id}_visual_manifest.json",
            VISUAL_ROOT,
        )
        _atomic_write_json(scene_manifest_path, scene_payload)
        scene_payload["manifest_path"] = str(scene_manifest_path)
        scene_payload["manifest_sha256"] = _sha256_file(scene_manifest_path)
        scene_manifests.append(scene_payload)

        cursor = max(cursor, end)
        total_duration += duration
        total_frames += frame_count

    manifest_path = _safe_path(
        target_dir / "manifests" / "frame_manifest.json",
        VISUAL_ROOT,
    )
    frame_manifest = {
        "manifest_version": MANIFEST_VERSION,
        "manifest_type": "frame_manifest",
        "created_at_unix": int(time.time()),
        "test_mode": bool(test_mode),
        "timeline_source": str(
            _first_value(timeline_data, ("timeline_id", "project_id", "id"), "slot4_audio_timeline")
        ),
        "base_seed": base_seed,
        "batch_size": batch_size,
        "total_duration_seconds": total_duration,
        "total_frames": total_frames,
        "scenes": scene_manifests,
        "frame_assets": all_frame_assets,
        "manifest_path": str(manifest_path),
    }
    _atomic_write_json(manifest_path, frame_manifest)
    frame_manifest["manifest_sha256"] = _sha256_file(manifest_path)
    return frame_manifest


def _load_ltx_pipeline(model_name: str, device: str) -> Any:
    """Load the configured LTX-Video pipeline for production motion generation."""
    try:
        import torch
        from diffusers import LTXPipeline
    except ImportError as exc:
        raise VisualPipelineError(
            "Production motion generation requires PyTorch and a Diffusers "
            "version that exposes LTXPipeline."
        ) from exc

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    try:
        pipeline = LTXPipeline.from_pretrained(model_name, torch_dtype=dtype)
        pipeline = pipeline.to(device)
        return pipeline
    except Exception as exc:
        raise VisualPipelineError(
            f"Unable to load LTX-Video model '{model_name}': {exc}"
        ) from exc


def _resolve_frame_paths(frame_manifest: Mapping[str, Any]) -> list[Path]:
    """Resolve frame files from a generated frame manifest and verify checksums."""
    raw_assets = frame_manifest.get("frame_assets", [])
    if not isinstance(raw_assets, list):
        raise VisualValidationError("frame_manifest.frame_assets must be a list.")
    paths: list[Path] = []
    for item in raw_assets:
        if not isinstance(item, Mapping):
            continue
        raw_path = item.get("path")
        if not raw_path:
            continue
        path = _safe_path(raw_path, VISUAL_ROOT, must_exist=True)
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        expected = item.get("sha256")
        if expected and _sha256_file(path) != expected:
            raise VisualValidationError(f"Frame checksum mismatch: {path}")
        paths.append(path)
    return paths


def _create_motion_with_ltx(
    frame_paths: Sequence[Path],
    duration_seconds: float,
    output_path: Path,
    model_name: str,
    scene_id: str,
) -> dict[str, Any]:
    """Generate a scene video using LTX-Video, conditioning on the first frame."""
    if not frame_paths:
        raise VisualValidationError(f"No input frames for scene '{scene_id}'.")

    pipeline = _load_ltx_pipeline(
        model_name,
        str(_VISUAL_RUNTIME.get("device", "cuda")),
    )
    try:
        import torch
        from PIL import Image

        first_frame = _load_anchor_image(frame_paths[0])
        fps = DEFAULT_FPS
        num_frames = max(9, int(math.ceil(duration_seconds * fps)))
        num_frames = min(257, num_frames)
        if num_frames % 8 != 1:
            num_frames = max(9, ((num_frames - 1) // 8) * 8 + 1)

        generator_device = (
            _VISUAL_RUNTIME["device"]
            if str(_VISUAL_RUNTIME["device"]).startswith("cuda")
            else "cpu"
        )
        seed = int(hashlib.sha256(scene_id.encode()).hexdigest()[:8], 16)
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        call_args: dict[str, Any] = {
            "image": first_frame,
            "prompt": (
                f"Cinematic motion for {scene_id}; preserve subject identity, "
                "lighting, composition, and scene continuity."
            ),
            "num_frames": num_frames,
            "num_inference_steps": 20,
            "generator": generator,
        }
        with torch.inference_mode():
            result = pipeline(**call_args)

        frames = getattr(result, "frames", None)
        if frames is None:
            raise VisualPipelineError("LTX-Video returned no frames.")
        video_frames = frames[0] if isinstance(frames, list) and frames else frames
        if not isinstance(video_frames, (list, tuple)) or not video_frames:
            raise VisualPipelineError("LTX-Video returned an unsupported frame sequence.")

        try:
            from diffusers.utils import export_to_video
            export_to_video(list(video_frames), str(output_path), fps=fps)
        except Exception as export_error:
            try:
                import cv2
                import numpy as np

                height = int(video_frames[0].height)
                width = int(video_frames[0].width)
                temporary_path = output_path.with_name(
                    f".{output_path.stem}.{uuid.uuid4().hex}.tmp.mp4"
                )
                writer = cv2.VideoWriter(
                    str(temporary_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise VisualPipelineError("OpenCV video writer failed to open.")
                try:
                    for frame in video_frames:
                        array = np.asarray(frame.convert("RGB"))
                        writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
                finally:
                    writer.release()
                os.replace(temporary_path, output_path)
            except Exception as secondary_error:
                raise VisualPipelineError(
                    f"Video export failed: {export_error}; fallback failed: {secondary_error}"
                ) from secondary_error

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise VisualPipelineError("LTX-Video export created no output file.")
        return {
            "backend": "ltx-video",
            "model_name": model_name,
            "fps": fps,
            "frame_count": len(video_frames),
            "duration_seconds": duration_seconds,
            "test_placeholder": False,
        }
    finally:
        del pipeline
        flush_visual_resources()


def generate_scene_motion(
    frame_manifest: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    model_name: str = "Lightricks/LTX-Video",
    test_mode: bool = False,
) -> dict[str, Any]:
    """
    Produce motion clips from storyboard frames.

    In test mode each scene receives a deterministic placeholder video or a
    clearly labelled JSON motion placeholder if OpenCV is unavailable.
    Production mode uses LTX-Video and constrains the clip duration to the
    exact scene interval established from Slot 4's audio timeline.
    """
    _ensure_visual_dirs()
    if isinstance(frame_manifest, Mapping):
        manifest_data = dict(frame_manifest)
    else:
        path = Path(frame_manifest).expanduser()
        if not path.is_absolute():
            path = VISUAL_ROOT / path
        path = _safe_path(path, VISUAL_ROOT, must_exist=True)
        manifest_data = _read_json(path)

    scenes = manifest_data.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise VisualValidationError("Frame manifest must contain a non-empty scenes list.")

    target_dir = _safe_path(output_dir, VISUAL_ROOT, allow_root=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    scene_results: list[dict[str, Any]] = []
    all_motion_assets: list[dict[str, Any]] = []

    for scene in scenes:
        if not isinstance(scene, Mapping):
            raise VisualValidationError("Each scene in frame manifest must be an object.")
        scene_id = str(scene.get("scene_id", "")).strip()
        if not scene_id:
            raise VisualValidationError("Each scene manifest needs scene_id.")
        start = _as_float(scene.get("start_seconds", 0), f"{scene_id} start")
        end = _as_float(scene.get("end_seconds"), f"{scene_id} end")
        duration = end - start
        if duration <= 0:
            raise VisualValidationError(f"Scene '{scene_id}' has no positive duration.")

        safe_scene_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in scene_id
        )[:100]
        scene_motion_dir = _safe_path(
            target_dir / safe_scene_id,
            VISUAL_ROOT,
            allow_root=True,
        )
        scene_motion_dir.mkdir(parents=True, exist_ok=True)
        frame_assets = scene.get("frame_assets", [])
        if not isinstance(frame_assets, list):
            frame_assets = []
        frame_paths: list[Path] = []
        for asset in frame_assets:
            if isinstance(asset, Mapping) and asset.get("path"):
                asset_path = _safe_path(asset["path"], VISUAL_ROOT, must_exist=True)
                expected = asset.get("sha256")
                if expected and _sha256_file(asset_path) != expected:
                    raise VisualValidationError(
                        f"Frame checksum mismatch for scene '{scene_id}': {asset_path}"
                    )
                frame_paths.append(asset_path)

        if not frame_paths:
            raise VisualValidationError(f"Scene '{scene_id}' has no storyboard frames.")

        preferred_path = _safe_path(
            scene_motion_dir / f"{safe_scene_id}.mp4",
            VISUAL_ROOT,
        )
        if test_mode:
            seed = _scene_seed(
                int(manifest_data.get("base_seed", DEFAULT_SEED)),
                scene_id,
                0,
            )
            motion_metadata = _write_dummy_mp4(preferred_path, duration, seed)
            actual_path = (
                preferred_path
                if preferred_path.exists()
                else preferred_path.with_suffix(".motion.json")
            )
        else:
            if not _VISUAL_RUNTIME.get("initialized"):
                raise VisualPipelineError(
                    "Call initialize_visual_runtime(..., test_mode=False) before production motion."
                )
            actual_path = preferred_path
            motion_metadata = _create_motion_with_ltx(
                frame_paths,
                duration,
                actual_path,
                model_name,
                scene_id,
            )

        motion_asset = _file_metadata(
            actual_path,
            asset_id=f"{safe_scene_id}_motion",
            scene_id=scene_id,
            asset_type="scene_motion" if actual_path.suffix.lower() == ".mp4" else "scene_motion_placeholder",
            duration_seconds=duration,
            width=int(motion_metadata.get("width", 0)),
            height=int(motion_metadata.get("height", 0)),
            fps=float(motion_metadata.get("fps", DEFAULT_FPS)),
            frame_count=int(motion_metadata.get("frame_count", max(1, math.ceil(duration * DEFAULT_FPS)))),
            seed=_scene_seed(
                int(manifest_data.get("base_seed", DEFAULT_SEED)),
                scene_id,
                0,
            ),
            metadata=motion_metadata,
        )
        result = {
            "scene_id": scene_id,
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": duration,
            "motion_asset": motion_asset,
            "motion_path": str(actual_path),
            "motion_sha256": motion_asset["sha256"],
            "test_mode": bool(test_mode),
        }
        scene_results.append(result)
        all_motion_assets.append(motion_asset)

        scene_manifest_path = _safe_path(
            target_dir / "manifests" / f"{safe_scene_id}_visual_manifest.json",
            VISUAL_ROOT,
        )
        if scene_manifest_path.exists():
            scene_manifest_data = _read_json(scene_manifest_path)
        else:
            scene_manifest_data = dict(scene)
        scene_manifest_data["motion_asset"] = motion_asset
        scene_manifest_data["motion_path"] = str(actual_path)
        scene_manifest_data["motion_sha256"] = motion_asset["sha256"]
        scene_manifest_data["duration_seconds"] = duration
        scene_manifest_data["test_mode"] = bool(test_mode)
        _atomic_write_json(scene_manifest_path, scene_manifest_data)
        scene_manifest_data["manifest_path"] = str(scene_manifest_path)
        scene_manifest_data["manifest_sha256"] = _sha256_file(scene_manifest_path)

        flush_visual_resources()

    motion_manifest_path = _safe_path(
        target_dir / "manifests" / "motion_manifest.json",
        VISUAL_ROOT,
    )
    motion_manifest = {
        "manifest_version": MANIFEST_VERSION,
        "manifest_type": "motion_manifest",
        "created_at_unix": int(time.time()),
        "test_mode": bool(test_mode),
        "model_name": "deterministic-test-motion" if test_mode else model_name,
        "timeline_source": manifest_data.get("timeline_source", "slot4_audio_timeline"),
        "total_duration_seconds": sum(
            item["duration_seconds"] for item in scene_results
        ),
        "scenes": scene_results,
        "motion_assets": all_motion_assets,
        "manifest_path": str(motion_manifest_path),
    }
    _atomic_write_json(motion_manifest_path, motion_manifest)
    motion_manifest["manifest_sha256"] = _sha256_file(motion_manifest_path)
    return motion_manifest


def flush_visual_resources() -> dict[str, Any]:
    """
    Release Python and GPU resources after visual generation batches.

    CUDA cache clearing is attempted only when PyTorch is installed and CUDA
    is available. Garbage collection always runs.
    """
    result: dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "cuda_cache_cleared": False,
        "gc_collected": 0,
        "errors": [],
    }
    pipeline = _VISUAL_RUNTIME.get("pipeline")
    if pipeline is not None:
        try:
            if hasattr(pipeline, "maybe_free_model_hooks"):
                pipeline.maybe_free_model_hooks()
        except Exception as exc:
            result["errors"].append(
                f"pipeline cleanup: {type(exc).__name__}: {exc}"
            )

    try:
        import torch

        result["torch_available"] = True
        try:
            result["cuda_available"] = bool(torch.cuda.is_available())
        except Exception as exc:
            result["errors"].append(
                f"cuda availability: {type(exc).__name__}: {exc}"
            )
        if result["cuda_available"]:
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                result["errors"].append(
                    f"cuda synchronize: {type(exc).__name__}: {exc}"
                )
            try:
                torch.cuda.empty_cache()
                result["cuda_cache_cleared"] = True
            except Exception as exc:
                result["errors"].append(
                    f"cuda empty_cache: {type(exc).__name__}: {exc}"
                )
    except ImportError:
        pass
    except Exception as exc:
        result["errors"].append(f"torch cleanup: {type(exc).__name__}: {exc}")

    result["gc_collected"] = int(gc.collect())
    return result


def validate_visual_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate scene timings, required assets, and checksum integrity."""
    errors: list[str] = []
    warnings_list: list[str] = []
    scenes = manifest.get("scenes", [])
    if not isinstance(scenes, list):
        errors.append("scenes must be a list.")
        scenes = []

    previous_start = -1.0
    for index, scene in enumerate(scenes):
        if not isinstance(scene, Mapping):
            errors.append(f"scenes[{index}] must be an object.")
            continue
        scene_id = str(scene.get("scene_id", f"scene_{index + 1}"))
        try:
            start = _as_float(scene.get("start_seconds"), f"{scene_id} start")
            end = _as_float(scene.get("end_seconds"), f"{scene_id} end")
            duration = _as_float(scene.get("duration_seconds"), f"{scene_id} duration")
            if end <= start:
                errors.append(f"{scene_id}: end must be greater than start.")
            if not math.isclose(end - start, duration, rel_tol=1e-5, abs_tol=0.02):
                errors.append(f"{scene_id}: duration does not match its timing interval.")
            if start < previous_start:
                warnings_list.append(f"{scene_id}: scenes are not sorted by start time.")
            previous_start = start
        except VisualValidationError as exc:
            errors.append(str(exc))

        frame_assets = scene.get("frame_assets", [])
        if not isinstance(frame_assets, list) or not frame_assets:
            errors.append(f"{scene_id}: missing frame_assets.")
        else:
            for asset in frame_assets:
                if not isinstance(asset, Mapping):
                    errors.append(f"{scene_id}: invalid frame asset entry.")
                    continue
                try:
                    path = _safe_path(asset.get("path", ""), VISUAL_ROOT, must_exist=True)
                    expected = asset.get("sha256")
                    if expected and _sha256_file(path) != expected:
                        errors.append(f"{scene_id}: checksum mismatch for {path}.")
                except Exception as exc:
                    errors.append(f"{scene_id}: invalid frame path: {exc}")

        motion = scene.get("motion_asset")
        if motion is not None:
            if not isinstance(motion, Mapping):
                errors.append(f"{scene_id}: motion_asset must be an object.")
            else:
                try:
                    motion_path = _safe_path(
                        motion.get("path", ""),
                        VISUAL_ROOT,
                        must_exist=True,
                    )
                    expected = motion.get("sha256")
                    if expected and _sha256_file(motion_path) != expected:
                        errors.append(f"{scene_id}: motion checksum mismatch.")
                    motion_duration = float(motion.get("duration_seconds", duration))
                    if not math.isclose(
                        motion_duration, duration, rel_tol=1e-4, abs_tol=0.05
                    ):
                        errors.append(f"{scene_id}: motion duration differs from scene duration.")
                except Exception as exc:
                    errors.append(f"{scene_id}: invalid motion asset: {exc}")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings_list,
        "scene_count": len(scenes),
        "validated_at_unix": int(time.time()),
    }


def _manifest_path_from_record(record: Mapping[str, Any], key: str = "manifest_path") -> Path | None:
    """Resolve a manifest path if it is present and inside the visual sandbox."""
    raw = record.get(key)
    if not raw:
        return None
    return _safe_path(raw, VISUAL_ROOT, must_exist=True)


def run_visuals(
    screenplay_path: str | Path,
    timeline_path: str | Path,
    output_dir: str | Path,
    reference_image_path: str | Path | None = None,
    device: str = "auto",
    test_mode: bool = False,
) -> dict[str, Any]:
    """
    Execute Slot 3 from screenplay and Slot 4 audio timeline inputs.

    The output directory is constrained to workspace/04_visual_lab/. The
    screenplay and audio timeline are read-only inputs from their own slots.
    """
    _ensure_visual_dirs()
    screenplay_file = _resolve_screenplay_path(screenplay_path)
    timeline_file = _resolve_audio_timeline_path(timeline_path)
    output_root = _safe_path(output_dir, VISUAL_ROOT, allow_root=True)
    output_root.mkdir(parents=True, exist_ok=True)

    screenplay_data = _read_json(screenplay_file)
    timeline_data = _read_json(timeline_file)

    if not isinstance(screenplay_data, Mapping):
        raise VisualValidationError("Screenplay blueprint must be a JSON object.")
    if not isinstance(timeline_data, Mapping):
        raise VisualValidationError("Audio timeline must be a JSON object.")

    runtime_info = initialize_visual_runtime(
        device=device,
        model_name=os.environ.get(
            "VISUAL_FLUX_MODEL",
            "black-forest-labs/FLUX.1-schnell",
        ),
        test_mode=test_mode,
    )

    project_id = str(
        _first_value(
            screenplay_data,
            ("project_id", "project_name", "id"),
            _first_value(timeline_data, ("project_id", "project_name", "id"), "cinema_project"),
        )
    )
    safe_project_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in project_id
    )[:100]
    project_dir = _safe_path(output_root / safe_project_id, VISUAL_ROOT, allow_root=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    reference_manifest: dict[str, Any]
    if reference_image_path is not None and str(reference_image_path).strip():
        reference_manifest = establish_character_anchor(
            reference_image_path=reference_image_path,
            output_dir=project_dir / "anchors",
            character_id="primary_character",
            test_mode=test_mode,
        )
    elif test_mode:
        reference_manifest = establish_character_anchor(
            reference_image_path=None,
            output_dir=project_dir / "anchors",
            character_id="primary_character",
            test_mode=True,
        )
    else:
        reference_manifest = {
            "manifest_version": MANIFEST_VERSION,
            "manifest_type": "character_anchor",
            "anchors": [],
            "test_mode": False,
            "note": "No reference image supplied; character anchoring is disabled.",
        }
        _atomic_write_json(
            _safe_path(project_dir / "anchors" / "empty_anchor_manifest.json", VISUAL_ROOT),
            reference_manifest,
        )

    frame_manifest = generate_scene_frames(
        blueprint=screenplay_data,
        audio_timeline=timeline_data,
        anchor_manifest=reference_manifest,
        output_dir=project_dir,
        batch_size=_as_int(os.environ.get("VISUAL_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)), "VISUAL_BATCH_SIZE", 1),
        seed=_as_int(os.environ.get("VISUAL_SEED", str(DEFAULT_SEED)), "VISUAL_SEED", 0),
        test_mode=test_mode,
    )

    motion_manifest = generate_scene_motion(
        frame_manifest=frame_manifest,
        output_dir=project_dir / "motion",
        model_name=os.environ.get("VISUAL_LTX_MODEL", "Lightricks/LTX-Video"),
        test_mode=test_mode,
    )

    motion_by_scene = {
        str(scene.get("scene_id")): scene
        for scene in motion_manifest.get("scenes", [])
        if isinstance(scene, Mapping)
    }
    final_scenes: list[dict[str, Any]] = []
    for scene in frame_manifest.get("scenes", []):
        if not isinstance(scene, Mapping):
            continue
        combined = dict(scene)
        motion = motion_by_scene.get(str(scene.get("scene_id")))
        if motion:
            combined["motion_asset"] = motion.get("motion_asset")
            combined["motion_path"] = motion.get("motion_path")
            combined["motion_sha256"] = motion.get("motion_sha256")
        final_scenes.append(combined)

    final_manifest_path = _safe_path(
        project_dir / "manifests" / "visual_project_manifest.json",
        VISUAL_ROOT,
    )
    final_manifest = {
        "manifest_version": MANIFEST_VERSION,
        "manifest_type": "visual_project_manifest",
        "project_id": project_id,
        "created_at_unix": int(time.time()),
        "test_mode": bool(test_mode),
        "runtime": runtime_info,
        "inputs": {
            "screenplay_path": str(screenplay_file),
            "screenplay_sha256": _sha256_file(screenplay_file),
            "audio_timeline_path": str(timeline_file),
            "audio_timeline_sha256": _sha256_file(timeline_file),
        },
        "character_anchor_manifest": reference_manifest,
        "frame_manifest_path": frame_manifest.get("manifest_path"),
        "frame_manifest_sha256": frame_manifest.get("manifest_sha256"),
        "motion_manifest_path": motion_manifest.get("manifest_path"),
        "motion_manifest_sha256": motion_manifest.get("manifest_sha256"),
        "scenes": final_scenes,
        "total_duration_seconds": frame_manifest.get("total_duration_seconds", 0.0),
        "total_frames": frame_manifest.get("total_frames", 0),
        "output_root": str(project_dir),
        "manifest_path": str(final_manifest_path),
    }
    _atomic_write_json(final_manifest_path, final_manifest)
    final_manifest["manifest_sha256"] = _sha256_file(final_manifest_path)

    validation = validate_visual_manifest(final_manifest)
    validation_path = _safe_path(
        project_dir / "manifests" / "visual_validation.json",
        VISUAL_ROOT,
    )
    _atomic_write_json(validation_path, validation)
    final_manifest["validation"] = validation
    final_manifest["validation_path"] = str(validation_path)
    final_manifest["validation_sha256"] = _sha256_file(validation_path)

    if not validation["valid"]:
        _atomic_write_json(final_manifest_path, final_manifest)
        raise VisualValidationError(
            "Visual output validation failed: " + "; ".join(validation["errors"])
        )

    _atomic_write_json(final_manifest_path, final_manifest)
    final_manifest["manifest_sha256"] = _sha256_file(final_manifest_path)

    summary_path = _safe_path(
        project_dir / "manifests" / "visual_run_summary.json",
        VISUAL_ROOT,
    )
    summary = {
        "status": "success",
        "project_id": project_id,
        "test_mode": bool(test_mode),
        "scene_count": len(final_scenes),
        "total_frames": frame_manifest.get("total_frames", 0),
        "total_duration_seconds": frame_manifest.get("total_duration_seconds", 0.0),
        "manifest_path": str(final_manifest_path),
        "manifest_sha256": final_manifest["manifest_sha256"],
        "validation": validation,
        "resource_cleanup": flush_visual_resources(),
    }
    _atomic_write_json(summary_path, summary)

    return {
        **summary,
        "summary_path": str(summary_path),
        "summary_sha256": _sha256_file(summary_path),
        "visual_manifest": final_manifest,
        "scene_manifests": final_scenes,
        "frame_manifest": frame_manifest,
        "motion_manifest": motion_manifest,
        "character_anchor_manifest": reference_manifest,
    }


__all__ = [
    "VisualPipelineError",
    "VisualSandboxError",
    "VisualValidationError",
    "VisualAsset",
    "VisualSceneManifest",
    "initialize_visual_runtime",
    "establish_character_anchor",
    "generate_scene_frames",
    "generate_scene_motion",
    "flush_visual_resources",
    "validate_visual_manifest",
    "run_visuals",
]
