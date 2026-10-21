# File: slot4_audio.py
# Location: Project Root
# Purpose: Generate narration, align spoken words, and build a validated audio timeline.
# Architecture: Slot 4 — Audio Studio
# Sandbox: workspace/03_audio_studio/
# Python: 3.11+

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


try:
    from schemas import AudioTimeline
except ImportError as exc:
    raise ImportError(
        "slot4_audio.py requires AudioTimeline from schemas.py"
    ) from exc


LOGGER = logging.getLogger(__name__)

SLOT_NAME = "slot4_audio"
SANDBOX_RELATIVE_PATH = Path("workspace") / "03_audio_studio"
SCREENPLAY_SANDBOX_RELATIVE_PATH = Path("workspace") / "02_screenplay_desk"

DEFAULT_VOICE = "en-US-AriaNeural"
DEFAULT_LANGUAGE = "en"
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_WORD_GAP_SECONDS = 0.035
DEFAULT_SILENCE_PADDING_SECONDS = 0.15
DEFAULT_MUSIC_DUCKING_FACTOR = 0.25
HIGH_TENSION_THRESHOLD = 0.80
MAX_SCREENPLAY_BYTES = 16_000_000
MAX_NARRATION_CHARS = 12000
MAX_AUDIO_DURATION_SECONDS = 24 * 60 * 60

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

HIGH_TENSION_MOODS = {
    "tense",
    "suspenseful",
    "urgent",
    "ominous",
    "dramatic_tension",
    "fearful",
    "intense",
    "thriller",
}

SFX_ANCHOR_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("impact", "boom", "hit", "crash"), "scene_start"),
    (("riser", "build", "tension"), "before_emphasis"),
    (("whoosh", "transition", "sweep"), "scene_transition"),
    (("ambient", "ambience", "room_tone"), "scene_start"),
    (("foley", "footstep", "rustle"), "word_anchor"),
    (("tail", "fade", "reverb"), "scene_end"),
)


class AudioStudioError(RuntimeError):
    """Raised when audio generation or timeline construction fails."""


class AudioSandboxViolation(AudioStudioError):
    """Raised when a path escapes the permitted audio workspace."""


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _iso_utc() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return _utc_now().isoformat().replace("+00:00", "Z")


def _clean_text(value: Any, limit: int = 12000) -> str:
    """Normalize text and remove control characters and markup."""
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text[:limit]


def _safe_float(
    value: Any,
    default: float = 0.0,
    minimum: float = 0.0,
    maximum: float = MAX_AUDIO_DURATION_SECONDS,
) -> float:
    """Convert a value into a finite float bounded by a specified range."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default

    if not math.isfinite(result):
        result = default

    return max(minimum, min(maximum, result))


def _model_dump(value: Any) -> Any:
    """Convert schema models and common containers into JSON-compatible data."""
    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()

    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)

    if isinstance(value, Mapping):
        return {key: _model_dump(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_model_dump(item) for item in value]

    return value


def _as_mapping(value: Any) -> dict[str, Any]:
    """Convert a mapping or schema model into a dictionary."""
    dumped = _model_dump(value)

    if isinstance(dumped, Mapping):
        return dict(dumped)

    return {}


def _construct_model(model_type: Any, payload: Mapping[str, Any]) -> Any:
    """Validate a payload using only fields declared by the target schema."""
    if hasattr(model_type, "model_fields"):
        fields = model_type.model_fields
        accepted = {key: value for key, value in payload.items() if key in fields}
        return model_type.model_validate(accepted)

    if hasattr(model_type, "__fields__"):
        fields = model_type.__fields__
        accepted = {key: value for key, value in payload.items() if key in fields}
        return model_type.parse_obj(accepted)

    return model_type(**dict(payload))


def _project_root() -> Path:
    """Return the project root containing this module."""
    return Path(__file__).resolve().parent


def _resolve_audio_sandbox() -> Path:
    """Create and resolve the Slot 4 audio workspace."""
    sandbox = (_project_root() / SANDBOX_RELATIVE_PATH).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)

    if not sandbox.is_dir():
        raise AudioStudioError(f"Audio sandbox is not a directory: {sandbox}")

    return sandbox


def _resolve_output_path(
    output_path: str | Path,
    *,
    create_parent: bool = True,
) -> Path:
    """Resolve an output path and reject any path outside the audio sandbox."""
    sandbox = _resolve_audio_sandbox()
    supplied = Path(output_path).expanduser()

    if supplied.is_absolute():
        candidate = supplied.resolve()
    else:
        candidate = (sandbox / supplied).resolve()

    try:
        candidate.relative_to(sandbox)
    except ValueError as exc:
        raise AudioSandboxViolation(
            f"Output must remain inside {SANDBOX_RELATIVE_PATH.as_posix()}/"
        ) from exc

    if candidate == sandbox:
        raise AudioSandboxViolation("Output path must identify a file or directory")

    if create_parent:
        candidate.parent.mkdir(parents=True, exist_ok=True)

    return candidate


def _resolve_output_directory(output_dir: str | Path) -> Path:
    """Resolve and create a directory inside the Slot 4 audio sandbox."""
    sandbox = _resolve_audio_sandbox()
    supplied = Path(output_dir).expanduser()

    if supplied.is_absolute():
        candidate = supplied.resolve()
    else:
        candidate = (sandbox / supplied).resolve()

    try:
        candidate.relative_to(sandbox)
    except ValueError as exc:
        raise AudioSandboxViolation(
            f"Audio output directory must remain inside "
            f"{SANDBOX_RELATIVE_PATH.as_posix()}/"
        ) from exc

    if candidate == sandbox:
        candidate.mkdir(parents=True, exist_ok=True)
    else:
        candidate.mkdir(parents=True, exist_ok=True)

    if not candidate.is_dir():
        raise AudioStudioError(f"Audio output path is not a directory: {candidate}")

    return candidate


def _resolve_screenplay_path(screenplay_path: str | Path) -> Path:
    """Resolve a screenplay input strictly inside the Slot 2 workspace."""
    sandbox = (
        _project_root() / SCREENPLAY_SANDBOX_RELATIVE_PATH
    ).resolve()

    supplied = Path(screenplay_path).expanduser()

    if supplied.is_absolute():
        candidate = supplied.resolve()
    else:
        candidate = (sandbox / supplied).resolve()

    try:
        candidate.relative_to(sandbox)
    except ValueError as exc:
        raise AudioSandboxViolation(
            "Screenplay input must remain inside "
            f"{SCREENPLAY_SANDBOX_RELATIVE_PATH.as_posix()}/"
        ) from exc

    if not candidate.is_file():
        raise FileNotFoundError(f"Screenplay artifact not found: {candidate}")

    if candidate.stat().st_size > MAX_SCREENPLAY_BYTES:
        raise AudioStudioError(
            f"Screenplay artifact exceeds {MAX_SCREENPLAY_BYTES} bytes"
        )

    return candidate


def _atomic_write_bytes(destination: Path, data: bytes) -> Path:
    """Atomically write bytes to a destination within the audio sandbox."""
    resolved = _resolve_output_path(destination)
    descriptor: int | None = None
    temporary_path: Path | None = None

    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            dir=str(resolved.parent),
        )
        temporary_path = Path(temporary_name)

        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_path, resolved)
        temporary_path = None

        try:
            directory_fd = os.open(
                resolved.parent,
                getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY,
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass

    except OSError as exc:
        raise AudioStudioError(
            f"Atomic write failed for {resolved}"
        ) from exc

    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Unable to remove temporary file %s", temporary_path)

    return resolved


def _atomic_write_json(output_path: str | Path, value: Any) -> Path:
    """Atomically write strict JSON inside the audio sandbox."""
    destination = _resolve_output_path(output_path)
    dumped = _model_dump(value)

    try:
        encoded = json.dumps(
            dumped,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise AudioStudioError("Audio artifact is not JSON serializable") from exc

    return _atomic_write_bytes(destination, encoded)


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read a bounded JSON object."""
    try:
        size = path.stat().st_size
        if size <= 0:
            raise AudioStudioError(f"JSON file is empty: {path}")
        if size > MAX_SCREENPLAY_BYTES:
            raise AudioStudioError(f"JSON file is too large: {path}")

        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AudioStudioError(f"Unable to read JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise AudioStudioError("Expected a JSON object")

    return payload


def _stable_id(*parts: str) -> str:
    """Generate a stable identifier for an audio artifact."""
    payload = "\x1f".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:20]


def _scene_list(blueprint: Any) -> list[dict[str, Any]]:
    """Extract scenes from a screenplay blueprint."""
    payload = _as_mapping(blueprint)
    scenes = (
        payload.get("scenes")
        or payload.get("scene_blueprints")
        or payload.get("scene_list")
    )

    if not isinstance(scenes, list) or not scenes:
        raise AudioStudioError("Screenplay blueprint must contain a non-empty scenes list")

    return [_as_mapping(scene) for scene in scenes]


def _scene_number(scene: Mapping[str, Any], index: int) -> int:
    """Return the best available scene number."""
    value = scene.get("scene_number", scene.get("index", index + 1))

    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return index + 1


def _scene_start(scene: Mapping[str, Any]) -> float:
    """Return a scene start offset in seconds."""
    value = scene.get("start_seconds")

    if value is None:
        value = scene.get("timecode", {})
        if isinstance(value, Mapping):
            value = value.get("start")

    if value is None:
        value = scene.get("start_timecode")

    return _parse_time(value, 0.0)


def _scene_end(scene: Mapping[str, Any], start: float) -> float:
    """Return a scene end offset in seconds."""
    value = scene.get("end_seconds")

    if value is None:
        value = scene.get("timecode", {})
        if isinstance(value, Mapping):
            value = value.get("end")

    if value is None:
        value = scene.get("end_timecode")

    duration = scene.get("duration_seconds")

    if value is None and duration is not None:
        return start + _safe_float(duration, 0.0, 0.0)

    return _parse_time(value, start)


def _parse_time(value: Any, default: float = 0.0) -> float:
    """Parse numeric seconds or HH:MM:SS[.mmm] timecodes."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _safe_float(value, default, 0.0)

    if value is None:
        return default

    text = _clean_text(value, 64)
    match = re.fullmatch(
        r"(?:(\d{1,4}):)?([0-5]?\d):([0-5]?\d)(?:\.(\d{1,3}))?",
        text,
    )

    if not match:
        return default

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    milliseconds = int((match.group(4) or "").ljust(3, "0") or "0")

    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0


def _format_timecode(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)

    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _normalize_language(language: str) -> str:
    """Validate and normalize a language tag."""
    value = _clean_text(language, 32)

    if not value:
        raise ValueError("language must not be empty")

    if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", value):
        raise ValueError("language must be a valid language tag such as 'en' or 'hi'")

    return value


def _word_tokens(text: str) -> list[str]:
    """Split narration into word-like tokens while retaining punctuation."""
    return re.findall(
        r"[^\W_]+(?:['’][^\W_]+)*(?:[-][^\W_]+)*|[^\w\s]",
        text,
        flags=re.UNICODE,
    )


def _estimate_speech_duration(text: str, language: str) -> float:
    """Estimate speech duration for deterministic CI fixtures."""
    words = _word_tokens(text)
    word_count = sum(1 for token in words if re.search(r"\w", token, re.UNICODE))

    if word_count == 0:
        return 0.25

    language_prefix = language.lower().split("-", 1)[0]
    words_per_minute = 145 if language_prefix in {"en", "de", "es", "fr"} else 125

    punctuation_pauses = len(re.findall(r"[,;:—]", text)) * 0.10
    sentence_pauses = len(re.findall(r"[.!?।]", text)) * 0.18

    return max(
        0.35,
        word_count * 60.0 / words_per_minute
        + punctuation_pauses
        + sentence_pauses,
    )


def _write_synthetic_wav(path: Path, duration_seconds: float = 0.25) -> Path:
    """Write a small valid silent PCM WAV file for offline tests."""
    destination = _resolve_output_path(path)
    sample_rate = 8000
    duration = max(0.05, min(2.0, float(duration_seconds)))
    sample_count = int(sample_rate * duration)
    silence = b"\x00\x00" * sample_count

    descriptor: int | None = None
    temporary_path: Path | None = None

    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        temporary_path = Path(temporary_name)

        with os.fdopen(descriptor, "wb") as raw_stream:
            descriptor = None
            with wave.open(raw_stream, "wb") as wav_stream:
                wav_stream.setnchannels(1)
                wav_stream.setsampwidth(2)
                wav_stream.setframerate(sample_rate)
                wav_stream.writeframes(silence)

            raw_stream.flush()
            os.fsync(raw_stream.fileno())

        os.replace(temporary_path, destination)
        temporary_path = None

    except (OSError, wave.Error) as exc:
        raise AudioStudioError(f"Unable to write synthetic WAV: {destination}") from exc

    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    return destination


def _check_audio_file(path: str | Path) -> Path:
    """Validate that an audio input is an existing supported file in the sandbox."""
    candidate = Path(path).expanduser().resolve()
    sandbox = _resolve_audio_sandbox()

    try:
        candidate.relative_to(sandbox)
    except ValueError as exc:
        raise AudioSandboxViolation(
            "Audio input must be located inside the Slot 4 audio sandbox"
        ) from exc

    if not candidate.is_file():
        raise FileNotFoundError(f"Audio file not found: {candidate}")

    if candidate.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise AudioStudioError(
            f"Unsupported audio extension: {candidate.suffix}"
        )

    return candidate


def _import_edge_tts() -> Any:
    """Import Edge-TTS only when production speech generation is requested."""
    try:
        import edge_tts
    except ImportError as exc:
        raise AudioStudioError(
            "Production narration requires edge-tts. Install it with "
            "'python -m pip install edge-tts'."
        ) from exc

    return edge_tts


def _import_faster_whisper() -> Any:
    """Import Faster-Whisper only when production alignment is requested."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AudioStudioError(
            "Production word alignment requires faster-whisper. Install it with "
            "'python -m pip install faster-whisper'."
        ) from exc

    return WhisperModel


def _run_async(coroutine: Any) -> Any:
    """Run an async function from this synchronous module."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # Avoid nesting event loops in callers that already use asyncio.
    raise AudioStudioError(
        "This synchronous audio pipeline cannot run inside an active event loop"
    )


def _safe_voice_name(voice: str) -> str:
    """Validate an Edge-TTS voice identifier."""
    normalized = _clean_text(voice, 128)

    if not normalized:
        raise ValueError("voice must not be empty")

    if not re.fullmatch(r"[A-Za-z0-9-]+", normalized):
        raise ValueError("voice contains unsupported characters")

    return normalized


def _narration_output_name(scene: Mapping[str, Any], index: int) -> str:
    """Create a deterministic narration filename."""
    number = _scene_number(scene, index)
    return f"narration_scene_{number:04d}.mp3"


def generate_narration(
    blueprint: Any,
    output_dir: str | Path,
    voice: str,
    language: str,
    test_mode: bool,
) -> list[dict[str, Any]]:
    """
    Generate one narration audio artifact per screenplay scene.

    Test mode writes tiny synthetic WAV files and deterministic timing
    metadata without importing Edge-TTS or downloading a speech model.
    Production mode uses Edge-TTS and writes MP3 files.
    """
    language = _normalize_language(language)
    output_directory = _resolve_output_directory(output_dir)
    scenes = _scene_list(blueprint)
    selected_voice = _safe_voice_name(voice or DEFAULT_VOICE)

    results: list[dict[str, Any]] = []

    for index, scene in enumerate(scenes):
        scene_number = _scene_number(scene, index)
        narration = _clean_text(
            scene.get("narration") or scene.get("voiceover") or scene.get("text"),
            MAX_NARRATION_CHARS,
        )

        filename = _narration_output_name(scene, index)

        if test_mode:
            filename = f"narration_scene_{scene_number:04d}.wav"

        destination = _resolve_output_path(output_directory / filename)

        if not narration:
            results.append(
                {
                    "scene_number": scene_number,
                    "text": "",
                    "audio_path": None,
                    "voice": selected_voice,
                    "language": language,
                    "start_seconds": _scene_start(scene),
                    "duration_seconds": 0.0,
                    "end_seconds": _scene_start(scene),
                    "word_count": 0,
                    "status": "skipped_empty_narration",
                    "test_mode": bool(test_mode),
                }
            )
            continue

        if test_mode:
            duration = _estimate_speech_duration(narration, language)
            _write_synthetic_wav(destination, duration_seconds=0.25)
            results.append(
                {
                    "id": _stable_id(str(scene_number), narration, language),
                    "scene_number": scene_number,
                    "text": narration,
                    "audio_path": str(destination),
                    "voice": selected_voice,
                    "language": language,
                    "start_seconds": _scene_start(scene),
                    "duration_seconds": round(duration, 3),
                    "end_seconds": round(_scene_start(scene) + duration, 3),
                    "word_count": sum(
                        1
                        for token in _word_tokens(narration)
                        if re.search(r"\w", token, re.UNICODE)
                    ),
                    "status": "synthetic_test_audio",
                    "test_mode": True,
                }
            )
            continue

        edge_tts = _import_edge_tts()

        async def _save_speech(
            text: str = narration,
            output_file: Path = destination,
            voice_name: str = selected_voice,
        ) -> None:
            communication = edge_tts.Communicate(text, voice_name)
            await communication.save(str(output_file))

        try:
            _run_async(_save_speech())
        except Exception as exc:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass

            raise AudioStudioError(
                f"Edge-TTS narration failed for scene {scene_number}"
            ) from exc

        if not destination.is_file() or destination.stat().st_size == 0:
            raise AudioStudioError(
                f"Edge-TTS did not produce a valid audio file for scene {scene_number}"
            )

        results.append(
            {
                "id": _stable_id(str(scene_number), narration, language),
                "scene_number": scene_number,
                "text": narration,
                "audio_path": str(destination),
                "voice": selected_voice,
                "language": language,
                "start_seconds": _scene_start(scene),
                "duration_seconds": None,
                "end_seconds": None,
                "word_count": sum(
                    1
                    for token in _word_tokens(narration)
                    if re.search(r"\w", token, re.UNICODE)
                ),
                "status": "generated",
                "test_mode": False,
            }
        )

    return results


def _resolve_alignment_output_dir(output_dir: str | Path) -> Path:
    """Resolve the word-alignment output directory inside the audio sandbox."""
    return _resolve_output_directory(output_dir)


def _synthetic_word_segments(
    text: str,
    start_seconds: float,
    language: str,
) -> list[dict[str, Any]]:
    """Create deterministic word-level timestamps without external models."""
    tokens = _word_tokens(text)
    meaningful_count = sum(
        1 for token in tokens if re.search(r"\w", token, re.UNICODE)
    )

    duration = _estimate_speech_duration(text, language)
    if meaningful_count == 0:
        return []

    per_word = duration / meaningful_count
    cursor = start_seconds
    segments: list[dict[str, Any]] = []

    for token in tokens:
        if not re.search(r"\w", token, re.UNICODE):
            continue

        word_duration = per_word
        end = cursor + word_duration

        segments.append(
            {
                "word": token,
                "start": round(cursor, 3),
                "end": round(end, 3),
                "start_timecode": _format_timecode(cursor),
                "end_timecode": _format_timecode(end),
                "confidence": 1.0,
                "synthetic": True,
            }
        )

        cursor = end + DEFAULT_WORD_GAP_SECONDS

    return segments


def _resolve_model_cache() -> Path:
    """Create a model-cache directory within the audio sandbox."""
    return _resolve_output_directory("models")


def _normalize_whisper_segments(
    whisper_segments: Any,
    *,
    offset_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert Faster-Whisper segments into word-level timestamp records."""
    result: list[dict[str, Any]] = []

    for segment in whisper_segments:
        words = getattr(segment, "words", None) or []

        for word in words:
            token = _clean_text(getattr(word, "word", ""), 256)
            if not token:
                continue

            start = _safe_float(
                getattr(word, "start", 0.0),
                0.0,
                0.0,
            ) + offset_seconds
            end = _safe_float(
                getattr(word, "end", start),
                start,
                start,
            ) + offset_seconds

            if end < start:
                continue

            result.append(
                {
                    "word": token,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "start_timecode": _format_timecode(start),
                    "end_timecode": _format_timecode(end),
                    "confidence": _safe_float(
                        getattr(word, "probability", 0.0),
                        0.0,
                        0.0,
                        1.0,
                    ),
                    "synthetic": False,
                }
            )

    return result


def align_narration_words(
    audio_paths: Sequence[str | Path | Mapping[str, Any]],
    output_dir: str | Path,
    language: str,
    model_name: str,
    test_mode: bool,
) -> list[dict[str, Any]]:
    """
    Produce word-level timestamps for narration audio.

    Test mode creates deterministic timing fixtures and does not load
    Faster-Whisper. Production mode loads Faster-Whisper and extracts
    word-level timestamps from each audio file.
    """
    language = _normalize_language(language)
    destination = _resolve_alignment_output_dir(output_dir)

    if isinstance(audio_paths, (str, bytes, Path)):
        audio_items: list[Any] = [audio_paths]
    else:
        audio_items = list(audio_paths)

    if not audio_items:
        return []

    results: list[dict[str, Any]] = []
    whisper_model: Any = None

    if not test_mode:
        WhisperModel = _import_faster_whisper()
        whisper_model = WhisperModel(
            model_name or DEFAULT_WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            download_root=str(_resolve_model_cache()),
        )

    for index, item in enumerate(audio_items):
        if isinstance(item, Mapping):
            record = dict(item)
            audio_path_value = record.get("audio_path") or record.get("path")
            narration = _clean_text(record.get("text"), MAX_NARRATION_CHARS)
            scene_number = int(record.get("scene_number", index + 1))
            offset_seconds = _safe_float(
                record.get("start_seconds"),
                0.0,
                0.0,
            )
            item_language = _normalize_language(
                record.get("language") or language
            )
        else:
            record = {}
            audio_path_value = item
            narration = ""
            scene_number = index + 1
            offset_seconds = 0.0
            item_language = language

        if not audio_path_value:
            if test_mode and narration:
                words = _synthetic_word_segments(
                    narration,
                    offset_seconds,
                    item_language,
                )
                results.append(
                    {
                        "scene_number": scene_number,
                        "audio_path": None,
                        "language": item_language,
                        "model_name": "synthetic",
                        "words": words,
                        "word_count": len(words),
                        "status": "synthetic_test_alignment",
                        "test_mode": True,
                    }
                )
                continue

            raise AudioStudioError(
                f"Audio path is missing for scene {scene_number}"
            )

        audio_path = _check_audio_file(audio_path_value)

        if not narration:
            narration = _clean_text(record.get("narration"), MAX_NARRATION_CHARS)

        if test_mode:
            if not narration:
                narration = f"Test narration scene {scene_number}"

            words = _synthetic_word_segments(
                narration,
                offset_seconds,
                item_language,
            )

            results.append(
                {
                    "scene_number": scene_number,
                    "audio_path": str(audio_path),
                    "language": item_language,
                    "model_name": "synthetic",
                    "words": words,
                    "word_count": len(words),
                    "status": "synthetic_test_alignment",
                    "test_mode": True,
                }
            )
            continue

        assert whisper_model is not None

        try:
            segments, info = whisper_model.transcribe(
                str(audio_path),
                language=item_language.split("-", 1)[0],
                word_timestamps=True,
                vad_filter=True,
                beam_size=1,
            )
            words = _normalize_whisper_segments(
                segments,
                offset_seconds=offset_seconds,
            )
        except Exception as exc:
            raise AudioStudioError(
                f"Faster-Whisper alignment failed for scene {scene_number}"
            ) from exc

        alignment_payload = {
            "scene_number": scene_number,
            "audio_path": str(audio_path),
            "language": item_language,
            "detected_language": getattr(info, "language", None),
            "model_name": model_name or DEFAULT_WHISPER_MODEL,
            "words": words,
            "word_count": len(words),
            "status": "aligned" if words else "no_words_detected",
            "test_mode": False,
        }

        alignment_path = _resolve_output_path(
            destination / f"word_alignment_scene_{scene_number:04d}.json"
        )
        _atomic_write_json(alignment_path, alignment_payload)
        alignment_payload["alignment_path"] = str(alignment_path)
        results.append(alignment_payload)

    return results


def _flatten_words(narration_segments: Sequence[Any]) -> list[dict[str, Any]]:
    """Flatten alignment results into a single ordered word list."""
    flattened: list[dict[str, Any]] = []

    for record_index, item in enumerate(narration_segments):
        record = _as_mapping(item)
        scene_number = int(record.get("scene_number", record_index + 1))
        words = record.get("words") or record.get("segments") or []

        if isinstance(words, list):
            for word_index, word_item in enumerate(words):
                word = _as_mapping(word_item)

                token = _clean_text(
                    word.get("word") or word.get("text"),
                    256,
                )

                if not token:
                    continue

                start = _safe_float(
                    word.get("start"),
                    0.0,
                    0.0,
                )
                end = _safe_float(
                    word.get("end"),
                    start,
                    start,
                )

                flattened.append(
                    {
                        "scene_number": scene_number,
                        "word_index": word_index,
                        "word": token,
                        "start": start,
                        "end": end,
                        "confidence": _safe_float(
                            word.get("confidence", word.get("probability")),
                            0.0,
                            0.0,
                            1.0,
                        ),
                        "synthetic": bool(word.get("synthetic", False)),
                    }
                )

    flattened.sort(key=lambda word: (word["start"], word["end"]))
    return flattened


def _scene_audio_metadata(
    scene: Mapping[str, Any],
    index: int,
    narration_segments: Sequence[Any],
) -> dict[str, Any]:
    """Find narration metadata matching a screenplay scene."""
    number = _scene_number(scene, index)

    for item in narration_segments:
        record = _as_mapping(item)
        try:
            if int(record.get("scene_number", -1)) == number:
                return record
        except (TypeError, ValueError):
            continue

    return {}


def _is_high_tension(scene: Mapping[str, Any]) -> bool:
    """Identify scenes requiring silence or near-silence."""
    mood = _clean_text(scene.get("audio_mood"), 128).lower()
    intensity = _safe_float(scene.get("music_intensity"), 0.0, 0.0, 1.0)
    role = _clean_text(scene.get("role"), 128).lower()

    return (
        mood in HIGH_TENSION_MOODS
        or intensity >= HIGH_TENSION_THRESHOLD
        or "high_tension" in role
        or "silence" in mood
    )


def _sfx_anchor(tag: str) -> str:
    """Map an SFX tag to an anchor strategy."""
    normalized = tag.lower()

    for keywords, anchor in SFX_ANCHOR_RULES:
        if any(keyword in normalized for keyword in keywords):
            return anchor

    return "scene_start"


def _music_ducking(
    scene: Mapping[str, Any],
    dialogue_active: bool,
) -> dict[str, Any]:
    """Calculate music volume and ducking for a scene."""
    base_intensity = _safe_float(
        scene.get("music_intensity"),
        0.35,
        0.0,
        1.0,
    )

    if _is_high_tension(scene):
        return {
            "base_volume": base_intensity,
            "volume": 0.0,
            "ducked": True,
            "ducking_factor": 0.0,
            "reason": "high_tension_silence",
        }

    if dialogue_active:
        # Reduce music by 75%, leaving 25% of its original level.
        return {
            "base_volume": base_intensity,
            "volume": round(base_intensity * DEFAULT_MUSIC_DUCKING_FACTOR, 3),
            "ducked": True,
            "ducking_factor": DEFAULT_MUSIC_DUCKING_FACTOR,
            "reason": "dialogue_ducking",
        }

    return {
        "base_volume": base_intensity,
        "volume": round(base_intensity, 3),
        "ducked": False,
        "ducking_factor": 1.0,
        "reason": "scene_base_level",
    }


def _word_intervals_for_scene(
    words: Sequence[Mapping[str, Any]],
    scene_number: int,
    scene_start: float,
    scene_end: float,
) -> list[dict[str, Any]]:
    """Return word intervals that overlap a scene."""
    output: list[dict[str, Any]] = []

    for word in words:
        if int(word.get("scene_number", -1)) != scene_number:
            continue

        start = _safe_float(word.get("start"), scene_start, 0.0)
        end = _safe_float(word.get("end"), start, start)

        if end < scene_start or start > scene_end:
            continue

        output.append(
            {
                "word": word.get("word", ""),
                "start": max(scene_start, start),
                "end": min(scene_end, end),
                "confidence": word.get("confidence", 0.0),
            }
        )

    return output


def _build_scene_audio_events(
    scene: Mapping[str, Any],
    index: int,
    narration_segments: Sequence[Any],
    words: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create dialogue, music, silence, and SFX events for one scene."""
    number = _scene_number(scene, index)
    start = _scene_start(scene)
    end = _scene_end(scene, start)

    if end <= start:
        raise AudioStudioError(
            f"Scene {number} has a non-positive duration"
        )

    metadata = _scene_audio_metadata(scene, index, narration_segments)
    audio_path = metadata.get("audio_path")
    narration_text = _clean_text(
        metadata.get("text") or scene.get("narration"),
        MAX_NARRATION_CHARS,
    )

    scene_words = _word_intervals_for_scene(words, number, start, end)
    dialogue_active = bool(narration_text or scene_words)

    events: list[dict[str, Any]] = []

    if dialogue_active:
        if scene_words:
            for word_index, word in enumerate(scene_words):
                events.append(
                    {
                        "id": _stable_id(
                            str(number),
                            "dialogue",
                            str(word_index),
                            str(word["start"]),
                        ),
                        "type": "dialogue",
                        "scene_number": number,
                        "start_seconds": round(word["start"], 3),
                        "end_seconds": round(word["end"], 3),
                        "start_timecode": _format_timecode(word["start"]),
                        "end_timecode": _format_timecode(word["end"]),
                        "text": word["word"],
                        "audio_path": audio_path,
                        "volume": 1.0,
                        "word_level": True,
                        "confidence": word["confidence"],
                    }
                )
        else:
            events.append(
                {
                    "id": _stable_id(str(number), "dialogue", str(start)),
                    "type": "dialogue",
                    "scene_number": number,
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "start_timecode": _format_timecode(start),
                    "end_timecode": _format_timecode(end),
                    "text": narration_text,
                    "audio_path": audio_path,
                    "volume": 1.0,
                    "word_level": False,
                }
            )

    music = _music_ducking(scene, dialogue_active)
    events.append(
        {
            "id": _stable_id(str(number), "music", str(start)),
            "type": "music",
            "scene_number": number,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "start_timecode": _format_timecode(start),
            "end_timecode": _format_timecode(end),
            "mood": _clean_text(scene.get("audio_mood") or "neutral", 128),
            "intensity": _safe_float(
                scene.get("music_intensity"),
                0.35,
                0.0,
                1.0,
            ),
            "base_volume": music["base_volume"],
            "volume": music["volume"],
            "ducked": music["ducked"],
            "ducking_factor": music["ducking_factor"],
            "mix_reason": music["reason"],
        }
    )

    if music["volume"] == 0.0:
        events.append(
            {
                "id": _stable_id(str(number), "silence", str(start)),
                "type": "silence",
                "scene_number": number,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "start_timecode": _format_timecode(start),
                "end_timecode": _format_timecode(end),
                "volume": 0.0,
                "reason": "high_tension_music_cut",
            }
        )

    tags = scene.get("sfx_tags") or []

    if not isinstance(tags, list):
        tags = []

    for tag_index, raw_tag in enumerate(tags):
        tag = _clean_text(raw_tag, 128)

        if not tag:
            continue

        anchor = _sfx_anchor(tag)

        if anchor == "scene_end":
            anchor_seconds = end
        elif anchor == "before_emphasis":
            anchor_seconds = max(start, start + (end - start) * 0.65)
        elif anchor == "scene_transition":
            anchor_seconds = max(start, end - DEFAULT_SILENCE_PADDING_SECONDS)
        elif anchor == "word_anchor" and scene_words:
            anchor_seconds = scene_words[0]["start"]
        else:
            anchor_seconds = start

        events.append(
            {
                "id": _stable_id(
                    str(number),
                    "sfx",
                    str(tag_index),
                    tag,
                ),
                "type": "sfx",
                "scene_number": number,
                "tag": tag,
                "anchor": anchor,
                "start_seconds": round(anchor_seconds, 3),
                "end_seconds": round(
                    min(end, anchor_seconds + 1.5),
                    3,
                ),
                "start_timecode": _format_timecode(anchor_seconds),
                "end_timecode": _format_timecode(
                    min(end, anchor_seconds + 1.5)
                ),
                "volume": 0.7 if not _is_high_tension(scene) else 0.3,
                "asset_path": None,
                "asset_status": "tag_only",
            }
        )

    return events


def _timeline_events(timeline: Any) -> list[dict[str, Any]]:
    """Extract audio events from a timeline mapping or model."""
    payload = _as_mapping(timeline)
    raw_events = payload.get("events") or payload.get("segments") or []

    if not isinstance(raw_events, list):
        raise AudioStudioError("Audio timeline events must be a list")

    return [_as_mapping(item) for item in raw_events]


def build_audio_timeline(
    blueprint: Any,
    narration_segments: Sequence[Any],
    output_dir: str | Path,
    test_mode: bool,
) -> Any:
    """
    Build the AudioTimeline from screenplay scenes and narration alignments.

    Applies music ducking during dialogue, zero music volume during high
    tension, and SFX anchors for scene starts, transitions, emphasis, words,
    and scene endings. Persists the timeline atomically inside Slot 4.
    """
    output_directory = _resolve_output_directory(output_dir)
    screenplay = _as_mapping(blueprint)
    scenes = _scene_list(blueprint)

    words = _flatten_words(narration_segments)
    events: list[dict[str, Any]] = []

    for index, scene in enumerate(scenes):
        events.extend(
            _build_scene_audio_events(
                scene=scene,
                index=index,
                narration_segments=narration_segments,
                words=words,
            )
        )

    events.sort(
        key=lambda event: (
            _safe_float(event.get("start_seconds"), 0.0, 0.0),
            str(event.get("type", "")),
            int(event.get("scene_number", 0)),
        )
    )

    timeline_payload = {
        "id": _stable_id(
            _clean_text(screenplay.get("id") or screenplay.get("topic"), 500),
            str(len(scenes)),
            "audio_timeline",
        ),
        "topic": _clean_text(
            screenplay.get("topic") or screenplay.get("title") or "Untitled",
            500,
        ),
        "language": _clean_text(screenplay.get("language") or DEFAULT_LANGUAGE, 32),
        "scene_count": len(scenes),
        "event_count": len(events),
        "events": events,
        "segments": events,
        "duration_seconds": max(
            (_scene_end(scene, _scene_start(scene)) for scene in scenes),
            default=0.0,
        ),
        "music_ducking_policy": {
            "dialogue_volume_factor": DEFAULT_MUSIC_DUCKING_FACTOR,
            "dialogue_reduction_percent": 75,
            "high_tension_music_volume": 0.0,
            "high_tension_threshold": HIGH_TENSION_THRESHOLD,
        },
        "test_mode": bool(test_mode),
        "created_at": _iso_utc(),
        "metadata": {
            "slot": SLOT_NAME,
            "sandbox": SANDBOX_RELATIVE_PATH.as_posix(),
            "word_alignment_count": len(words),
            "synthetic_alignment": bool(test_mode),
        },
    }

    timeline_model = _construct_model(AudioTimeline, timeline_payload)

    validation = validate_audio_timeline(timeline_model, blueprint)

    if not validation["valid"]:
        raise AudioStudioError(
            "Audio timeline validation failed: "
            + "; ".join(validation["errors"])
        )

    # Rebuild with the validation report if the schema supports that field.
    timeline_payload["validation"] = validation
    timeline_model = _construct_model(AudioTimeline, timeline_payload)

    timeline_path = _resolve_output_path(
        output_directory / "audio_timeline.json"
    )
    _atomic_write_json(timeline_path, timeline_model)

    return timeline_model


def validate_audio_timeline(
    timeline: Any,
    blueprint: Any,
) -> dict[str, Any]:
    """
    Validate the audio timeline against the screenplay blueprint.

    Checks event time ranges, required event data, scene references, music
    ducking, high-tension silence, and SFX anchors.
    """
    payload = _as_mapping(timeline)
    events = _timeline_events(timeline)
    scenes = _scene_list(blueprint)

    errors: list[str] = []
    warnings: list[str] = []
    scene_numbers = {
        _scene_number(scene, index)
        for index, scene in enumerate(scenes)
    }

    scene_map = {
        _scene_number(scene, index): scene
        for index, scene in enumerate(scenes)
    }

    if not events:
        errors.append("Audio timeline contains no events")

    for index, event in enumerate(events):
        event_type = _clean_text(event.get("type"), 64).lower()

        if not event_type:
            errors.append(f"Event {index + 1}: missing event type")

        start = _safe_float(
            event.get("start_seconds"),
            -1.0,
            -1.0,
            MAX_AUDIO_DURATION_SECONDS,
        )
        end = _safe_float(
            event.get("end_seconds"),
            -1.0,
            -1.0,
            MAX_AUDIO_DURATION_SECONDS,
        )

        if start < 0:
            errors.append(f"Event {index + 1}: invalid start_seconds")

        if end < 0 or end < start:
            errors.append(f"Event {index + 1}: invalid end_seconds")

        if start > MAX_AUDIO_DURATION_SECONDS or end > MAX_AUDIO_DURATION_SECONDS:
            errors.append(f"Event {index + 1}: event exceeds maximum duration")

        scene_number_value = event.get("scene_number")

        try:
            scene_number = int(scene_number_value)
        except (TypeError, ValueError):
            errors.append(f"Event {index + 1}: invalid scene_number")
            continue

        if scene_number not in scene_numbers:
            errors.append(
                f"Event {index + 1}: references unknown scene {scene_number}"
            )
            continue

        volume = event.get("volume")

        if volume is not None:
            numeric_volume = _safe_float(volume, -1.0, -1.0, 2.0)
            if not 0.0 <= numeric_volume <= 1.0:
                errors.append(f"Event {index + 1}: volume must be in [0, 1]")

        if event_type == "music":
            factor = _safe_float(
                event.get("ducking_factor"),
                1.0,
                0.0,
                1.0,
            )

            scene = scene_map[scene_number]

            if _is_high_tension(scene) and _safe_float(
                event.get("volume"),
                1.0,
                0.0,
                1.0,
            ) != 0.0:
                errors.append(
                    f"Scene {scene_number}: high-tension music must be silent"
                )

            if event.get("ducked") and factor > 0.30:
                errors.append(
                    f"Scene {scene_number}: dialogue music ducking must "
                    "reduce music by at least 70%"
                )

        if event_type == "sfx":
            if not _clean_text(event.get("tag"), 128):
                errors.append(f"Event {index + 1}: SFX tag is missing")

            if not _clean_text(event.get("anchor"), 128):
                warnings.append(
                    f"Event {index + 1}: SFX event has no timing anchor"
                )

        if event_type == "dialogue":
            if not _clean_text(event.get("text"), 2000):
                warnings.append(
                    f"Event {index + 1}: dialogue event has no text"
                )

    dialogue_scenes = {
        int(event["scene_number"])
        for event in events
        if event.get("type") == "dialogue"
        and isinstance(event.get("scene_number"), (int, float, str))
        and str(event.get("scene_number")).isdigit()
    }

    music_events = [
        event for event in events if event.get("type") == "music"
    ]

    for music_event in music_events:
        try:
            number = int(music_event.get("scene_number"))
        except (TypeError, ValueError):
            continue

        scene = scene_map.get(number)

        if scene is None:
            continue

        if number in dialogue_scenes:
            if not music_event.get("ducked") and not _is_high_tension(scene):
                warnings.append(
                    f"Scene {number}: dialogue is present but music is not ducked"
                )

    return {
        "valid": not errors,
        "scene_count": len(scenes),
        "event_count": len(events),
        "dialogue_scene_count": len(dialogue_scenes),
        "music_event_count": len(music_events),
        "target_duration_seconds": _safe_float(
            _as_mapping(blueprint).get("target_duration_seconds"),
            0.0,
            0.0,
        ),
        "timeline_duration_seconds": _safe_float(
            payload.get("duration_seconds"),
            0.0,
            0.0,
        ),
        "errors": errors,
        "warnings": warnings,
        "validated_at": _iso_utc(),
    }


def _resolve_timeline_output_path(
    timeline_path: str | Path | None,
    output_directory: Path,
) -> Path:
    """Resolve a timeline path inside the Slot 4 sandbox."""
    if timeline_path is None:
        return _resolve_output_path(output_directory / "audio_timeline.json")

    return _resolve_output_path(timeline_path)


def _estimate_audio_file_duration(path: Path) -> float | None:
    """Read WAV duration when available; return None for unsupported metadata."""
    if path.suffix.lower() != ".wav":
        return None

    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()

        if sample_rate <= 0:
            return None

        return frames / sample_rate

    except (OSError, wave.Error):
        return None


def _align_generated_narration(
    narration_records: Sequence[Mapping[str, Any]],
    output_directory: Path,
    language: str,
    test_mode: bool,
) -> list[dict[str, Any]]:
    """Align generated narration using the selected test or production path."""
    return align_narration_words(
        audio_paths=narration_records,
        output_dir=output_directory / "alignments",
        language=language,
        model_name=DEFAULT_WHISPER_MODEL,
        test_mode=test_mode,
    )


def run_audio(
    screenplay_path: str | Path,
    output_dir: str | Path,
    timeline_path: str | Path | None,
    voice: str,
    language: str,
    test_mode: bool,
) -> Any:
    """
    Run the complete Slot 4 audio workflow.

    Loads a screenplay artifact from Slot 2, generates narration, aligns
    narration words, builds and validates AudioTimeline, then writes the
    timeline atomically inside workspace/03_audio_studio/.
    """
    started_at = time.monotonic()
    language = _normalize_language(language)
    output_directory = _resolve_output_directory(output_dir)

    screenplay_file = _resolve_screenplay_path(screenplay_path)
    screenplay_payload = _read_json_file(screenplay_file)

    if not (
        screenplay_payload.get("scenes")
        or screenplay_payload.get("scene_blueprints")
        or screenplay_payload.get("scene_list")
    ):
        raise AudioStudioError("Screenplay artifact contains no scenes")

    narration_records = generate_narration(
        blueprint=screenplay_payload,
        output_dir=output_directory / "narration",
        voice=voice or DEFAULT_VOICE,
        language=language,
        test_mode=test_mode,
    )

    alignment_records = _align_generated_narration(
        narration_records=narration_records,
        output_directory=output_directory,
        language=language,
        test_mode=test_mode,
    )

    timeline_model = build_audio_timeline(
        blueprint=screenplay_payload,
        narration_segments=alignment_records,
        output_dir=output_directory,
        test_mode=test_mode,
    )

    validation = validate_audio_timeline(
        timeline=timeline_model,
        blueprint=screenplay_payload,
    )

    if not validation["valid"]:
        raise AudioStudioError(
            "Final audio timeline is invalid: "
            + "; ".join(validation["errors"])
        )

    destination = _resolve_timeline_output_path(
        timeline_path,
        output_directory,
    )
    _atomic_write_json(destination, timeline_model)

    # Save a concise run report alongside the timeline.
    report = {
        "slot": SLOT_NAME,
        "status": "success",
        "screenplay_path": str(screenplay_file),
        "output_directory": str(output_directory),
        "timeline_path": str(destination),
        "scene_count": len(_scene_list(screenplay_payload)),
        "narration_file_count": sum(
            1 for record in narration_records if record.get("audio_path")
        ),
        "aligned_word_count": sum(
            int(record.get("word_count", 0))
            for record in alignment_records
        ),
        "timeline_event_count": len(_timeline_events(timeline_model)),
        "validation": validation,
        "test_mode": bool(test_mode),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "completed_at": _iso_utc(),
    }

    _atomic_write_json(output_directory / "audio_run_report.json", report)

    LOGGER.info(
        "Audio workflow completed: scenes=%d narration_files=%d "
        "events=%d elapsed_seconds=%.3f output=%s",
        report["scene_count"],
        report["narration_file_count"],
        report["timeline_event_count"],
        report["elapsed_seconds"],
        destination,
    )

    return timeline_model


__all__ = [
    "AudioStudioError",
    "AudioSandboxViolation",
    "generate_narration",
    "align_narration_words",
    "build_audio_timeline",
    "validate_audio_timeline",
    "run_audio",
]
