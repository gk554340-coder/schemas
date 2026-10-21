# File: slot2_screenplay.py
# Location: Project Root
# Purpose: Convert verified research into a validated, time-coded screenplay blueprint.
# Architecture: Slot 2 — Screenplay Desk
# Sandbox: workspace/02_screenplay_desk/
# Python: 3.11+

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


try:
    from config import StudioConfig, load_config
except ImportError:
    StudioConfig = None
    load_config = None

try:
    from schemas import (
        ScreenplayBlueprint,
        write_json_artifact,
    )
except ImportError as exc:
    raise ImportError(
        "slot2_screenplay.py requires ScreenplayBlueprint and "
        "write_json_artifact from schemas.py."
    ) from exc


LOGGER = logging.getLogger(__name__)

SLOT_NAME = "slot2_screenplay"
SANDBOX_RELATIVE_PATH = Path("workspace") / "02_screenplay_desk"
RESEARCH_SANDBOX_RELATIVE_PATH = Path("workspace") / "01_research_vault"

DEFAULT_LANGUAGE = "en"
DEFAULT_TOLERANCE_SECONDS = 1.0
DEFAULT_LONG_SCENES = 100
DEFAULT_REEL_SCENES = 7
MIN_LONG_SCENES = 80
MAX_LONG_SCENES = 120
MIN_REEL_SCENES = 6
MAX_REEL_SCENES = 8
MAX_RESEARCH_BYTES = 8_000_000
MAX_NARRATION_CHARS = 1800
MAX_VISUAL_PROMPT_CHARS = 1400


class ScreenplayError(RuntimeError):
    """Raised when screenplay planning or generation fails."""


class ScreenplaySandboxViolation(ScreenplayError):
    """Raised when an input or output path escapes its permitted sandbox."""


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def _iso_utc() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return _utc_now().isoformat().replace("+00:00", "Z")


def _clean_text(value: Any, limit: int = 12000) -> str:
    """Normalize text and remove markup-like fragments and control characters."""
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text[:limit]


def _validate_language(language: str) -> str:
    """Validate a language code such as en, hi, or en-US."""
    normalized = _clean_text(language, 32)

    if not normalized:
        raise ValueError("language must not be empty")

    if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", normalized):
        raise ValueError(
            "language must be a valid language code such as 'en' or 'hi'"
        )

    return normalized


def _validate_duration(target_duration_seconds: int | float) -> float:
    """Validate the target duration and return it as a float."""
    if isinstance(target_duration_seconds, bool) or not isinstance(
        target_duration_seconds, (int, float)
    ):
        raise TypeError("target_duration_seconds must be numeric")

    duration = float(target_duration_seconds)

    if duration <= 0 or duration > 24 * 60 * 60:
        raise ValueError(
            "target_duration_seconds must be greater than 0 and at most 86400"
        )

    return duration


def _validate_scene_count(target_scene_count: int) -> int:
    """Validate the requested scene count."""
    if isinstance(target_scene_count, bool) or not isinstance(
        target_scene_count, int
    ):
        raise TypeError("target_scene_count must be an integer")

    if not MIN_REEL_SCENES <= target_scene_count <= MAX_LONG_SCENES:
        raise ValueError(
            f"target_scene_count must be between {MIN_REEL_SCENES} "
            f"and {MAX_LONG_SCENES}"
        )

    return target_scene_count


def _validate_tolerance(tolerance_seconds: int | float) -> float:
    """Validate timeline tolerance."""
    if isinstance(tolerance_seconds, bool) or not isinstance(
        tolerance_seconds, (int, float)
    ):
        raise TypeError("tolerance_seconds must be numeric")

    tolerance = float(tolerance_seconds)

    if tolerance < 0 or tolerance > 300:
        raise ValueError("tolerance_seconds must be between 0 and 300")

    return tolerance


def _model_dump(value: Any) -> Any:
    """Convert schema models into JSON-compatible data."""
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
        return dict(value)

    if isinstance(value, (tuple, list)):
        return [_model_dump(item) for item in value]

    return value


def _construct_model(model_type: Any, payload: Mapping[str, Any]) -> Any:
    """Construct a schema model using only fields declared by that model."""
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


def _resolve_sandbox() -> Path:
    """Create and resolve the screenplay sandbox."""
    sandbox = (_project_root() / SANDBOX_RELATIVE_PATH).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)

    if not sandbox.is_dir():
        raise ScreenplayError(f"Screenplay sandbox is not a directory: {sandbox}")

    return sandbox


def _resolve_output_path(output_path: str | Path | None) -> Path:
    """Resolve an output path, rejecting paths outside the screenplay sandbox."""
    sandbox = _resolve_sandbox()

    if output_path is None:
        candidate = sandbox / "screenplay_blueprint.json"
    else:
        supplied = Path(output_path).expanduser()

        if supplied.is_absolute():
            candidate = supplied.resolve()
        else:
            candidate = (sandbox / supplied).resolve()

    try:
        candidate.relative_to(sandbox)
    except ValueError as exc:
        raise ScreenplaySandboxViolation(
            "Screenplay output must remain inside "
            f"{SANDBOX_RELATIVE_PATH.as_posix()}/"
        ) from exc

    if candidate == sandbox:
        raise ScreenplaySandboxViolation("Output path must identify a file")

    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _resolve_research_path(research_path: str | Path | None) -> Path:
    """Resolve a research artifact within the Slot 1 research sandbox."""
    project_root = _project_root()
    research_sandbox = (
        project_root / RESEARCH_SANDBOX_RELATIVE_PATH
    ).resolve()

    if research_path is None:
        candidate = research_sandbox / "research_output.json"
    else:
        supplied = Path(research_path).expanduser()

        if supplied.is_absolute():
            candidate = supplied.resolve()
        else:
            candidate = (research_sandbox / supplied).resolve()

    try:
        candidate.relative_to(research_sandbox)
    except ValueError as exc:
        raise ScreenplaySandboxViolation(
            "Research input must remain inside "
            f"{RESEARCH_SANDBOX_RELATIVE_PATH.as_posix()}/"
        ) from exc

    if not candidate.is_file():
        raise FileNotFoundError(f"Research artifact not found: {candidate}")

    return candidate


def _read_research_artifact(research_path: str | Path | None) -> dict[str, Any]:
    """Read a bounded JSON research artifact from the Slot 1 sandbox."""
    path = _resolve_research_path(research_path)

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ScreenplayError(f"Unable to inspect research artifact: {path}") from exc

    if size <= 0:
        raise ScreenplayError("Research artifact is empty")

    if size > MAX_RESEARCH_BYTES:
        raise ScreenplayError(
            f"Research artifact exceeds {MAX_RESEARCH_BYTES} bytes"
        )

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScreenplayError(
            f"Unable to read valid research JSON: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ScreenplayError("Research artifact must contain a JSON object")

    return payload


def _stable_id(*parts: str) -> str:
    """Generate a deterministic identifier."""
    raw = "\x1f".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:20]


def _as_mapping(value: Any) -> dict[str, Any]:
    """Normalize a dictionary or model object into a mapping."""
    dumped = _model_dump(value)

    if isinstance(dumped, Mapping):
        return dict(dumped)

    return {}


def _extract_research_material(research: Any) -> dict[str, Any]:
    """Normalize Slot 1 output into a consistent internal structure."""
    record = _as_mapping(research)

    topic = _clean_text(
        record.get("topic") or record.get("subject") or "Research topic",
        500,
    )
    language = _clean_text(record.get("language") or DEFAULT_LANGUAGE, 32)

    raw_facts = (
        record.get("facts")
        or record.get("evidence")
        or record.get("research_facts")
        or []
    )
    raw_sources = record.get("sources") or []
    raw_timeline = record.get("timeline") or record.get("timeline_events") or []
    raw_context = record.get("historical_context") or []
    raw_locations = record.get("locations") or []

    facts: list[dict[str, Any]] = []

    for item in raw_facts:
        fact = _as_mapping(item)

        claim = _clean_text(
            fact.get("claim")
            or fact.get("statement")
            or fact.get("text")
            or fact.get("description"),
            2000,
        )

        if not claim:
            continue

        facts.append(
            {
                "claim": claim,
                "source_url": _clean_text(
                    fact.get("source_url") or fact.get("url"),
                    2000,
                ),
                "source_title": _clean_text(
                    fact.get("source_title") or fact.get("title"),
                    500,
                ),
                "confidence": _safe_float(fact.get("confidence"), 0.5, 0.0, 1.0),
                "event_date": _clean_text(
                    fact.get("event_date") or fact.get("date"),
                    100,
                ),
                "verified": bool(fact.get("verified", False)),
            }
        )

    sources: list[dict[str, Any]] = []

    for item in raw_sources:
        source = _as_mapping(item)
        sources.append(
            {
                "title": _clean_text(source.get("title") or source.get("name"), 500),
                "url": _clean_text(source.get("url") or source.get("source_url"), 2000),
                "text": _clean_text(
                    source.get("text")
                    or source.get("content")
                    or source.get("snippet")
                    or source.get("description"),
                    6000,
                ),
            }
        )

    timeline: list[dict[str, Any]] = []

    for item in raw_timeline:
        event = _as_mapping(item)
        description = _clean_text(
            event.get("event")
            or event.get("description")
            or event.get("title")
            or event.get("summary"),
            1500,
        )

        if description:
            timeline.append(
                {
                    "date": _clean_text(
                        event.get("date") or event.get("event_date"),
                        100,
                    ),
                    "description": description,
                    "source_url": _clean_text(event.get("source_url"), 2000),
                }
            )

    contexts: list[str] = []

    for item in raw_context:
        context = _as_mapping(item)
        text = _clean_text(
            context.get("summary")
            or context.get("description")
            or context.get("context")
            or context.get("title")
            or (item if isinstance(item, str) else ""),
            1500,
        )

        if text:
            contexts.append(text)

    locations: list[str] = []

    for item in raw_locations:
        location = _as_mapping(item)
        name = _clean_text(
            location.get("name")
            or location.get("location")
            or (item if isinstance(item, str) else ""),
            200,
        )

        if name:
            locations.append(name)

    summary = _clean_text(record.get("summary"), 4000)

    if not summary and facts:
        summary = " ".join(fact["claim"] for fact in facts[:4])

    if not summary:
        summary = f"Research material collected for {topic}."

    return {
        "topic": topic,
        "language": language,
        "summary": summary,
        "facts": facts,
        "sources": sources,
        "timeline": timeline,
        "historical_context": contexts,
        "locations": locations,
        "test_mode": bool(record.get("test_mode", False)),
    }


def _safe_float(
    value: Any,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Convert a value into a bounded float."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default

    if result != result or result in (float("inf"), float("-inf")):
        result = default

    return max(minimum, min(maximum, result))


def _profile_kind(profile: Any) -> str:
    """Resolve a profile to either reel or long-form mode."""
    if isinstance(profile, Mapping):
        value = (
            profile.get("type")
            or profile.get("kind")
            or profile.get("profile")
            or profile.get("format")
            or profile.get("name")
            or ""
        )
    else:
        value = profile

    normalized = _clean_text(value, 128).lower()

    reel_terms = (
        "reel",
        "short",
        "shorts",
        "short-form",
        "vertical",
        "social",
        "tiktok",
    )

    if any(term in normalized for term in reel_terms):
        return "reel"

    if isinstance(profile, Mapping):
        target_count = profile.get("target_scene_count")
        if isinstance(target_count, int) and MIN_REEL_SCENES <= target_count <= MAX_REEL_SCENES:
            return "reel"

    return "long"


def _profile_scene_count(profile: Any) -> int:
    """Choose the scene count from a profile or the architecture defaults."""
    if isinstance(profile, Mapping):
        explicit = profile.get("target_scene_count") or profile.get("scene_count")

        if explicit is not None:
            return _validate_scene_count(int(explicit))

    if _profile_kind(profile) == "reel":
        return DEFAULT_REEL_SCENES

    return DEFAULT_LONG_SCENES


def _profile_narration_words_per_minute(profile: Any) -> int:
    """Choose a practical narration pace from the selected profile."""
    if isinstance(profile, Mapping):
        value = profile.get("narration_words_per_minute")

        if value is not None:
            try:
                pace = int(value)
                if 60 <= pace <= 220:
                    return pace
            except (TypeError, ValueError):
                pass

    return 135 if _profile_kind(profile) == "reel" else 125


def _allocate_scene_durations(
    target_duration_seconds: float,
    target_scene_count: int,
) -> list[float]:
    """Distribute duration evenly and preserve the exact total."""
    base_duration = target_duration_seconds / target_scene_count
    durations = [base_duration] * target_scene_count

    # Avoid cumulative floating-point drift by correcting the final scene.
    durations[-1] = target_duration_seconds - sum(durations[:-1])

    if any(duration <= 0 for duration in durations):
        raise ScreenplayError("Unable to allocate positive scene durations")

    return durations


def _format_timecode(seconds: float) -> str:
    """Format a time offset as HH:MM:SS.mmm."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)

    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _select_scene_material(
    research: Mapping[str, Any],
    scene_index: int,
    target_scene_count: int,
) -> dict[str, str]:
    """Select relevant source material deterministically for a scene."""
    facts = research["facts"]
    timeline = research["timeline"]
    contexts = research["historical_context"]
    locations = research["locations"]
    sources = research["sources"]

    if facts:
        fact = facts[scene_index % len(facts)]
        narration = fact["claim"]
        source_url = fact["source_url"]
        source_title = fact["source_title"]
    else:
        fact = {}
        narration = research["summary"]
        source_url = ""
        source_title = ""

    event = timeline[scene_index % len(timeline)] if timeline else {}
    context = contexts[scene_index % len(contexts)] if contexts else ""
    location = locations[scene_index % len(locations)] if locations else ""

    source = sources[scene_index % len(sources)] if sources else {}

    if not narration:
        narration = (
            f"This scene introduces an important part of "
            f"{research['topic']}."
        )

    return {
        "narration": _clean_text(narration, MAX_NARRATION_CHARS),
        "source_url": _clean_text(source_url or source.get("url"), 2000),
        "source_title": _clean_text(source_title or source.get("title"), 500),
        "event": _clean_text(event.get("description"), 1000),
        "context": _clean_text(context, 1000),
        "location": _clean_text(location, 200),
    }


def _scene_role(index: int, total: int) -> str:
    """Return a narrative role for a scene position."""
    if total <= 1:
        return "conclusion"

    fraction = index / max(1, total - 1)

    if index == 0:
        return "opening_hook"

    if index == total - 1:
        return "conclusion"

    if fraction < 0.18:
        return "introduction"

    if fraction < 0.42:
        return "context"

    if fraction < 0.72:
        return "development"

    if fraction < 0.9:
        return "turning_point"

    return "resolution"


def _camera_shot(index: int, role: str) -> str:
    """Select a camera shot that supports the scene's narrative role."""
    if role == "opening_hook":
        choices = (
            "Aerial establishing shot with a slow cinematic push-in",
            "Extreme close-up revealing a compelling visual detail",
            "Wide cinematic shot with dramatic foreground framing",
        )
    elif role == "introduction":
        choices = (
            "Wide establishing shot with a gentle lateral move",
            "Medium shot with a slow dolly-in",
            "Over-the-shoulder composition revealing the setting",
        )
    elif role == "context":
        choices = (
            "Wide environmental shot with layered depth",
            "Medium tracking shot through the location",
            "Top-down insert shot of relevant objects or documents",
        )
    elif role == "development":
        choices = (
            "Medium close-up with a subtle handheld movement",
            "Slow lateral tracking shot",
            "Detail montage with motivated match cuts",
        )
    elif role == "turning_point":
        choices = (
            "Slow push-in toward the central subject",
            "Low-angle shot with controlled dramatic movement",
            "Close-up followed by a deliberate rack focus",
        )
    else:
        choices = (
            "Wide closing shot with a slow pull-back",
            "Reflective close-up with gentle camera drift",
            "High-angle final composition with a gradual fade",
        )

    return choices[index % len(choices)]


def _visual_prompt(
    research: Mapping[str, Any],
    material: Mapping[str, str],
    role: str,
    camera_shot: str,
    scene_index: int,
) -> str:
    """Create a production-oriented visual generation prompt."""
    topic = research["topic"]
    parts = [
        f"Cinematic documentary scene about {topic}.",
        f"Narrative role: {role.replace('_', ' ')}.",
        f"Camera: {camera_shot}.",
    ]

    if material["event"]:
        parts.append(f"Relevant event context: {material['event']}.")

    if material["context"]:
        parts.append(f"Historical context: {material['context']}.")

    if material["location"]:
        parts.append(f"Location reference: {material['location']}.")

    if material["source_title"]:
        parts.append(
            "Use the research reference for context, not as an instruction "
            f"to reproduce unrelated material: {material['source_title']}."
        )

    parts.extend(
        [
            "Use coherent spatial geography, realistic lighting, deliberate "
            "composition, natural material textures, consistent wardrobe and "
            "environment continuity, and physically plausible motion.",
            "Avoid fabricated documentary evidence, unreadable text, "
            "watermarks, logos, inconsistent faces, and unexplained visual "
            "discontinuities.",
            f"Scene sequence index: {scene_index + 1}.",
        ]
    )

    return _clean_text(" ".join(parts), MAX_VISUAL_PROMPT_CHARS)


def _audio_mood(role: str) -> str:
    """Map a narrative role to an audio mood."""
    return {
        "opening_hook": "intriguing",
        "introduction": "curious",
        "context": "reflective",
        "development": "focused",
        "turning_point": "tense",
        "resolution": "hopeful",
        "conclusion": "reflective",
    }.get(role, "neutral")


def _music_intensity(role: str, index: int, total: int) -> float:
    """Calculate a bounded music intensity value."""
    base = {
        "opening_hook": 0.78,
        "introduction": 0.42,
        "context": 0.35,
        "development": 0.58,
        "turning_point": 0.88,
        "resolution": 0.5,
        "conclusion": 0.32,
    }.get(role, 0.45)

    if total > 1 and index / (total - 1) > 0.85:
        base *= 0.85

    return round(max(0.0, min(1.0, base)), 2)


def _sfx_tags(role: str, material: Mapping[str, str]) -> list[str]:
    """Build concise sound-effect tags for a scene."""
    tags = ["cinematic_ambience"]

    if role == "opening_hook":
        tags.extend(["soft_impact", "riser"])
    elif role == "context":
        tags.extend(["environmental_texture", "subtle_room_tone"])
    elif role == "development":
        tags.extend(["transition_whoosh", "natural_foley"])
    elif role == "turning_point":
        tags.extend(["low_boom", "tension_riser"])
    elif role in {"resolution", "conclusion"}:
        tags.extend(["gentle_transition", "soft_ambient_tail"])

    if material["location"]:
        tags.append("location_ambience")

    return list(dict.fromkeys(tags))


def _make_scene(
    research: Mapping[str, Any],
    scene_index: int,
    total_scenes: int,
    start_seconds: float,
    duration_seconds: float,
    language: str,
) -> dict[str, Any]:
    """Generate one complete time-coded scene blueprint."""
    role = _scene_role(scene_index, total_scenes)
    material = _select_scene_material(research, scene_index, total_scenes)
    shot = _camera_shot(scene_index, role)
    narration = material["narration"]

    scene_payload = {
        "id": _stable_id(
            research["topic"],
            str(scene_index + 1),
            _format_timecode(start_seconds),
        ),
        "scene_number": scene_index + 1,
        "index": scene_index + 1,
        "title": f"{role.replace('_', ' ').title()} — Scene {scene_index + 1}",
        "role": role,
        "start_seconds": round(start_seconds, 3),
        "end_seconds": round(start_seconds + duration_seconds, 3),
        "duration_seconds": round(duration_seconds, 3),
        "start_timecode": _format_timecode(start_seconds),
        "end_timecode": _format_timecode(start_seconds + duration_seconds),
        "timecode": {
            "start": _format_timecode(start_seconds),
            "end": _format_timecode(start_seconds + duration_seconds),
        },
        "camera_shot": shot,
        "visual_prompt": _visual_prompt(
            research,
            material,
            role,
            shot,
            scene_index,
        ),
        "narration": narration,
        "narration_language": language,
        "audio_mood": _audio_mood(role),
        "music_intensity": _music_intensity(role, scene_index, total_scenes),
        "sfx_tags": _sfx_tags(role, material),
        "source_references": [
            material["source_url"]
        ] if material["source_url"] else [],
        "source_titles": [
            material["source_title"]
        ] if material["source_title"] else [],
        "research_confidence": _safe_float(
            _select_scene_material(research, scene_index, total_scenes).get(
                "confidence"
            ),
            0.5,
            0.0,
            1.0,
        ),
    }

    return scene_payload


def _scene_payload(scene: Any) -> dict[str, Any]:
    """Normalize a scene model or dictionary."""
    return _as_mapping(scene)


def _blueprint_payload(blueprint: Any) -> dict[str, Any]:
    """Normalize a screenplay blueprint model into a dictionary."""
    payload = _as_mapping(blueprint)

    if not payload:
        raise ScreenplayError("Screenplay blueprint must be a mapping or model")

    return payload


def _extract_scenes(blueprint: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read scene records from supported blueprint field names."""
    raw_scenes = (
        blueprint.get("scenes")
        or blueprint.get("scene_blueprints")
        or blueprint.get("scene_list")
        or []
    )

    if not isinstance(raw_scenes, Sequence) or isinstance(
        raw_scenes, (str, bytes)
    ):
        raise ScreenplayError("Blueprint scenes must be a list")

    scenes = [_scene_payload(scene) for scene in raw_scenes]

    if not scenes:
        raise ScreenplayError("Screenplay blueprint has no scenes")

    return scenes


def plan_scene_structure(
    research: Any,
    target_duration_seconds: int | float,
    target_scene_count: int,
    profile: Any,
    test_mode: bool = False,
) -> dict[str, Any]:
    """
    Plan scene roles, durations, and narrative progression.

    Returns a structured scene plan. Reel profiles default to 6–8 scenes;
    long-form profiles default to 80–120 scenes. An explicit scene count
    takes precedence after validation.
    """
    duration = _validate_duration(target_duration_seconds)
    scene_count = _validate_scene_count(target_scene_count)
    material = _extract_research_material(research)
    profile_kind = _profile_kind(profile)

    if profile_kind == "reel" and not MIN_REEL_SCENES <= scene_count <= MAX_REEL_SCENES:
        raise ValueError(
            f"Reel profiles require {MIN_REEL_SCENES}–{MAX_REEL_SCENES} scenes"
        )

    if profile_kind == "long" and not MIN_LONG_SCENES <= scene_count <= MAX_LONG_SCENES:
        raise ValueError(
            f"Long-form profiles require {MIN_LONG_SCENES}–{MAX_LONG_SCENES} scenes"
        )

    durations = _allocate_scene_durations(duration, scene_count)
    scenes: list[dict[str, Any]] = []
    cursor = 0.0

    for index, scene_duration in enumerate(durations):
        role = _scene_role(index, scene_count)

        scenes.append(
            {
                "scene_number": index + 1,
                "role": role,
                "start_seconds": round(cursor, 3),
                "end_seconds": round(cursor + scene_duration, 3),
                "duration_seconds": round(scene_duration, 3),
                "start_timecode": _format_timecode(cursor),
                "end_timecode": _format_timecode(cursor + scene_duration),
            }
        )

        cursor += scene_duration

    return {
        "id": _stable_id(
            material["topic"],
            str(duration),
            str(scene_count),
            profile_kind,
        ),
        "topic": material["topic"],
        "profile": profile_kind,
        "profile_name": _clean_text(
            profile.get("name", profile_kind)
            if isinstance(profile, Mapping)
            else profile or profile_kind,
            128,
        ),
        "target_duration_seconds": duration,
        "target_scene_count": scene_count,
        "language": material["language"],
        "test_mode": bool(test_mode),
        "narrative_structure": [
            "opening_hook",
            "introduction",
            "context",
            "development",
            "turning_point",
            "resolution",
            "conclusion",
        ],
        "scenes": scenes,
        "created_at": _iso_utc(),
    }


def generate_scene_blueprints(
    research: Any,
    scene_plan: Mapping[str, Any],
    language: str,
    test_mode: bool = False,
) -> list[dict[str, Any]]:
    """
    Generate complete scene blueprints from research and a scene plan.

    Each scene includes timecodes, camera_shot, visual_prompt, narration,
    audio_mood, music_intensity, and sfx_tags.
    """
    language = _validate_language(language)
    material = _extract_research_material(research)

    if not isinstance(scene_plan, Mapping):
        raise TypeError("scene_plan must be a mapping")

    raw_scenes = scene_plan.get("scenes")

    if not isinstance(raw_scenes, Sequence) or isinstance(
        raw_scenes, (str, bytes)
    ):
        raise ScreenplayError("scene_plan must contain a scenes list")

    total_scenes = len(raw_scenes)

    if total_scenes == 0:
        raise ScreenplayError("scene_plan must contain at least one scene")

    if total_scenes > MAX_LONG_SCENES:
        raise ScreenplayError(
            f"scene_plan must not exceed {MAX_LONG_SCENES} scenes"
        )

    generated: list[dict[str, Any]] = []

    for index, raw_scene in enumerate(raw_scenes):
        planned = _as_mapping(raw_scene)

        start_seconds = _safe_float(
            planned.get("start_seconds"),
            0.0,
            0.0,
            24 * 60 * 60,
        )
        end_seconds = _safe_float(
            planned.get("end_seconds"),
            start_seconds,
            start_seconds,
            24 * 60 * 60,
        )

        if end_seconds <= start_seconds:
            raise ScreenplayError(
                f"Scene {index + 1} has an invalid time interval"
            )

        scene_duration = end_seconds - start_seconds
        generated.append(
            _make_scene(
                research=material,
                scene_index=index,
                total_scenes=total_scenes,
                start_seconds=start_seconds,
                duration_seconds=scene_duration,
                language=language,
            )
        )

    return generated


def validate_screenplay_timeline(
    blueprint: Any,
    target_duration_seconds: int | float,
    tolerance_seconds: int | float,
) -> dict[str, Any]:
    """
    Validate scene durations, ordering, continuity, and total runtime.

    Returns a validation report. The timeline is valid only if it has
    positive-duration scenes, no gaps or overlaps beyond the tolerance,
    and a final duration within the requested tolerance.
    """
    target_duration = _validate_duration(target_duration_seconds)
    tolerance = _validate_tolerance(tolerance_seconds)
    payload = _blueprint_payload(blueprint)
    scenes = _extract_scenes(payload)

    errors: list[str] = []
    warnings: list[str] = []
    normalized_intervals: list[tuple[float, float, int]] = []

    for index, scene in enumerate(scenes):
        number = scene.get("scene_number", scene.get("index", index + 1))

        start = scene.get("start_seconds")
        end = scene.get("end_seconds")

        if start is None or end is None:
            timecode = scene.get("timecode")
            if isinstance(timecode, Mapping):
                start = start if start is not None else _timecode_to_seconds(
                    timecode.get("start")
                )
                end = end if end is not None else _timecode_to_seconds(
                    timecode.get("end")
                )

        if start is None:
            start = _timecode_to_seconds(scene.get("start_timecode"))

        if end is None:
            end = _timecode_to_seconds(scene.get("end_timecode"))

        try:
            start_seconds = float(start)
            end_seconds = float(end)
        except (TypeError, ValueError):
            errors.append(f"Scene {number}: missing or invalid timecodes")
            continue

        if start_seconds < 0:
            errors.append(f"Scene {number}: start time is negative")

        if end_seconds <= start_seconds:
            errors.append(f"Scene {number}: end must be after start")

        if index == 0 and abs(start_seconds) > tolerance:
            errors.append(
                f"Scene {number}: timeline must begin at zero seconds"
            )

        if index > 0 and normalized_intervals:
            previous_end = normalized_intervals[-1][1]
            delta = start_seconds - previous_end

            if delta > tolerance:
                errors.append(
                    f"Scene {number}: gap of {delta:.3f} seconds"
                )
            elif delta < -tolerance:
                errors.append(
                    f"Scene {number}: overlap of {abs(delta):.3f} seconds"
                )
            elif abs(delta) > 0.001:
                warnings.append(
                    f"Scene {number}: minor boundary difference "
                    f"of {delta:.3f} seconds"
                )

        normalized_intervals.append(
            (start_seconds, end_seconds, int(number))
        )

        intensity = scene.get("music_intensity")

        if intensity is not None:
            try:
                numeric_intensity = float(intensity)
                if not 0.0 <= numeric_intensity <= 1.0:
                    errors.append(
                        f"Scene {number}: music_intensity must be in [0, 1]"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"Scene {number}: music_intensity must be numeric"
                )

        for required_field in (
            "camera_shot",
            "visual_prompt",
            "narration",
            "audio_mood",
            "sfx_tags",
        ):
            if required_field not in scene:
                errors.append(
                    f"Scene {number}: missing required field {required_field}"
                )

        if not _clean_text(scene.get("camera_shot")):
            errors.append(f"Scene {number}: camera_shot is empty")

        if not _clean_text(scene.get("visual_prompt")):
            errors.append(f"Scene {number}: visual_prompt is empty")

        if not _clean_text(scene.get("narration")):
            warnings.append(f"Scene {number}: narration is empty")

        tags = scene.get("sfx_tags")
        if not isinstance(tags, list):
            errors.append(f"Scene {number}: sfx_tags must be a list")

    if normalized_intervals:
        actual_start = normalized_intervals[0][0]
        actual_end = normalized_intervals[-1][1]
        actual_duration = actual_end - actual_start
        difference = actual_duration - target_duration

        if abs(difference) > tolerance:
            errors.append(
                f"Runtime differs from target by {difference:.3f} seconds "
                f"(tolerance: {tolerance:.3f})"
            )
    else:
        actual_start = 0.0
        actual_end = 0.0
        actual_duration = 0.0
        difference = -target_duration
        errors.append("No valid scene intervals were found")

    return {
        "valid": not errors,
        "scene_count": len(scenes),
        "target_duration_seconds": target_duration,
        "actual_duration_seconds": round(actual_duration, 3),
        "duration_difference_seconds": round(difference, 3),
        "tolerance_seconds": tolerance,
        "start_seconds": round(actual_start, 3),
        "end_seconds": round(actual_end, 3),
        "errors": errors,
        "warnings": warnings,
        "validated_at": _iso_utc(),
    }


def _timecode_to_seconds(value: Any) -> float | None:
    """Convert HH:MM:SS or HH:MM:SS.mmm to seconds."""
    if value is None:
        return None

    text = _clean_text(value, 64)
    match = re.fullmatch(
        r"(?:(\d{1,4}):)?([0-5]?\d):([0-5]?\d)(?:\.(\d{1,3}))?",
        text,
    )

    if not match:
        return None

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    fraction = (match.group(4) or "").ljust(3, "0")
    milliseconds = int(fraction or 0)

    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0


def _serialize_json(value: Any) -> bytes:
    """Serialize schema-validated output into strict JSON."""
    dumped = _model_dump(value)

    try:
        text = json.dumps(
            dumped,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ScreenplayError(
            "Screenplay blueprint cannot be serialized as strict JSON"
        ) from exc

    return (text + "\n").encode("utf-8")


def _atomic_write_json(output_path: Path, blueprint: Any) -> Path:
    """Atomically write the screenplay blueprint within the Slot 2 sandbox."""
    destination = _resolve_output_path(output_path)
    payload = _serialize_json(blueprint)

    file_descriptor: int | None = None
    temporary_path: Path | None = None

    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        temporary_path = Path(temporary_name)

        with os.fdopen(file_descriptor, "wb") as stream:
            file_descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_path, destination)
        temporary_path = None

        try:
            directory_descriptor = os.open(
                destination.parent,
                getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY,
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            pass

    except OSError as exc:
        raise ScreenplayError(
            f"Unable to atomically write screenplay artifact: {destination}"
        ) from exc

    finally:
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                pass

        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning(
                    "Unable to remove temporary file %s",
                    temporary_path,
                )

    return destination


def _write_artifact_compat(output_path: Path, blueprint: Any) -> Path:
    """Prefer the project helper when compatible, then perform atomic output."""
    if callable(write_json_artifact):
        try:
            import inspect

            signature = inspect.signature(write_json_artifact)
            parameter_names = set(signature.parameters)
            data = _model_dump(blueprint)

            if {"path", "data"}.issubset(parameter_names):
                write_json_artifact(path=output_path, data=data)
            elif {"output_path", "data"}.issubset(parameter_names):
                write_json_artifact(output_path=output_path, data=data)
            elif {"path", "payload"}.issubset(parameter_names):
                write_json_artifact(path=output_path, payload=data)

        except (TypeError, ValueError, OSError):
            LOGGER.debug(
                "Project JSON writer was not compatible; "
                "using the local atomic writer.",
                exc_info=True,
            )

    return _atomic_write_json(output_path, blueprint)


def _build_blueprint(
    research: Mapping[str, Any],
    scene_plan: Mapping[str, Any],
    scenes: Sequence[Mapping[str, Any]],
    language: str,
    test_mode: bool,
) -> Any:
    """Build the strict ScreenplayBlueprint schema model."""
    duration = _validate_duration(scene_plan["target_duration_seconds"])

    timeline_source = {
        "topic": research["topic"],
        "profile": scene_plan["profile"],
        "language": language,
        "target_duration_seconds": duration,
        "scenes": list(scenes),
    }

    validation = validate_screenplay_timeline(
        timeline_source,
        target_duration_seconds=duration,
        tolerance_seconds=DEFAULT_TOLERANCE_SECONDS,
    )

    if not validation["valid"]:
        raise ScreenplayError(
            "Generated screenplay timeline is invalid: "
            + "; ".join(validation["errors"])
        )

    payload = {
        "id": _stable_id(
            research["topic"],
            str(duration),
            str(len(scenes)),
            language,
        ),
        "topic": research["topic"],
        "title": research["topic"],
        "language": language,
        "profile": scene_plan["profile"],
        "target_duration_seconds": duration,
        "actual_duration_seconds": validation["actual_duration_seconds"],
        "target_scene_count": len(scenes),
        "scene_count": len(scenes),
        "scenes": list(scenes),
        "scene_blueprints": list(scenes),
        "research_summary": research["summary"],
        "research_sources": research["sources"],
        "research_fact_count": len(research["facts"]),
        "test_mode": bool(test_mode),
        "timeline_validation": validation,
        "created_at": _iso_utc(),
        "metadata": {
            "slot": SLOT_NAME,
            "sandbox": SANDBOX_RELATIVE_PATH.as_posix(),
            "scene_count": len(scenes),
            "profile": scene_plan["profile"],
            "timeline_valid": validation["valid"],
        },
    }

    blueprint = _construct_model(ScreenplayBlueprint, payload)

    # Serialization forces a JSON compatibility check before persistence.
    _serialize_json(blueprint)

    return blueprint


def run_screenplay(
    research_path: str | Path | None = None,
    output_path: str | Path | None = None,
    target_duration_seconds: int | float = 600,
    profile: Any = "long",
    language: str = DEFAULT_LANGUAGE,
    test_mode: bool = False,
) -> Any:
    """
    Execute the Slot 2 screenplay pipeline.

    Reads Slot 1 research_output.json, plans scene structure, generates
    complete scene blueprints, validates the timeline, and atomically writes
    a strict ScreenplayBlueprint artifact inside the Slot 2 sandbox.
    """
    started_at = time.monotonic()

    duration = _validate_duration(target_duration_seconds)
    language = _validate_language(language)

    research_payload = _read_research_artifact(research_path)
    research = _extract_research_material(research_payload)

    selected_profile = _profile_kind(profile)

    if isinstance(profile, Mapping):
        scene_count = _profile_scene_count(profile)
    elif selected_profile == "reel":
        scene_count = DEFAULT_REEL_SCENES
    else:
        scene_count = DEFAULT_LONG_SCENES

    scene_plan = plan_scene_structure(
        research=research,
        target_duration_seconds=duration,
        target_scene_count=scene_count,
        profile=profile,
        test_mode=test_mode,
    )

    scenes = generate_scene_blueprints(
        research=research,
        scene_plan=scene_plan,
        language=language,
        test_mode=test_mode,
    )

    blueprint = _build_blueprint(
        research=research,
        scene_plan=scene_plan,
        scenes=scenes,
        language=language,
        test_mode=test_mode,
    )

    destination = _resolve_output_path(output_path)
    _write_artifact_compat(destination, blueprint)

    LOGGER.info(
        "Screenplay completed: topic=%r profile=%s scenes=%d duration=%.3f "
        "elapsed_seconds=%.3f output=%s",
        research["topic"],
        selected_profile,
        len(scenes),
        duration,
        time.monotonic() - started_at,
        destination,
    )

    return blueprint


__all__ = [
    "ScreenplayError",
    "ScreenplaySandboxViolation",
    "plan_scene_structure",
    "generate_scene_blueprints",
    "validate_screenplay_timeline",
    "run_screenplay",
]
