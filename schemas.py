# schemas.py
"""Strict data models, artifact validation, and atomic JSON artifact I/O."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


SchemaModel = TypeVar("SchemaModel", bound="StrictModel")

SHA256_PATTERN = r"^[a-fA-F0-9]{64}$"


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and validates assignments."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class ResearchSource(StrictModel):
    """A source used to support research claims."""

    source_id: str = Field(pattern=r"^SRC_[A-Za-z0-9_-]+$")
    source_type: Literal[
        "web",
        "news_rss",
        "scripture_api",
        "official_document",
        "book",
        "archive",
        "other",
    ]
    title: str = Field(min_length=1, max_length=1000)
    url: str | None = Field(default=None, max_length=4000)
    author: str | None = Field(default=None, max_length=500)
    publisher: str | None = Field(default=None, max_length=500)
    published_at: datetime | None = None
    retrieved_at: datetime
    language: str = Field(min_length=2, max_length=35)
    excerpt: str | None = Field(default=None, max_length=20000)
    content_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    reliability_score: float = Field(default=0.5, ge=0.0, le=1.0)
    verification_status: Literal[
        "verified",
        "partially_verified",
        "unverified",
        "disputed",
    ] = "unverified"


class ResearchFact(StrictModel):
    """A research claim with source references and verification metadata."""

    fact_id: str = Field(pattern=r"^FACT_[A-Za-z0-9_-]+$")
    statement: str = Field(min_length=1, max_length=10000)
    source_ids: list[str] = Field(min_length=1)
    verification_status: Literal[
        "verified",
        "partially_verified",
        "unverified",
        "disputed",
    ] = "unverified"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    category: str = Field(default="general", min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=4000)

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, values: list[str]) -> list[str]:
        """Require unique, correctly formatted source references."""
        if len(values) != len(set(values)):
            raise ValueError("source_ids must not contain duplicates.")
        if any(not value.startswith("SRC_") for value in values):
            raise ValueError("Every source reference must start with 'SRC_'.")
        return values


class TimelineEvent(StrictModel):
    """A chronological event supported by research."""

    event_id: str = Field(pattern=r"^EVENT_[A-Za-z0-9_-]+$")
    date_text: str = Field(min_length=1, max_length=300)
    date_start: datetime | None = None
    date_end: datetime | None = None
    title: str = Field(min_length=1, max_length=1000)
    description: str = Field(min_length=1, max_length=10000)
    fact_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    certainty: Literal[
        "confirmed",
        "approximate",
        "disputed",
        "unknown",
    ] = "unknown"

    @model_validator(mode="after")
    def validate_event_dates(self) -> TimelineEvent:
        """Ensure the event's start does not follow its end."""
        if (
            self.date_start is not None
            and self.date_end is not None
            and self.date_start > self.date_end
        ):
            raise ValueError("date_start must not be later than date_end.")
        return self


class LocationRecord(StrictModel):
    """A geographic location relevant to the documentary."""

    location_id: str = Field(pattern=r"^LOC_[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=5000)
    country: str | None = Field(default=None, max_length=200)
    region: str | None = Field(default=None, max_length=200)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    source_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)


class HistoricalContext(StrictModel):
    """Historical or cultural context associated with a research topic."""

    context_id: str = Field(pattern=r"^CTX_[A-Za-z0-9_-]+$")
    title: str = Field(min_length=1, max_length=1000)
    description: str = Field(min_length=1, max_length=10000)
    period: str | None = Field(default=None, max_length=500)
    source_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)


class ResearchOutput(StrictModel):
    """Validated output produced by the research stage."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    job_id: UUID
    topic: str = Field(min_length=1, max_length=2000)
    language: str = Field(min_length=2, max_length=35)
    retrieved_at: datetime
    sources: list[ResearchSource] = Field(default_factory=list)
    facts: list[ResearchFact] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    locations: list[LocationRecord] = Field(default_factory=list)
    historical_context: list[HistoricalContext] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    research_status: Literal[
        "complete",
        "partial",
        "insufficient_evidence",
        "failed",
    ]

    @model_validator(mode="after")
    def validate_research_references(self) -> ResearchOutput:
        """Ensure all referenced sources and facts exist in this artifact."""
        source_ids = [source.source_id for source in self.sources]
        fact_ids = [fact.fact_id for fact in self.facts]
        event_ids = [event.event_id for event in self.timeline]
        location_ids = [location.location_id for location in self.locations]
        context_ids = [context.context_id for context in self.historical_context]

        self._require_unique(source_ids, "source_id")
        self._require_unique(fact_ids, "fact_id")
        self._require_unique(event_ids, "event_id")
        self._require_unique(location_ids, "location_id")
        self._require_unique(context_ids, "context_id")

        known_sources = set(source_ids)
        known_facts = set(fact_ids)

        for fact in self.facts:
            missing = set(fact.source_ids) - known_sources
            if missing:
                raise ValueError(
                    f"Fact {fact.fact_id} references unknown sources: "
                    f"{sorted(missing)}"
                )

        for event in self.timeline:
            missing_sources = set(event.source_ids) - known_sources
            missing_facts = set(event.fact_ids) - known_facts

            if missing_sources:
                raise ValueError(
                    f"Event {event.event_id} references unknown sources: "
                    f"{sorted(missing_sources)}"
                )
            if missing_facts:
                raise ValueError(
                    f"Event {event.event_id} references unknown facts: "
                    f"{sorted(missing_facts)}"
                )

        for location in self.locations:
            missing_sources = set(location.source_ids) - known_sources
            missing_facts = set(location.fact_ids) - known_facts

            if missing_sources or missing_facts:
                raise ValueError(
                    f"Location {location.location_id} contains unresolved "
                    f"references: sources={sorted(missing_sources)}, "
                    f"facts={sorted(missing_facts)}"
                )

        for context in self.historical_context:
            missing_sources = set(context.source_ids) - known_sources
            missing_facts = set(context.fact_ids) - known_facts

            if missing_sources or missing_facts:
                raise ValueError(
                    f"Context {context.context_id} contains unresolved "
                    f"references: sources={sorted(missing_sources)}, "
                    f"facts={sorted(missing_facts)}"
                )

        return self

    @staticmethod
    def _require_unique(values: list[str], field_name: str) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"{field_name} values must be unique.")


class CameraShot(StrictModel):
    """Camera and composition instructions for a scene."""

    shot_type: Literal[
        "establishing",
        "extreme_wide",
        "wide",
        "medium",
        "medium_close_up",
        "close_up",
        "extreme_close_up",
        "over_the_shoulder",
        "point_of_view",
        "aerial",
        "macro",
        "cutaway",
        "insert",
        "montage",
        "screen_capture",
        "archival",
        "custom",
    ]
    camera_movement: Literal[
        "static",
        "pan_left",
        "pan_right",
        "tilt_up",
        "tilt_down",
        "dolly_in",
        "dolly_out",
        "truck_left",
        "truck_right",
        "tracking",
        "orbit",
        "crane_up",
        "crane_down",
        "handheld",
        "steadicam",
        "zoom_in",
        "zoom_out",
        "drone_flyover",
        "custom",
    ]
    framing: Literal[
        "centered",
        "rule_of_thirds",
        "symmetrical",
        "leading_lines",
        "negative_space",
        "layered_depth",
        "custom",
    ]
    angle: Literal[
        "eye_level",
        "high_angle",
        "low_angle",
        "birds_eye",
        "worms_eye",
        "dutch_angle",
        "overhead",
        "ground_level",
        "custom",
    ]
    lens_description: str | None = Field(default=None, max_length=500)
    shot_notes: str | None = Field(default=None, max_length=2000)


class NarrationSegment(StrictModel):
    """Narration text assigned to a screenplay scene."""

    segment_id: str = Field(pattern=r"^NAR_[A-Za-z0-9_-]+$")
    text: str = Field(min_length=1, max_length=10000)
    estimated_duration_seconds: float = Field(gt=0, le=600)
    delivery_style: Literal[
        "neutral",
        "conversational",
        "documentary",
        "dramatic",
        "whispered",
        "urgent",
        "reflective",
        "emotional",
        "authoritative",
        "custom",
    ]
    emphasis_words: list[str] = Field(default_factory=list)
    pause_after_ms: int = Field(default=0, ge=0, le=10000)


class DialogueLine(StrictModel):
    """A line of dialogue assigned to a speaker."""

    speaker: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=5000)
    delivery_style: Literal[
        "neutral",
        "calm",
        "excited",
        "angry",
        "sad",
        "whispered",
        "urgent",
        "formal",
        "casual",
        "custom",
    ] = "neutral"
    estimated_duration_seconds: float | None = Field(default=None, gt=0, le=300)


class SceneBlueprint(StrictModel):
    """Complete creative, visual, and audio direction for one scene."""

    scene_id: str = Field(pattern=r"^SCN_[A-Za-z0-9_-]+$")
    scene_number: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=1, max_length=2000)
    duration_seconds: float = Field(gt=0, le=1800)
    narration: list[NarrationSegment] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    visual_description: str = Field(min_length=1, max_length=5000)
    camera_shot: CameraShot
    visual_prompt: str = Field(min_length=1, max_length=10000)
    negative_visual_prompt: str | None = Field(default=None, max_length=5000)
    music_intensity: float = Field(ge=0.0, le=1.0)
    music_mood: Literal[
        "none",
        "neutral",
        "calm",
        "hopeful",
        "inspiring",
        "tense",
        "suspenseful",
        "sad",
        "melancholic",
        "triumphant",
        "mysterious",
        "epic",
        "reflective",
        "urgent",
        "custom",
    ] = "neutral"
    sfx_tags: list[str] = Field(default_factory=list)
    ambience_tags: list[str] = Field(default_factory=list)
    transition: Literal[
        "cut",
        "fade_in",
        "fade_out",
        "cross_dissolve",
        "match_cut",
        "whip_pan",
        "dip_to_black",
        "dip_to_white",
        "audio_bridge",
        "montage_transition",
        "custom",
    ]
    transition_notes: str | None = Field(default=None, max_length=2000)
    research_fact_ids: list[str] = Field(default_factory=list)
    continuity_requirements: list[str] = Field(default_factory=list)
    scene_status: Literal[
        "draft",
        "approved",
        "needs_review",
        "blocked",
    ] = "draft"

    @field_validator("sfx_tags", "ambience_tags")
    @classmethod
    def validate_audio_tags(cls, values: list[str]) -> list[str]:
        """Reject duplicate or empty sound tags."""
        if len(values) != len(set(values)):
            raise ValueError("Audio tags must not contain duplicates.")
        for value in values:
            if not value or len(value) > 80:
                raise ValueError("Each audio tag must contain 1 to 80 characters.")
        return values


class ScreenplayBlueprint(StrictModel):
    """Complete screenplay artifact produced by Slot 2."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    job_id: UUID
    project_title: str = Field(min_length=1, max_length=300)
    language: str = Field(min_length=2, max_length=35)
    target_duration_seconds: int = Field(ge=60, le=7200)
    format: Literal[
        "documentary",
        "short_documentary",
        "reel",
        "short_film",
        "educational",
        "historical_narrative",
        "biographical",
        "explainer",
    ]
    genre: str = Field(min_length=1, max_length=120)
    narrative_style: Literal[
        "investigative",
        "cinematic",
        "educational",
        "journalistic",
        "dramatic",
        "conversational",
        "poetic",
        "neutral",
    ]
    research_reference: dict[str, Any]
    scenes: list[SceneBlueprint] = Field(min_length=1)
    total_estimated_duration_seconds: float = Field(gt=0)
    continuity_notes: list[str] = Field(default_factory=list)
    screenplay_status: Literal[
        "complete",
        "partial",
        "needs_review",
        "failed",
    ]
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("research_reference")
    @classmethod
    def validate_research_reference(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Validate the screenplay's research reference structure."""
        allowed_keys = {
            "research_schema_version",
            "research_status",
            "source_ids",
            "fact_ids",
        }
        unknown = set(value) - allowed_keys
        missing = allowed_keys - set(value)

        if unknown:
            raise ValueError(
                f"Unknown research_reference fields: {sorted(unknown)}"
            )
        if missing:
            raise ValueError(
                f"Missing research_reference fields: {sorted(missing)}"
            )

        if value["research_schema_version"] != "1.0.0":
            raise ValueError("Unsupported research schema version.")

        if value["research_status"] not in {
            "complete",
            "partial",
            "insufficient_evidence",
        }:
            raise ValueError("Invalid research_status in research_reference.")

        for key, prefix in (("source_ids", "SRC_"), ("fact_ids", "FACT_")):
            identifiers = value[key]
            if not isinstance(identifiers, list):
                raise ValueError(f"{key} must be a list.")
            if any(
                not isinstance(identifier, str)
                or not identifier.startswith(prefix)
                for identifier in identifiers
            ):
                raise ValueError(f"All {key} entries must start with {prefix!r}.")
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{key} must not contain duplicates.")

        return value

    @model_validator(mode="after")
    def validate_scenes_and_duration(self) -> ScreenplayBlueprint:
        """Check unique scene identifiers, scene order, and duration consistency."""
        scene_ids = [scene.scene_id for scene in self.scenes]
        scene_numbers = [scene.scene_number for scene in self.scenes]

        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("scene_id values must be unique.")

        if len(scene_numbers) != len(set(scene_numbers)):
            raise ValueError("scene_number values must be unique.")

        if scene_numbers != sorted(scene_numbers):
            raise ValueError("Scenes must be ordered by scene_number.")

        narration_ids = [
            segment.segment_id
            for scene in self.scenes
            for segment in scene.narration
        ]
        if len(narration_ids) != len(set(narration_ids)):
            raise ValueError("Narration segment identifiers must be unique.")

        expected_duration = sum(scene.duration_seconds for scene in self.scenes)
        tolerance = max(1.0, expected_duration * 0.02)

        if abs(expected_duration - self.total_estimated_duration_seconds) > tolerance:
            raise ValueError(
                "total_estimated_duration_seconds does not match the sum of "
                "scene durations within the allowed tolerance."
            )

        return self


class WordTimestamp(StrictModel):
    """Timing information for an individual word in narration audio."""

    word_index: int = Field(ge=0)
    word: str = Field(min_length=1, max_length=300)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_word_interval(self) -> WordTimestamp:
        """Ensure each word has a positive duration."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Word end_seconds must be greater than start_seconds.")
        return self


class VolumeEnvelopePoint(StrictModel):
    """A timestamped gain value for an audio volume envelope."""

    time_seconds: float = Field(ge=0)
    gain: float = Field(ge=0.0, le=2.0)


class NarrationAsset(StrictModel):
    """Generated narration asset and its word-level alignment."""

    asset_id: str = Field(pattern=r"^NAR_AUD_[A-Za-z0-9_-]+$")
    scene_id: str = Field(pattern=r"^SCN_[A-Za-z0-9_-]+$")
    segment_id: str = Field(pattern=r"^NAR_[A-Za-z0-9_-]+$")
    text: str = Field(min_length=1, max_length=20000)
    audio_path: str | None = Field(default=None, max_length=2000)
    audio_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    voice_profile: dict[str, Any]
    word_timestamps: list[WordTimestamp] = Field(default_factory=list)
    generation_status: Literal[
        "pending",
        "generating",
        "complete",
        "partial",
        "failed",
    ]

    @field_validator("voice_profile")
    @classmethod
    def validate_voice_profile(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Validate required voice profile fields and reject unknown properties."""
        required = {
            "provider",
            "voice_id",
            "language",
            "speaking_rate",
            "pitch",
        }
        allowed = required | {
            "gender_presentation",
            "style",
            "emotion",
        }

        if set(value) - allowed:
            raise ValueError(
                f"Unknown voice_profile fields: {sorted(set(value) - allowed)}"
            )
        if required - set(value):
            raise ValueError(
                f"Missing voice_profile fields: {sorted(required - set(value))}"
            )

        if not isinstance(value["provider"], str) or not value["provider"].strip():
            raise ValueError("voice_profile.provider must be a non-empty string.")
        if not isinstance(value["voice_id"], str) or not value["voice_id"].strip():
            raise ValueError("voice_profile.voice_id must be a non-empty string.")
        if not isinstance(value["language"], str) or len(value["language"]) < 2:
            raise ValueError("voice_profile.language must contain at least two characters.")

        rate = value["speaking_rate"]
        pitch = value["pitch"]

        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0.5 <= rate <= 2.0:
            raise ValueError("voice_profile.speaking_rate must be between 0.5 and 2.0.")
        if isinstance(pitch, bool) or not isinstance(pitch, (int, float)) or not -20 <= pitch <= 20:
            raise ValueError("voice_profile.pitch must be between -20 and 20.")

        return value

    @model_validator(mode="after")
    def validate_narration_timing(self) -> NarrationAsset:
        """Ensure asset and word timing intervals are internally consistent."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Narration end_seconds must exceed start_seconds.")

        if abs((self.end_seconds - self.start_seconds) - self.duration_seconds) > 0.1:
            raise ValueError("Narration duration_seconds does not match its time interval.")

        previous_start = -1.0
        previous_index = -1

        for timestamp in self.word_timestamps:
            if timestamp.start_seconds < previous_start:
                raise ValueError("Word timestamps must be ordered by start_seconds.")
            if timestamp.word_index <= previous_index:
                raise ValueError("word_index values must be strictly increasing.")
            if timestamp.start_seconds < self.start_seconds:
                raise ValueError("Word timestamp starts before its narration asset.")
            if timestamp.end_seconds > self.end_seconds:
                raise ValueError("Word timestamp ends after its narration asset.")

            previous_start = timestamp.start_seconds
            previous_index = timestamp.word_index

        return self


class AudioSegment(StrictModel):
    """A narration, music, effect, ambience, or silence segment on the audio timeline."""

    segment_id: str = Field(pattern=r"^AUD_[A-Za-z0-9_-]+$")
    segment_type: Literal[
        "narration",
        "dialogue",
        "background_music",
        "sound_effect",
        "ambience",
        "room_tone",
        "silence",
        "transition",
        "other",
    ]
    scene_id: str | None = Field(default=None, pattern=r"^SCN_[A-Za-z0-9_-]+$")
    asset_id: str | None = Field(default=None, min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    audio_path: str | None = Field(default=None, max_length=2000)
    volume: float = Field(ge=0.0, le=2.0)
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    fade_in_seconds: float = Field(default=0.0, ge=0.0)
    fade_out_seconds: float = Field(default=0.0, ge=0.0)
    priority: int = Field(default=50, ge=0, le=100)
    ducking_group: str | None = Field(default=None, max_length=100)
    loop: bool = False

    @model_validator(mode="after")
    def validate_segment_timing(self) -> AudioSegment:
        """Validate segment duration and fade boundaries."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Audio segment end must exceed its start.")

        actual_duration = self.end_seconds - self.start_seconds
        if abs(actual_duration - self.duration_seconds) > 0.1:
            raise ValueError("Audio segment duration does not match its time interval.")

        if self.fade_in_seconds + self.fade_out_seconds > self.duration_seconds:
            raise ValueError("Combined fade durations cannot exceed segment duration.")

        return self


class VolumeEnvelope(StrictModel):
    """Gain automation for a target audio segment."""

    envelope_id: str = Field(pattern=r"^ENV_[A-Za-z0-9_-]+$")
    target_segment_id: str = Field(pattern=r"^AUD_[A-Za-z0-9_-]+$")
    points: list[VolumeEnvelopePoint] = Field(min_length=2)
    interpolation: Literal["linear", "smooth", "step", "cubic"]
    purpose: Literal[
        "fade",
        "ducking",
        "emphasis",
        "normalization",
        "manual_mix",
        "other",
    ] = "manual_mix"

    @model_validator(mode="after")
    def validate_envelope_points(self) -> VolumeEnvelope:
        """Ensure envelope points are strictly ordered by timestamp."""
        times = [point.time_seconds for point in self.points]

        if times != sorted(times):
            raise ValueError("Volume envelope points must be sorted by time_seconds.")
        if len(times) != len(set(times)):
            raise ValueError("Volume envelope point timestamps must be unique.")

        return self


class SilenceInterval(StrictModel):
    """An intentional silence or pause on the master timeline."""

    silence_id: str = Field(pattern=r"^SIL_[A-Za-z0-9_-]+$")
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    silence_type: Literal[
        "absolute_silence",
        "narrative_pause",
        "dramatic_pause",
        "breath_pause",
        "transition_gap",
        "intentional_room_tone",
        "other",
    ]
    reason: str | None = Field(default=None, max_length=2000)
    preserve_ambience: bool = False

    @model_validator(mode="after")
    def validate_silence_timing(self) -> SilenceInterval:
        """Validate the duration of the silence interval."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Silence end_seconds must exceed start_seconds.")

        if abs((self.end_seconds - self.start_seconds) - self.duration_seconds) > 0.1:
            raise ValueError("Silence duration does not match its time interval.")

        return self


class SceneAnchor(StrictModel):
    """An audio timeline anchor corresponding to a screenplay scene."""

    scene_id: str = Field(pattern=r"^SCN_[A-Za-z0-9_-]+$")
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    anchor_type: Literal[
        "scene_start",
        "scene_end",
        "narration_start",
        "narration_end",
        "music_cue",
        "sound_effect_cue",
        "transition_point",
        "visual_sync_point",
        "other",
    ]
    label: str | None = Field(default=None, max_length=500)
    related_segment_ids: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_anchor_timing(self) -> SceneAnchor:
        """Ensure the anchor represents a positive time interval."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Scene anchor end_seconds must exceed start_seconds.")
        return self


class AudioTimeline(StrictModel):
    """Complete audio timeline produced by Slot 4."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    job_id: UUID
    project_id: str = Field(min_length=1, max_length=200)
    language: str = Field(min_length=2, max_length=35)
    sample_rate_hz: Literal[22050, 24000, 32000, 44100, 48000, 96000]
    channels: Literal[1, 2]
    total_duration_seconds: float = Field(gt=0, le=7200)
    audio_format: Literal["wav", "flac", "mp3", "ogg", "aac", "pcm"] = "wav"
    master_audio_path: str | None = Field(default=None, max_length=2000)
    narration_assets: list[NarrationAsset] = Field(default_factory=list)
    audio_segments: list[AudioSegment] = Field(default_factory=list)
    volume_envelopes: list[VolumeEnvelope] = Field(default_factory=list)
    silence_intervals: list[SilenceInterval] = Field(default_factory=list)
    scene_anchors: list[SceneAnchor] = Field(default_factory=list)
    mixing_settings: dict[str, Any] | None = None
    loudness_report: dict[str, Any] | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    audio_status: Literal[
        "pending",
        "generating",
        "partial",
        "complete",
        "needs_review",
        "failed",
    ]

    @field_validator("mixing_settings")
    @classmethod
    def validate_mixing_settings(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Validate the supported master mixing configuration."""
        if value is None:
            return None

        required = {
            "target_loudness_lufs",
            "true_peak_limit_dbtp",
            "normalize_output",
            "stereo_mode",
        }
        allowed = required | {
            "limiter_enabled",
            "compressor_enabled",
            "noise_reduction_enabled",
            "crossfade_enabled",
        }

        if set(value) - allowed:
            raise ValueError(
                f"Unknown mixing_settings fields: {sorted(set(value) - allowed)}"
            )
        if required - set(value):
            raise ValueError(
                f"Missing mixing_settings fields: {sorted(required - set(value))}"
            )

        loudness = value["target_loudness_lufs"]
        true_peak = value["true_peak_limit_dbtp"]

        if isinstance(loudness, bool) or not isinstance(loudness, (int, float)):
            raise ValueError("target_loudness_lufs must be numeric.")
        if not -30 <= loudness <= -5:
            raise ValueError("target_loudness_lufs must be between -30 and -5 LUFS.")

        if isinstance(true_peak, bool) or not isinstance(true_peak, (int, float)):
            raise ValueError("true_peak_limit_dbtp must be numeric.")
        if not -9 <= true_peak <= 0:
            raise ValueError("true_peak_limit_dbtp must be between -9 and 0 dBTP.")

        if not isinstance(value["normalize_output"], bool):
            raise ValueError("normalize_output must be a boolean.")

        if value["stereo_mode"] not in {"mono", "stereo", "preserve_source"}:
            raise ValueError("Invalid stereo_mode.")

        for key in allowed - required:
            if key in value and not isinstance(value[key], bool):
                raise ValueError(f"{key} must be a boolean.")

        return value

    @field_validator("loudness_report")
    @classmethod
    def validate_loudness_report(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Validate loudness measurement metadata."""
        if value is None:
            return None

        required = {
            "integrated_loudness_lufs",
            "true_peak_dbtp",
            "measured_duration_seconds",
            "measurement_status",
        }
        allowed = required | {"measurement_notes"}

        if set(value) - allowed:
            raise ValueError(
                f"Unknown loudness_report fields: {sorted(set(value) - allowed)}"
            )
        if required - set(value):
            raise ValueError(
                f"Missing loudness_report fields: {sorted(required - set(value))}"
            )

        for key in ("integrated_loudness_lufs", "true_peak_dbtp"):
            number = value[key]
            if number is not None and (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not -100 <= number <= 20
            ):
                raise ValueError(f"{key} must be null or a number between -100 and 20.")

        duration = value["measured_duration_seconds"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration < 0
        ):
            raise ValueError("measured_duration_seconds must be non-negative.")

        if value["measurement_status"] not in {
            "not_measured",
            "passed",
            "warning",
            "failed",
        }:
            raise ValueError("Invalid loudness measurement_status.")

        return value

    @model_validator(mode="after")
    def validate_timeline_references(self) -> AudioTimeline:
        """Check unique identifiers and references across audio timeline objects."""
        self._require_unique(
            [asset.asset_id for asset in self.narration_assets],
            "narration asset_id",
        )
        self._require_unique(
            [segment.segment_id for segment in self.audio_segments],
            "audio segment_id",
        )
        self._require_unique(
            [envelope.envelope_id for envelope in self.volume_envelopes],
            "volume envelope_id",
        )
        self._require_unique(
            [interval.silence_id for interval in self.silence_intervals],
            "silence_id",
        )

        asset_ids = {asset.asset_id for asset in self.narration_assets}
        segment_ids = {segment.segment_id for segment in self.audio_segments}

        for asset in self.narration_assets:
            if asset.end_seconds > self.total_duration_seconds + 0.1:
                raise ValueError(
                    f"Narration asset {asset.asset_id} exceeds total timeline duration."
                )

        for segment in self.audio_segments:
            if segment.end_seconds > self.total_duration_seconds + 0.1:
                raise ValueError(
                    f"Audio segment {segment.segment_id} exceeds total timeline duration."
                )

            if segment.asset_id is not None and segment.asset_id not in asset_ids:
                raise ValueError(
                    f"Audio segment {segment.segment_id} references unknown asset "
                    f"{segment.asset_id}."
                )

        for envelope in self.volume_envelopes:
            if envelope.target_segment_id not in segment_ids:
                raise ValueError(
                    f"Envelope {envelope.envelope_id} references unknown segment "
                    f"{envelope.target_segment_id}."
                )
            if envelope.points[-1].time_seconds > self.total_duration_seconds + 0.1:
                raise ValueError(
                    f"Envelope {envelope.envelope_id} exceeds total timeline duration."
                )

        for interval in self.silence_intervals:
            if interval.end_seconds > self.total_duration_seconds + 0.1:
                raise ValueError(
                    f"Silence interval {interval.silence_id} exceeds total timeline duration."
                )

        for anchor in self.scene_anchors:
            if anchor.end_seconds > self.total_duration_seconds + 0.1:
                raise ValueError(
                    f"Scene anchor {anchor.scene_id} exceeds total timeline duration."
                )

            missing_segments = set(anchor.related_segment_ids) - segment_ids
            if missing_segments:
                raise ValueError(
                    f"Scene anchor {anchor.scene_id} references unknown audio segments: "
                    f"{sorted(missing_segments)}"
                )

        return self

    @staticmethod
    def _require_unique(values: list[str], field_name: str) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"{field_name} values must be unique.")


class StageCheckpoint(StrictModel):
    """Integrity-checked checkpoint metadata for one pipeline stage."""

    checkpoint_id: str = Field(pattern=r"^CHK_[A-Za-z0-9_-]+$")
    stage_id: Literal[
        "initialization",
        "slot1_research",
        "slot2_screenplay",
        "slot4_audio",
        "slot3_visuals",
        "slot5_editor",
        "final_validation",
    ]
    revision: int = Field(ge=1)
    created_at: datetime
    checkpoint_path: str | None = Field(default=None, max_length=2000)
    checkpoint_hash: str = Field(pattern=SHA256_PATTERN)
    previous_checkpoint_hash: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    state_snapshot_hash: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    integrity_verified: bool = False
    recovery_compatible: bool = True

    @field_validator("artifacts")
    @classmethod
    def validate_checkpoint_artifacts(
        cls,
        values: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reject unknown artifact reference fields."""
        allowed = {
            "artifact_id",
            "artifact_type",
            "path",
            "sha256",
            "size_bytes",
            "media_type",
            "created_at",
        }
        required = {"artifact_id", "artifact_type", "path", "sha256"}

        for index, artifact in enumerate(values):
            unknown = set(artifact) - allowed
            missing = required - set(artifact)

            if unknown:
                raise ValueError(
                    f"Checkpoint artifact {index} has unknown fields: {sorted(unknown)}"
                )
            if missing:
                raise ValueError(
                    f"Checkpoint artifact {index} is missing fields: {sorted(missing)}"
                )

            if not isinstance(artifact["artifact_id"], str) or not artifact["artifact_id"]:
                raise ValueError(f"Checkpoint artifact {index} requires artifact_id.")
            if not isinstance(artifact["path"], str) or not artifact["path"]:
                raise ValueError(f"Checkpoint artifact {index} requires path.")
            if not isinstance(artifact["sha256"], str) or not __import__("re").fullmatch(
                SHA256_PATTERN, artifact["sha256"]
            ):
                raise ValueError(f"Checkpoint artifact {index} has an invalid SHA-256 hash.")

        return values


class ProjectError(StrictModel):
    """Structured error record for pipeline failure and recovery diagnostics."""

    error_id: str = Field(pattern=r"^ERR_[A-Za-z0-9_-]+$")
    timestamp: datetime
    stage_id: Literal[
        "initialization",
        "slot1_research",
        "slot2_screenplay",
        "slot4_audio",
        "slot3_visuals",
        "slot5_editor",
        "final_validation",
    ] | None
    error_type: Literal[
        "validation_error",
        "configuration_error",
        "dependency_error",
        "network_error",
        "provider_error",
        "resource_exhaustion",
        "timeout",
        "process_crash",
        "checkpoint_corruption",
        "artifact_integrity_error",
        "permission_error",
        "cancelled",
        "unexpected_error",
        "other",
    ]
    message: str = Field(min_length=1, max_length=8000)
    severity: Literal["info", "warning", "error", "critical"]
    recoverable: bool
    retryable: bool = False
    retry_attempt: int = Field(default=0, ge=0)
    exception_class: str | None = Field(default=None, max_length=500)
    stack_trace: str | None = Field(default=None, max_length=30000)
    component: str | None = Field(default=None, max_length=300)
    process_id: int | None = Field(default=None, ge=0)
    exit_code: int | None = None
    recovery_action: Literal[
        "retry_stage",
        "restore_checkpoint",
        "restart_worker",
        "skip_stage",
        "abort_job",
        "manual_intervention",
        "none",
    ] | None = None
    context: dict[str, str] = Field(default_factory=dict)
    resolved: bool = False
    resolved_at: datetime | None = None

    @field_validator("context")
    @classmethod
    def validate_error_context(cls, value: dict[str, str]) -> dict[str, str]:
        """Limit error context keys and values to bounded strings."""
        for key, item in value.items():
            if not key or len(key) > 500:
                raise ValueError("Error context keys must contain 1 to 500 characters.")
            if not isinstance(item, str) or len(item) > 8000:
                raise ValueError("Error context values must be strings of at most 8000 characters.")
        return value


class StageState(StrictModel):
    """Execution state and checkpoint references for one pipeline stage."""

    stage_id: Literal[
        "initialization",
        "slot1_research",
        "slot2_screenplay",
        "slot4_audio",
        "slot3_visuals",
        "slot5_editor",
        "final_validation",
    ]
    stage_name: str = Field(min_length=1, max_length=200)
    status: Literal[
        "pending",
        "queued",
        "running",
        "paused",
        "completed",
        "failed",
        "skipped",
        "retrying",
        "cancelled",
        "needs_review",
    ]
    attempt: int = Field(ge=0)
    max_attempts: int = Field(default=3, ge=0)
    checkpoint_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    checkpoint_path: str | None = Field(default=None, max_length=2000)
    input_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    output_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    input_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    output_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    duration_seconds: float | None = Field(default=None, ge=0)
    worker_id: str | None = Field(default=None, max_length=200)
    error_ids: list[str] = Field(default_factory=list)
    retryable: bool = False
    notes: list[str] = Field(default_factory=list)

    @field_validator("input_artifacts", "output_artifacts")
    @classmethod
    def validate_stage_artifacts(
        cls,
        values: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate stage artifact references using the checkpoint artifact contract."""
        return StageCheckpoint.validate_checkpoint_artifacts(values)

    @field_validator("error_ids")
    @classmethod
    def validate_error_ids(cls, values: list[str]) -> list[str]:
        """Require unique structured error identifiers."""
        if len(values) != len(set(values)):
            raise ValueError("error_ids must not contain duplicates.")
        if any(not value.startswith("ERR_") for value in values):
            raise ValueError("Every error ID must start with 'ERR_'.")
        return values

    @model_validator(mode="after")
    def validate_stage_state(self) -> StageState:
        """Ensure stage attempt counts and timestamps are internally consistent."""
        if self.attempt > self.max_attempts and self.max_attempts > 0:
            raise ValueError("attempt cannot exceed max_attempts.")

        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("completed_at must not precede started_at.")

        if self.status == "completed" and self.completed_at is None:
            raise ValueError("A completed stage must have completed_at set.")

        return self


class ProjectState(StrictModel):
    """Persistent state for crash-safe pipeline execution and checkpoint recovery."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    job_id: UUID
    project_id: str = Field(min_length=1, max_length=200)
    project_title: str | None = Field(default=None, max_length=300)
    status: Literal[
        "created",
        "queued",
        "running",
        "paused",
        "recovering",
        "completed",
        "failed",
        "cancelled",
        "needs_review",
    ]
    revision: int = Field(ge=1)
    current_stage: Literal[
        "initialization",
        "slot1_research",
        "slot2_screenplay",
        "slot4_audio",
        "slot3_visuals",
        "slot5_editor",
        "final_validation",
        "completed",
        "none",
    ]
    stages: list[StageState]
    checkpoint: StageCheckpoint | None = None
    errors: list[ProjectError] = Field(default_factory=list)
    recovery_attempts: int = Field(default=0, ge=0)
    max_recovery_attempts: int = Field(default=3, ge=0)
    last_recovery_at: datetime | None = None
    last_error_id: str | None = Field(default=None, pattern=r"^ERR_[A-Za-z0-9_-]+$")
    cancel_requested: bool = False
    pause_requested: bool = False
    created_at: datetime
    started_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_project_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Validate project metadata keys and value types."""
        allowed = {
            "requested_format",
            "target_duration_seconds",
            "language",
            "test_mode",
            "configuration_hash",
            "engine_version",
            "schema_versions",
            "tags",
        }

        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown project metadata fields: {sorted(unknown)}")

        if "target_duration_seconds" in value:
            duration = value["target_duration_seconds"]
            if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 7200:
                raise ValueError("metadata.target_duration_seconds must be an integer from 1 to 7200.")

        if "test_mode" in value and not isinstance(value["test_mode"], bool):
            raise ValueError("metadata.test_mode must be a boolean.")

        if "configuration_hash" in value:
            import re

            if not isinstance(value["configuration_hash"], str) or not re.fullmatch(
                SHA256_PATTERN, value["configuration_hash"]
            ):
                raise ValueError("metadata.configuration_hash must be a SHA-256 hash.")

        if "schema_versions" in value:
            versions = value["schema_versions"]
            allowed_versions = {
                "research_output",
                "screenplay_blueprint",
                "audio_timeline",
                "project_state",
            }
            if not isinstance(versions, dict):
                raise ValueError("metadata.schema_versions must be an object.")
            if set(versions) - allowed_versions:
                raise ValueError(
                    f"Unknown schema version keys: {sorted(set(versions) - allowed_versions)}"
                )
            if any(not isinstance(version, str) for version in versions.values()):
                raise ValueError("Schema version values must be strings.")

        if "tags" in value:
            tags = value["tags"]
            if not isinstance(tags, list) or any(
                not isinstance(tag, str) or not tag.strip() for tag in tags
            ):
                raise ValueError("metadata.tags must be a list of non-empty strings.")

        return value

    @model_validator(mode="after")
    def validate_project_state(self) -> ProjectState:
        """Ensure stage identities, checkpoint references, and errors are consistent."""
        stage_ids = [stage.stage_id for stage in self.stages]

        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("Project stages must have unique stage_id values.")

        error_ids = [error.error_id for error in self.errors]
        if len(error_ids) != len(set(error_ids)):
            raise ValueError("Project errors must have unique error_id values.")

        known_error_ids = set(error_ids)

        for stage in self.stages:
            unknown_errors = set(stage.error_ids) - known_error_ids
            if unknown_errors:
                raise ValueError(
                    f"Stage {stage.stage_id} references unknown errors: "
                    f"{sorted(unknown_errors)}"
                )

        if self.last_error_id is not None and self.last_error_id not in known_error_ids:
            raise ValueError("last_error_id must refer to an error in the errors array.")

        if self.current_stage not in {"completed", "none"} and self.current_stage not in stage_ids:
            raise ValueError(
                f"current_stage {self.current_stage!r} is missing from stages."
            )

        if self.checkpoint is not None:
            checkpoint_stage = self.checkpoint.stage_id
            if checkpoint_stage not in stage_ids:
                raise ValueError(
                    f"Checkpoint stage {checkpoint_stage!r} is missing from stages."
                )

        if self.recovery_attempts > self.max_recovery_attempts and self.max_recovery_attempts > 0:
            raise ValueError("recovery_attempts cannot exceed max_recovery_attempts.")

        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("completed_at must not precede started_at.")

        if self.status == "completed" and self.completed_at is None:
            raise ValueError("A completed project must have completed_at set.")

        return self


class ArtifactWriteResult(StrictModel):
    """Metadata returned after atomically writing a validated JSON artifact."""

    path: str
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0)
    written_at: datetime
    schema_name: str
    schema_version: str


def validate_research_output(data: object) -> ResearchOutput:
    """Validate untrusted research artifact data.

    Args:
        data: A dictionary or JSON-compatible object.

    Returns:
        A validated ResearchOutput instance.

    Raises:
        ValidationError: If data violates the research output contract.
    """
    return ResearchOutput.model_validate(data)


def validate_screenplay_blueprint(data: object) -> ScreenplayBlueprint:
    """Validate untrusted screenplay artifact data."""
    return ScreenplayBlueprint.model_validate(data)


def validate_audio_timeline(data: object) -> AudioTimeline:
    """Validate untrusted audio timeline artifact data."""
    return AudioTimeline.model_validate(data)


def validate_project_state(data: object) -> ProjectState:
    """Validate untrusted persistent project state data."""
    return ProjectState.model_validate(data)


def load_json_artifact(
    path: str | Path,
    model: type[SchemaModel],
    *,
    expected_sha256: str | None = None,
) -> tuple[SchemaModel, str]:
    """Load, hash, and validate a JSON artifact.

    The file is hashed as raw bytes before decoding. If expected_sha256 is supplied,
    the hash must match before JSON parsing and model validation proceed.

    Args:
        path: Artifact file path.
        model: Strict Pydantic model class used to validate the decoded JSON.
        expected_sha256: Optional expected SHA-256 digest.

    Returns:
        A tuple containing the validated model and lowercase SHA-256 digest.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        OSError: If the artifact cannot be read.
        ValueError: If the hash mismatches or the JSON is invalid.
        ValidationError: If the decoded JSON violates the model contract.
    """
    artifact_path = Path(path).expanduser().resolve(strict=True)

    if not artifact_path.is_file():
        raise ValueError(f"Artifact path is not a file: {artifact_path}")

    raw_bytes = artifact_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    if expected_sha256 is not None:
        normalized_expected = expected_sha256.lower()
        if normalized_expected != actual_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {artifact_path}: expected "
                f"{normalized_expected}, got {actual_sha256}."
            )

    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Artifact is not valid UTF-8 JSON: {artifact_path}") from exc

    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {artifact_path} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    validated = model.model_validate(payload)
    return validated, actual_sha256


def write_json_artifact(
    path: str | Path,
    artifact: StrictModel | dict[str, Any] | list[Any],
    *,
    indent: int = 2,
    overwrite: bool = True,
    fsync: bool = True,
) -> ArtifactWriteResult:
    """Atomically serialize a JSON artifact and calculate its SHA-256 digest.

    Writes to a temporary file in the destination directory, flushes it, and
    atomically replaces the destination with os.replace(). A failed write leaves
    the previous destination intact and attempts to remove the temporary file.

    Args:
        path: Destination JSON path.
        artifact: Strict Pydantic model or JSON-compatible dictionary/list.
        indent: JSON indentation width.
        overwrite: Whether an existing destination may be replaced.
        fsync: Whether to synchronize the temporary file before replacement.

    Returns:
        ArtifactWriteResult containing the destination, digest, and file size.

    Raises:
        FileExistsError: If the destination exists and overwrite is False.
        OSError: If filesystem operations fail.
        TypeError: If artifact contains values that cannot be serialized to JSON.
        ValueError: If indent is invalid.
    """
    if indent < 0:
        raise ValueError("indent must be a non-negative integer.")

    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not destination.is_file():
        raise ValueError(f"Artifact destination is not a regular file: {destination}")

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Artifact already exists: {destination}")

    if isinstance(artifact, BaseModel):
        payload: Any = artifact.model_dump(mode="json")
        schema_name = type(artifact).__name__
        schema_version_value = getattr(artifact, "schema_version", "unspecified")
        schema_version = str(schema_version_value)
    else:
        payload = artifact
        schema_name = "JSONArtifact"
        schema_version = "unspecified"

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        sort_keys=True,
    )
    encoded = (serialized + "\n").encode("utf-8")
    temporary_path: Path | None = None

    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        temporary_path = Path(temporary_name)

        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(encoded)
            temporary_file.flush()

            if fsync:
                os.fsync(temporary_file.fileno())

        if destination.exists() and not overwrite:
            raise FileExistsError(f"Artifact already exists: {destination}")

        os.replace(temporary_path, destination)
        temporary_path = None

        if fsync:
            directory_descriptor: int | None = None
            try:
                directory_descriptor = os.open(
                    destination.parent,
                    os.O_RDONLY,
                )
                os.fsync(directory_descriptor)
            except OSError:
                # Some platforms do not support fsync on directory descriptors.
                pass
            finally:
                if directory_descriptor is not None:
                    os.close(directory_descriptor)

    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    final_bytes = destination.read_bytes()
    digest = hashlib.sha256(final_bytes).hexdigest()

    return ArtifactWriteResult(
        path=str(destination),
        sha256=digest,
        size_bytes=len(final_bytes),
        written_at=utc_now(),
        schema_name=schema_name,
        schema_version=schema_version,
    )
