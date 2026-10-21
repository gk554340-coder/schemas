# slot1_research.py
"""Research pipeline for the Autonomous AI Cinema and Documentary Studio Engine.

The research stage retrieves sources, extracts evidence, verifies references,
and writes a validated ResearchOutput artifact inside the research workspace.

Test mode is fully offline and deterministic, making it suitable for CI runs.
Production mode uses public web search, Google News RSS, and Open Scripture API
where applicable. Network results are treated as untrusted evidence and are
never assumed to be verified merely because a source was retrieved.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID, uuid4

from pydantic import ValidationError

from config import StudioConfig, load_config, resolve_workspace_path
from schemas import (
    HistoricalContext,
    LocationRecord,
    ResearchFact,
    ResearchOutput,
    ResearchSource,
    TimelineEvent,
    write_json_artifact,
)

LOGGER = logging.getLogger(__name__)

STAGE_ID = "slot1_research"
WORKSPACE_NAME = "research_vault"
WORKSPACE_RELATIVE_PATH = Path("01_research_vault")
DEFAULT_MAX_SOURCES = 12
MAX_ALLOWED_SOURCES = 50
DEFAULT_TIMEOUT_SECONDS = 12.0
MAX_TIMEOUT_SECONDS = 60.0
MAX_RESPONSE_BYTES = 2_000_000
USER_AGENT = (
    "AutonomousAIStudioResearch/1.0 "
    "(public-source research; contact information not configured)"
)

SOURCE_TYPE_BY_PROVIDER: dict[str, str] = {
    "duckduckgo_html": "web",
    "google_news_rss": "news_rss",
    "open_scripture_api": "scripture_api",
}

# Fixed, offline fixtures for CI. These fixtures are intentionally generic and
# clearly marked as synthetic test material rather than historical evidence.
TEST_FIXTURES: dict[str, dict[str, Any]] = {
    "default": {
        "title": "Offline Research Fixture",
        "summary": (
            "This is deterministic synthetic research data used to test the "
            "research pipeline. It is not a factual claim about a real event."
        ),
        "url": "https://example.invalid/offline-research-fixture",
        "publisher": "Autonomous AI Studio Test Fixture",
        "author": "CI Fixture",
        "language": "en",
        "verification_status": "unverified",
    }
}


class ResearchStageError(RuntimeError):
    """Raised when the research stage cannot safely produce its artifact."""


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _normalize_text(value: str | None) -> str:
    """Normalize whitespace and remove invalid control characters."""
    if not value:
        return ""

    cleaned = "".join(
        character
        for character in value
        if character in "\n\r\t" or ord(character) >= 32
    )
    return re.sub(r"\s+", " ", html.unescape(cleaned)).strip()


def _truncate(value: str, maximum: int) -> str:
    """Trim a string to a maximum length without leaving trailing whitespace."""
    value = _normalize_text(value)
    if len(value) <= maximum:
        return value
    return value[: maximum - 1].rstrip() + "…"


def _sha256_text(value: str) -> str:
    """Calculate the lowercase SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_id(provider: str, canonical_url: str, title: str) -> str:
    """Generate a stable source identifier from provider and source identity."""
    identity = f"{provider}\n{canonical_url}\n{title}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"SRC_{digest}"


def _fact_id(source_id: str, statement: str) -> str:
    """Generate a stable fact identifier from its source and statement."""
    digest = hashlib.sha256(
        f"{source_id}\n{statement}".encode("utf-8")
    ).hexdigest()[:20]
    return f"FACT_{digest}"


def _event_id(date_text: str, title: str) -> str:
    """Generate a stable identifier for a timeline event."""
    digest = hashlib.sha256(
        f"{date_text}\n{title}".encode("utf-8")
    ).hexdigest()[:20]
    return f"EVENT_{digest}"


def _resolve_research_workspace(
    output_path: str | Path | None = None,
    config: StudioConfig | None = None,
) -> tuple[StudioConfig, Path, Path]:
    """Resolve and validate the research sandbox and output destination.

    All paths used by this stage must remain inside the configured
    ``workspace/01_research_vault`` directory. Existing symlinks are resolved
    before containment checks to prevent path traversal outside the sandbox.
    """
    active_config = config or load_config()

    workspace_root = Path(active_config.workspace_root).expanduser().resolve()
    configured_research_path = getattr(
        active_config,
        "research_vault_path",
        workspace_root / WORKSPACE_RELATIVE_PATH,
    )
    research_workspace = Path(configured_research_path).expanduser().resolve()

    expected_workspace = (workspace_root / WORKSPACE_RELATIVE_PATH).resolve()

    if research_workspace != expected_workspace:
        raise ResearchStageError(
            "The configured research workspace must be exactly "
            f"{expected_workspace}; received {research_workspace}."
        )

    research_workspace.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        destination = research_workspace / "research_output.json"
    else:
        requested_path = Path(output_path).expanduser()
        if not requested_path.is_absolute():
            destination = research_workspace / requested_path
        else:
            destination = requested_path

        destination = destination.resolve(strict=False)

    try:
        destination.relative_to(research_workspace)
    except ValueError as exc:
        raise ResearchStageError(
            "Research output path escapes workspace/01_research_vault."
        ) from exc

    if destination == research_workspace:
        raise ResearchStageError("Output path must identify a file, not the workspace directory.")

    if destination.suffix.lower() != ".json":
        raise ResearchStageError("Research output must use a .json file extension.")

    return active_config, research_workspace, destination


def _safe_url(value: str | None) -> str | None:
    """Accept only HTTP(S) URLs and strip fragments."""
    if not value:
        return None

    candidate = _normalize_text(value)
    parsed = urllib.parse.urlsplit(candidate)

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None

    if parsed.username is not None or parsed.password is not None:
        return None

    return urllib.parse.urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


def _http_get(
    url: str,
    timeout_seconds: float,
    *,
    accept: str = "text/html,application/rss+xml,application/xml;q=0.9,*/*;q=0.5",
) -> tuple[bytes, str]:
    """Fetch a public HTTP(S) resource with bounded response size and timeout."""
    safe_url = _safe_url(url)
    if safe_url is None:
        raise ValueError("Only valid HTTP(S) URLs may be fetched.")

    request = urllib.request.Request(
        safe_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            final_url = _safe_url(response.geturl())
            if final_url is None:
                raise ResearchStageError("The server redirected to an invalid URL.")

            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    declared_length = 0
                if declared_length > MAX_RESPONSE_BYTES:
                    raise ResearchStageError("Source response exceeds the size limit.")

            payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise ResearchStageError("Source response exceeds the size limit.")

            return payload, final_url

    except urllib.error.HTTPError as exc:
        raise ResearchStageError(
            f"HTTP request failed with status {exc.code} for {safe_url}."
        ) from exc
    except urllib.error.URLError as exc:
        raise ResearchStageError(
            f"Network request failed for {safe_url}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise ResearchStageError(f"Network request timed out for {safe_url}.") from exc


def _decode_response(payload: bytes) -> str:
    """Decode a response using UTF-8 with replacement for malformed bytes."""
    return payload.decode("utf-8", errors="replace")


def _search_duckduckgo(
    topic: str,
    timeout_seconds: float,
    max_results: int,
) -> list[dict[str, Any]]:
    """Retrieve public search result snippets from DuckDuckGo HTML."""
    query = urllib.parse.urlencode(
        {
            "q": topic,
            "kl": "wt-wt",
        }
    )
    search_url = f"https://html.duckduckgo.com/html/?{query}"
    payload, _ = _http_get(
        search_url,
        timeout_seconds,
        accept="text/html,application/xhtml+xml",
    )
    document = _decode_response(payload)

    # DuckDuckGo result markup may change. The parser intentionally extracts
    # only recognized result blocks and ignores malformed or absent elements.
    blocks = re.findall(
        r'<div[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*(?=<div|</div>|$)',
        document,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not blocks:
        blocks = re.split(
            r'<div[^>]*class="[^"]*result__body[^"]*"[^>]*>',
            document,
            flags=re.IGNORECASE,
        )[1:]

    results: list[dict[str, Any]] = []

    for block in blocks:
        if len(results) >= max_results:
            break

        link_match = re.search(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not link_match:
            link_match = re.search(
                r'<a[^>]*href="([^"]+)"[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>',
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
        if not link_match:
            continue

        raw_url = html.unescape(link_match.group(1))
        parsed_url = urllib.parse.urlsplit(raw_url)

        if "duckduckgo.com" in parsed_url.netloc.lower() and parsed_url.path.startswith("/l/"):
            query_parameters = urllib.parse.parse_qs(parsed_url.query)
            raw_url = query_parameters.get("uddg", [raw_url])[0]

        canonical_url = _safe_url(raw_url)
        if canonical_url is None:
            continue

        title = _truncate(
            re.sub(r"<[^>]+>", " ", link_match.group(2)),
            1000,
        )

        snippet_match = re.search(
            r'<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippet = ""
        if snippet_match:
            snippet = _truncate(
                re.sub(r"<[^>]+>", " ", snippet_match.group(1)),
                5000,
            )

        if not title:
            continue

        results.append(
            {
                "provider": "duckduckgo_html",
                "source_type": "web",
                "title": title,
                "url": canonical_url,
                "publisher": urllib.parse.urlsplit(canonical_url).netloc,
                "author": None,
                "published_at": None,
                "excerpt": snippet or title,
                "language": "und",
                "retrieved_at": _utc_now(),
                "verification_status": "unverified",
            }
        )

    return results


def _search_google_news_rss(
    topic: str,
    language: str,
    timeout_seconds: float,
    max_results: int,
) -> list[dict[str, Any]]:
    """Retrieve public Google News RSS entries for a topic."""
    language_code = re.split(r"[-_]", language.strip())[0].lower() or "en"
    query = urllib.parse.urlencode(
        {
            "q": topic,
            "hl": language_code,
            "gl": "US",
            "ceid": f"US:{language_code}",
        }
    )
    rss_url = f"https://news.google.com/rss/search?{query}"
    payload, _ = _http_get(
        rss_url,
        timeout_seconds,
        accept="application/rss+xml,application/xml,text/xml",
    )

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ResearchStageError("Google News returned malformed RSS/XML.") from exc

    results: list[dict[str, Any]] = []

    for item in root.findall(".//item"):
        if len(results) >= max_results:
            break

        title = _truncate(item.findtext("title", default=""), 1000)
        raw_url = item.findtext("link", default="")
        canonical_url = _safe_url(raw_url)
        description = _truncate(item.findtext("description", default=""), 5000)
        pub_date = _normalize_text(item.findtext("pubDate", default=""))
        publisher = _truncate(item.findtext("source", default="Google News"), 500)

        if not title or canonical_url is None:
            continue

        published_at: datetime | None = None
        if pub_date:
            try:
                from email.utils import parsedate_to_datetime

                parsed_date = parsedate_to_datetime(pub_date)
                if parsed_date.tzinfo is None:
                    parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                published_at = parsed_date.astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError):
                published_at = None

        results.append(
            {
                "provider": "google_news_rss",
                "source_type": "news_rss",
                "title": title,
                "url": canonical_url,
                "publisher": publisher or "Google News",
                "author": None,
                "published_at": published_at,
                "excerpt": description or title,
                "language": language_code,
                "retrieved_at": _utc_now(),
                "verification_status": "unverified",
            }
        )

    return results


def _search_open_scripture(
    topic: str,
    language: str,
    timeout_seconds: float,
    max_results: int,
) -> list[dict[str, Any]]:
    """Retrieve relevant public Bible search results from Open Scripture API.

    The API endpoint and response format may evolve. This adapter accepts a
    small set of common JSON response shapes and safely returns no results when
    the service does not provide a recognized structure.
    """
    endpoint = "https://api.scripture.api.bible/v1/bibles"
    payload, final_url = _http_get(
        endpoint,
        timeout_seconds,
        accept="application/json",
    )

    try:
        parsed = json.loads(_decode_response(payload))
    except json.JSONDecodeError as exc:
        raise ResearchStageError("Open Scripture API returned invalid JSON.") from exc

    bible_entries = parsed.get("data", []) if isinstance(parsed, dict) else []
    if not isinstance(bible_entries, list):
        return []

    normalized_topic = topic.casefold()
    candidates: list[dict[str, Any]] = []

    for bible in bible_entries:
        if not isinstance(bible, dict):
            continue

        bible_id = bible.get("id")
        bible_name = _normalize_text(str(bible.get("name", "")))
        bible_language = bible.get("language") or {}
        bible_language_id = (
            bible_language.get("id", "")
            if isinstance(bible_language, dict)
            else str(bible_language)
        )

        if not bible_id or not bible_name:
            continue

        if language and language.split("-")[0].split("_")[0].lower() not in {
            str(bible_language_id).split("-")[0].split("_")[0].lower(),
            "",
            "und",
        }:
            continue

        # This endpoint lists available Bible editions, not topical passages.
        # Include an edition only when its metadata meaningfully matches the
        # requested topic; never invent a passage or verse from edition metadata.
        if normalized_topic not in bible_name.casefold():
            continue

        candidates.append(
            {
                "provider": "open_scripture_api",
                "source_type": "scripture_api",
                "title": bible_name,
                "url": final_url,
                "publisher": "Open Scripture API",
                "author": None,
                "published_at": None,
                "excerpt": _truncate(
                    f"API metadata for Bible edition: {bible_name}. "
                    "This result identifies an edition, not a verified passage.",
                    2000,
                ),
                "language": str(bible_language_id or "und"),
                "retrieved_at": _utc_now(),
                "verification_status": "unverified",
            }
        )

        if len(candidates) >= max_results:
            break

    return candidates


def _build_test_sources(
    topic: str,
    language: str,
    max_sources: int,
) -> list[dict[str, Any]]:
    """Create deterministic offline fixture records for test mode."""
    topic_clean = _truncate(topic, 2000)
    language_clean = _truncate(language, 35) or "en"
    fixture = TEST_FIXTURES["default"]

    source_specs = [
        {
            "provider": "duckduckgo_html",
            "source_type": "web",
            "title": f"Offline test fixture: {topic_clean}",
            "url": fixture["url"],
            "publisher": fixture["publisher"],
            "author": fixture["author"],
            "published_at": None,
            "excerpt": (
                f"Synthetic CI fixture for the requested topic '{topic_clean}'. "
                "This content exists only to validate pipeline contracts."
            ),
            "language": language_clean,
            "verification_status": "unverified",
        },
        {
            "provider": "google_news_rss",
            "source_type": "news_rss",
            "title": f"Offline synthetic news fixture: {topic_clean}",
            "url": "https://example.invalid/offline-news-fixture",
            "publisher": "Autonomous AI Studio Test Fixture",
            "author": None,
            "published_at": None,
            "excerpt": (
                "Synthetic news-shaped fixture for offline testing. "
                "It does not represent a real news article."
            ),
            "language": language_clean,
            "verification_status": "unverified",
        },
        {
            "provider": "open_scripture_api",
            "source_type": "scripture_api",
            "title": f"Offline scripture-provider fixture: {topic_clean}",
            "url": "https://example.invalid/offline-scripture-fixture",
            "publisher": "Autonomous AI Studio Test Fixture",
            "author": None,
            "published_at": None,
            "excerpt": (
                "Synthetic scripture-provider fixture. It is not a quotation "
                "from any scripture or religious text."
            ),
            "language": language_clean,
            "verification_status": "unverified",
        },
    ]

    selected = source_specs[:max_sources]
    for item in selected:
        item["retrieved_at"] = datetime(2026, 1, 1, tzinfo=timezone.utc)

    return selected


def _normalize_source_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a raw provider result into the ResearchSource input contract."""
    title = _truncate(str(record.get("title") or ""), 1000)
    url = _safe_url(str(record.get("url") or ""))
    excerpt = _truncate(str(record.get("excerpt") or title), 20000)

    if not title or url is None:
        return None

    source_type = str(record.get("source_type") or "other")
    if source_type not in {
        "web",
        "news_rss",
        "scripture_api",
        "official_document",
        "book",
        "archive",
        "other",
    }:
        source_type = "other"

    published_at = record.get("published_at")
    if isinstance(published_at, str):
        try:
            published_at = datetime.fromisoformat(
                published_at.replace("Z", "+00:00")
            )
        except ValueError:
            published_at = None

    if isinstance(published_at, datetime) and published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    retrieved_at = record.get("retrieved_at")
    if not isinstance(retrieved_at, datetime):
        retrieved_at = _utc_now()
    elif retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.replace(tzinfo=timezone.utc)

    raw_verification = record.get("verification_status", "unverified")
    verification_status = (
        raw_verification
        if raw_verification in {
            "verified",
            "partially_verified",
            "unverified",
            "disputed",
        }
        else "unverified"
    )

    return {
        "source_id": _source_id(
            str(record.get("provider") or "other"),
            url,
            title,
        ),
        "source_type": source_type,
        "title": title,
        "url": url,
        "author": _truncate(str(record["author"]), 500)
        if record.get("author")
        else None,
        "publisher": _truncate(str(record["publisher"]), 500)
        if record.get("publisher")
        else None,
        "published_at": published_at,
        "retrieved_at": retrieved_at,
        "language": _truncate(str(record.get("language") or "und"), 35),
        "excerpt": excerpt or None,
        "content_sha256": _sha256_text(excerpt) if excerpt else None,
        "reliability_score": 0.5,
        "verification_status": verification_status,
    }


def retrieve_sources(
    topic: str,
    language: str,
    max_sources: int = DEFAULT_MAX_SOURCES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    test_mode: bool = False,
) -> list[ResearchSource]:
    """Retrieve and normalize source records for the requested research topic.

    In test mode, this function performs no network requests and returns
    deterministic synthetic fixtures.

    In production mode, DuckDuckGo HTML and Google News RSS are queried. The
    Open Scripture API is queried only when its available metadata can match
    the topic; its edition-list endpoint is not treated as a passage search API.

    Network providers can fail independently. Failures are logged, and the
    stage continues with any valid sources returned by the other providers.

    Args:
        topic: Research topic.
        language: BCP-47-like language code or language identifier.
        max_sources: Maximum number of source records to return.
        timeout_seconds: Per-request network timeout.
        test_mode: Whether to use deterministic offline fixtures.

    Returns:
        A list of validated ResearchSource models.

    Raises:
        ValueError: If the topic, language, source limit, or timeout is invalid.
        ResearchStageError: If no usable sources can be produced.
    """
    normalized_topic = _normalize_text(topic)
    normalized_language = _normalize_text(language)

    if not normalized_topic:
        raise ValueError("topic must not be empty.")
    if not normalized_language:
        raise ValueError("language must not be empty.")
    if not 1 <= max_sources <= MAX_ALLOWED_SOURCES:
        raise ValueError(
            f"max_sources must be between 1 and {MAX_ALLOWED_SOURCES}."
        )
    if not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout_seconds must be greater than 0 and at most "
            f"{MAX_TIMEOUT_SECONDS}."
        )

    if test_mode:
        raw_records = _build_test_sources(
            normalized_topic,
            normalized_language,
            max_sources,
        )
    else:
        raw_records: list[dict[str, Any]] = []
        providers = (
            (
                "duckduckgo_html",
                lambda: _search_duckduckgo(
                    normalized_topic,
                    timeout_seconds,
                    max_sources,
                ),
            ),
            (
                "google_news_rss",
                lambda: _search_google_news_rss(
                    normalized_topic,
                    normalized_language,
                    timeout_seconds,
                    max_sources,
                ),
            ),
            (
                "open_scripture_api",
                lambda: _search_open_scripture(
                    normalized_topic,
                    normalized_language,
                    timeout_seconds,
                    max_sources,
                ),
            ),
        )

        for provider_name, provider_call in providers:
            if len(raw_records) >= max_sources:
                break

            try:
                results = provider_call()
                raw_records.extend(results)
            except (ResearchStageError, ValueError, OSError, ET.ParseError) as exc:
                LOGGER.warning(
                    "Research provider %s failed: %s",
                    provider_name,
                    exc,
                )
            except Exception:
                LOGGER.exception(
                    "Unexpected error from research provider %s.",
                    provider_name,
                )

            raw_records = raw_records[:max_sources]

    sources: list[ResearchSource] = []
    seen_urls: set[str] = set()
    seen_ids: set[str] = set()

    for record in raw_records:
        normalized = _normalize_source_record(record)
        if normalized is None:
            continue

        source = ResearchSource.model_validate(normalized)

        if source.url in seen_urls or source.source_id in seen_ids:
            continue

        sources.append(source)
        seen_urls.add(source.url or "")
        seen_ids.add(source.source_id)

        if len(sources) >= max_sources:
            break

    if not sources:
        raise ResearchStageError(
            "No usable research sources were retrieved. Check network access "
            "or enable test_mode for an offline CI run."
        )

    return sources


def extract_research_evidence(
    sources: Iterable[ResearchSource | dict[str, Any]],
    topic: str,
    language: str,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Extract conservative research evidence from retrieved source records.

    This function does not infer that a claim is true merely because it appears
    in a search result. It creates source-linked evidence records from the
    available snippets and marks them unverified until independent corroboration
    is found by the verification step.

    The returned object contains facts, timeline candidates, location
    candidates, historical context, and unresolved questions. The output is
    intentionally conservative because search snippets alone are not sufficient
    to establish a detailed historical claim.
    """
    normalized_topic = _normalize_text(topic)
    normalized_language = _normalize_text(language)

    if not normalized_topic:
        raise ValueError("topic must not be empty.")
    if not normalized_language:
        raise ValueError("language must not be empty.")

    validated_sources: list[ResearchSource] = []
    for source in sources:
        validated_sources.append(
            source
            if isinstance(source, ResearchSource)
            else ResearchSource.model_validate(source)
        )

    facts: list[dict[str, Any]] = []
    unresolved_questions: list[str] = []

    for source in validated_sources:
        excerpt = _normalize_text(source.excerpt or "")
        title = _normalize_text(source.title)

        if not excerpt:
            unresolved_questions.append(
                f"Retrieve the full content of source {source.source_id} "
                "before making claims from it."
            )
            continue

        statement = _truncate(
            f"Source {source.source_id} reports: {excerpt}",
            10000,
        )

        facts.append(
            {
                "fact_id": _fact_id(source.source_id, statement),
                "statement": statement,
                "source_ids": [source.source_id],
                "verification_status": "unverified",
                "confidence": 0.25 if not test_mode else 0.1,
                "category": "source_excerpt",
                "notes": (
                    "Synthetic offline fixture; not factual evidence."
                    if test_mode
                    else "Extracted from a search result or public metadata; "
                    "requires independent corroboration."
                ),
            }
        )

        if title and title.casefold() == normalized_topic.casefold():
            unresolved_questions.append(
                f"Confirm whether source {source.source_id} directly addresses "
                "the research topic or only contains a matching title."
            )

    if not facts:
        unresolved_questions.append(
            "No usable source excerpts were available; retrieve source content "
            "before drafting factual narration."
        )

    if test_mode:
        unresolved_questions.append(
            "Test mode is enabled: all fixture evidence is synthetic and must "
            "not be used as factual documentary material."
        )

    return {
        "facts": facts,
        "timeline": [],
        "locations": [],
        "historical_context": [],
        "unresolved_questions": list(dict.fromkeys(unresolved_questions)),
    }


def verify_research_evidence(
    evidence: dict[str, Any],
    sources: Iterable[ResearchSource | dict[str, Any]],
    test_mode: bool = False,
) -> dict[str, Any]:
    """Validate evidence references and conservatively assess corroboration.

    A claim is promoted to ``verified`` only when at least two distinct source
    records support the same normalized statement. Since the extraction step
    retains source-specific snippets, independent corroboration usually requires
    further evidence extraction and is not inferred from similar URLs alone.

    Test fixtures always remain unverified.
    """
    if not isinstance(evidence, dict):
        raise TypeError("evidence must be a dictionary.")

    validated_sources: list[ResearchSource] = [
        source
        if isinstance(source, ResearchSource)
        else ResearchSource.model_validate(source)
        for source in sources
    ]
    sources_by_id = {source.source_id: source for source in validated_sources}

    raw_facts = evidence.get("facts", [])
    if not isinstance(raw_facts, list):
        raise ValueError("evidence.facts must be a list.")

    verified_facts: list[dict[str, Any]] = []
    statement_to_sources: dict[str, set[str]] = {}

    for raw_fact in raw_facts:
        fact = ResearchFact.model_validate(raw_fact)
        missing_sources = set(fact.source_ids) - set(sources_by_id)

        if missing_sources:
            raise ValueError(
                f"Evidence {fact.fact_id} references missing source IDs: "
                f"{sorted(missing_sources)}"
            )

        normalized_statement = re.sub(
            r"\W+",
            " ",
            fact.statement.casefold(),
            flags=re.UNICODE,
        ).strip()

        statement_to_sources.setdefault(normalized_statement, set()).update(
            fact.source_ids
        )
        verified_facts.append(fact.model_dump(mode="json"))

    if test_mode:
        for fact in verified_facts:
            fact["verification_status"] = "unverified"
            fact["confidence"] = min(float(fact.get("confidence", 0.1)), 0.1)
            fact["notes"] = (
                "Synthetic offline test evidence. Verification is intentionally "
                "disabled in test mode."
            )
    else:
        for fact in verified_facts:
            normalized_statement = re.sub(
                r"\W+",
                " ",
                fact["statement"].casefold(),
                flags=re.UNICODE,
            ).strip()
            supporting_sources = statement_to_sources.get(
                normalized_statement,
                set(),
            )

            if len(supporting_sources) >= 2:
                fact["verification_status"] = "partially_verified"
                fact["confidence"] = min(
                    max(float(fact.get("confidence", 0.25)), 0.5),
                    0.75,
                )
                fact["notes"] = (
                    "Multiple source records reference this normalized claim. "
                    "Editorial review is still required to establish source "
                    "independence and factual accuracy."
                )
            else:
                fact["verification_status"] = "unverified"
                fact["confidence"] = min(
                    float(fact.get("confidence", 0.25)),
                    0.49,
                )

    result = dict(evidence)
    result["facts"] = verified_facts
    result["sources"] = [source.model_dump(mode="json") for source in validated_sources]

    return result


def _build_research_output(
    topic: str,
    language: str,
    sources: list[ResearchSource],
    evidence: dict[str, Any],
    test_mode: bool,
) -> ResearchOutput:
    """Build and validate a complete ResearchOutput model."""
    raw_facts = evidence.get("facts", [])
    facts = [ResearchFact.model_validate(item) for item in raw_facts]

    source_ids = {source.source_id for source in sources}
    facts = [
        fact
        for fact in facts
        if set(fact.source_ids).issubset(source_ids)
    ]

    timeline = [
        TimelineEvent.model_validate(item)
        for item in evidence.get("timeline", [])
    ]
    locations = [
        LocationRecord.model_validate(item)
        for item in evidence.get("locations", [])
    ]
    historical_context = [
        HistoricalContext.model_validate(item)
        for item in evidence.get("historical_context", [])
    ]

    if test_mode:
        research_status = "partial"
    elif not facts:
        research_status = "insufficient_evidence"
    elif any(fact.verification_status == "verified" for fact in facts):
        research_status = "complete"
    else:
        research_status = "partial"

    unresolved_questions = [
        _truncate(str(question), 2000)
        for question in evidence.get("unresolved_questions", [])
        if _normalize_text(str(question))
    ]

    if not sources:
        research_status = "insufficient_evidence"
        unresolved_questions.append("No sources were available for this research job.")

    if test_mode:
        unresolved_questions.append(
            "Offline test output is synthetic and is not suitable for factual "
            "publication or narration."
        )

    unique_questions = list(dict.fromkeys(unresolved_questions))

    return ResearchOutput(
        schema_version="1.0.0",
        job_id=uuid4(),
        topic=_truncate(topic, 2000),
        language=_truncate(language, 35),
        retrieved_at=_utc_now(),
        sources=sources,
        facts=facts,
        timeline=timeline,
        locations=locations,
        historical_context=historical_context,
        unresolved_questions=unique_questions,
        research_status=research_status,
    )


def run_research(
    topic: str,
    language: str,
    output_path: str | Path | None = None,
    max_sources: int = DEFAULT_MAX_SOURCES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    test_mode: bool = False,
) -> ResearchOutput:
    """Run the complete research stage and atomically write its JSON artifact.

    The artifact is written only within ``workspace/01_research_vault``.
    The destination is checked before writing, and ``write_json_artifact``
    performs an atomic replacement and calculates a SHA-256 digest.

    Args:
        topic: Topic to research.
        language: Language code for research metadata and news discovery.
        output_path: Optional absolute or workspace-relative JSON destination.
        max_sources: Maximum number of source records to include.
        timeout_seconds: Per-request network timeout in production mode.
        test_mode: If True, avoid all network calls and use offline fixtures.

    Returns:
        A validated ResearchOutput instance.

    Raises:
        ValueError: If input arguments are invalid.
        ResearchStageError: If the sandbox or research pipeline fails.
        ValidationError: If the output violates the strict data model.
        OSError: If the output artifact cannot be written.
    """
    active_config, research_workspace, destination = _resolve_research_workspace(
        output_path=output_path,
    )

    configured_test_mode = bool(getattr(active_config, "test_mode", False))
    effective_test_mode = bool(test_mode or configured_test_mode)

    normalized_topic = _normalize_text(topic)
    normalized_language = _normalize_text(language)

    if not normalized_topic:
        raise ValueError("topic must not be empty.")
    if not normalized_language:
        raise ValueError("language must not be empty.")

    LOGGER.info(
        "Starting research stage: topic=%r language=%r test_mode=%s",
        normalized_topic,
        normalized_language,
        effective_test_mode,
    )

    started_at = time.monotonic()

    try:
        sources = retrieve_sources(
            topic=normalized_topic,
            language=normalized_language,
            max_sources=max_sources,
            timeout_seconds=timeout_seconds,
            test_mode=effective_test_mode,
        )

        extracted = extract_research_evidence(
            sources=sources,
            topic=normalized_topic,
            language=normalized_language,
            test_mode=effective_test_mode,
        )

        verified = verify_research_evidence(
            evidence=extracted,
            sources=sources,
            test_mode=effective_test_mode,
        )

        output = _build_research_output(
            topic=normalized_topic,
            language=normalized_language,
            sources=sources,
            evidence=verified,
            test_mode=effective_test_mode,
        )

        # Resolve again immediately before writing to reduce the chance that
        # a changed symlink or destination path can escape the research sandbox.
        _, verified_workspace, verified_destination = _resolve_research_workspace(
            output_path=destination,
            config=active_config,
        )

        if verified_workspace != research_workspace:
            raise ResearchStageError(
                "Research workspace changed during execution; refusing to write."
            )

        write_result = write_json_artifact(
            verified_destination,
            output,
            overwrite=True,
            fsync=True,
        )

        # Read back and validate the exact bytes that were persisted.
        persisted_payload = json.loads(
            verified_destination.read_text(encoding="utf-8")
        )
        persisted_output = ResearchOutput.model_validate(persisted_payload)

        if persisted_output.job_id != output.job_id:
            raise ResearchStageError(
                "Persisted research artifact does not match the current job."
            )

        LOGGER.info(
            "Research stage completed: path=%s sha256=%s sources=%d "
            "facts=%d duration_seconds=%.3f status=%s",
            write_result.path,
            write_result.sha256,
            len(output.sources),
            len(output.facts),
            time.monotonic() - started_at,
            output.research_status,
        )

        return persisted_output

    except (ValidationError, ValueError, OSError, ResearchStageError):
        LOGGER.exception(
            "Research stage failed for topic=%r.",
            normalized_topic,
        )
        raise
    except Exception as exc:
        LOGGER.exception(
            "Unexpected research stage failure for topic=%r.",
            normalized_topic,
        )
        raise ResearchStageError(
            f"Unexpected research failure: {type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "ResearchStageError",
    "retrieve_sources",
    "extract_research_evidence",
    "verify_research_evidence",
    "run_research",
]
