from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
from os import getenv
import re
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from lightrag.config import settings

try:
    from playwright.async_api import (
        Error as PlaywrightError,
        TimeoutError as PlaywrightTimeoutError,
        async_playwright,
    )
except ImportError as exc:
    PlaywrightError = RuntimeError
    PlaywrightTimeoutError = TimeoutError
    async_playwright = None
    PLAYWRIGHT_IMPORT_ERROR = exc
else:
    PLAYWRIGHT_IMPORT_ERROR = None

HN_FRONT_PAGE_URL = "https://news.ycombinator.com/"
DEFAULT_BLOOM_HASH_COUNT = 7
DEFAULT_BLOOM_SIZE_MB = 1
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_STATE_DIR = "./data/hn_front_page_loader_state"
DEFAULT_WAIT_TIMEOUT = 900
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.5 Safari/605.1.15"
)
HTML_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
}
MAX_EXTERNAL_SOURCE_LINKS = 3
SUCCESS_DOC_STATUSES = {"processed", "preprocessed"}
TERMINAL_DOC_STATUSES = SUCCESS_DOC_STATUSES | {"failed"}


@dataclass(slots=True)
class HNPost:
    item_id: str
    rank: int
    title: str
    article_url: str
    discussion_url: str


@dataclass(slots=True)
class LoadedDocument:
    doc_id: str
    file_path: str
    title: str
    content: str


@dataclass(slots=True)
class SubmissionRecord:
    document: LoadedDocument
    response: dict[str, Any]


@dataclass(slots=True)
class HNDiscussionPage:
    html: str
    resolved_url: str
    content_type: str | None
    external_links: list[str]


@dataclass(slots=True)
class PersistentBloomFilter:
    file_path: Path
    num_bits: int
    num_hashes: int
    bits: bytearray
    dirty: bool = False

    @classmethod
    def load_or_create(
        cls,
        file_path: Path,
        *,
        requested_num_bits: int,
        requested_num_hashes: int,
    ) -> PersistentBloomFilter:
        expected_bytes = cls._byte_count_for_bits(requested_num_bits)
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            return cls(
                file_path=file_path,
                num_bits=requested_num_bits,
                num_hashes=requested_num_hashes,
                bits=bytearray(expected_bytes),
            )

        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            num_bits = int(payload["num_bits"])
            num_hashes = int(payload["num_hashes"])
            raw_bits = base64.b64decode(payload["bits_b64"], validate=True)
            loaded_expected_bytes = cls._byte_count_for_bits(num_bits)
            if len(raw_bits) != loaded_expected_bytes:
                raise ValueError(
                    f"Bloom filter byte length mismatch: expected {loaded_expected_bytes}, got {len(raw_bits)}"
                )
            if num_bits != requested_num_bits or num_hashes != requested_num_hashes:
                logging.info(
                    "Using existing bloom filter configuration bits=%d hashes=%d from %s",
                    num_bits,
                    num_hashes,
                    file_path,
                )
            return cls(
                file_path=file_path,
                num_bits=num_bits,
                num_hashes=num_hashes,
                bits=bytearray(raw_bits),
            )
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            logging.warning(
                "Bloom filter state at %s is invalid, recreating it: %s",
                file_path,
                exc,
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            return cls(
                file_path=file_path,
                num_bits=requested_num_bits,
                num_hashes=requested_num_hashes,
                bits=bytearray(expected_bytes),
            )

    @staticmethod
    def _byte_count_for_bits(num_bits: int) -> int:
        return (num_bits + 7) // 8

    def _iter_indexes(self, key: str):
        normalized_key = key.encode("utf-8")
        digest = hashlib.sha256(normalized_key).digest()
        primary_hash = int.from_bytes(digest[:16], "big")
        secondary_hash = int.from_bytes(digest[16:], "big") or 1
        for index in range(self.num_hashes):
            yield (primary_hash + index * secondary_hash) % self.num_bits

    def contains(self, key: str) -> bool:
        for index in self._iter_indexes(key):
            byte_index = index // 8
            bit_mask = 1 << (index % 8)
            if not (self.bits[byte_index] & bit_mask):
                return False
        return True

    def add(self, key: str) -> bool:
        changed = False
        for index in self._iter_indexes(key):
            byte_index = index // 8
            bit_mask = 1 << (index % 8)
            if self.bits[byte_index] & bit_mask:
                continue
            self.bits[byte_index] |= bit_mask
            changed = True

        if changed:
            self.dirty = True
        return changed

    def save(self) -> None:
        if not self.dirty:
            return

        payload = {
            "version": 1,
            "num_bits": self.num_bits,
            "num_hashes": self.num_hashes,
            "bits_b64": base64.b64encode(bytes(self.bits)).decode("ascii"),
        }
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.file_path.with_suffix(f"{self.file_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.file_path)
        self.dirty = False


class PlaywrightWebKitFetcher:
    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._context = None

    async def __aenter__(self) -> PlaywrightWebKitFetcher:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> None:
        if PLAYWRIGHT_IMPORT_ERROR is not None or async_playwright is None:
            raise RuntimeError(
                "Playwright is required for page extraction. Run `uv sync` and `uv run playwright install webkit`."
            ) from PLAYWRIGHT_IMPORT_ERROR

        if self._context is not None:
            return

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.webkit.launch(headless=True)
            self._context = await self._browser.new_context(
                user_agent=settings.hn_loader_user_agent or DEFAULT_USER_AGENT,
                locale="en-US",
                viewport={"width": 1440, "height": 900},
                screen={"width": 1440, "height": 900},
                device_scale_factor=2,
                is_mobile=False,
                has_touch=False,
                ignore_https_errors=True,
                extra_http_headers={
                    key: value
                    for key, value in build_page_headers().items()
                    if key != "User-Agent"
                },
            )
        except PlaywrightError as exc:
            await self.close()
            raise RuntimeError(
                "Failed to start Playwright WebKit. Ensure the browser runtime is installed with `uv run playwright install webkit`."
            ) from exc

    async def close(self) -> None:
        if self._context is not None:
            await self._context.close()
            self._context = None

        if self._browser is not None:
            await self._browser.close()
            self._browser = None

        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def fetch_html(
        self,
        url: str,
        *,
        timeout: int,
        max_bytes: int,
    ) -> tuple[str, str, str | None]:
        if self._context is None:
            await self.start()

        context = self._context
        if context is None:
            raise RuntimeError("Playwright browser context is not initialized")

        page = await context.new_page()
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            try:
                await page.wait_for_load_state("networkidle", timeout=min(timeout * 1000, 5000))
            except PlaywrightTimeoutError:
                pass

            final_url = page.url
            content_type = None
            if response is not None:
                headers = await response.all_headers()
                content_type = headers.get("content-type")

            html = await page.content()
            payload = html.encode("utf-8", errors="replace")
            if len(payload) > max_bytes:
                payload = payload[:max_bytes]
                html = payload.decode("utf-8", errors="ignore")

            return html, final_url, content_type
        finally:
            await page.close()

    async def fetch_hn_discussion_page(
        self,
        url: str,
        *,
        timeout: int,
        max_bytes: int,
    ) -> HNDiscussionPage:
        if self._context is None:
            await self.start()

        context = self._context
        if context is None:
            raise RuntimeError("Playwright browser context is not initialized")

        page = await context.new_page()
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            try:
                await page.wait_for_load_state("networkidle", timeout=min(timeout * 1000, 5000))
            except PlaywrightTimeoutError:
                pass

            final_url = page.url
            content_type = None
            if response is not None:
                headers = await response.all_headers()
                content_type = headers.get("content-type")

            html = await page.content()
            payload = html.encode("utf-8", errors="replace")
            if len(payload) > max_bytes:
                payload = payload[:max_bytes]
                html = payload.decode("utf-8", errors="ignore")

            external_links = await extract_hn_external_links_from_page(page)
            return HNDiscussionPage(
                html=html,
                resolved_url=final_url,
                content_type=content_type,
                external_links=external_links,
            )
        finally:
            await page.close()


class HNFrontPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.posts: list[HNPost] = []
        self._current_item_id: str | None = None
        self._current_rank = 0
        self._in_titleline = False
        self._capture_title = False
        self._captured_title_link = False
        self._current_title_parts: list[str] = []
        self._current_href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())

        if tag == "tr" and "athing" in classes and attr_map.get("id"):
            self._current_item_id = attr_map["id"]
            self._current_title_parts = []
            self._current_href = ""
            self._captured_title_link = False
            self._current_rank += 1
            return

        if self._current_item_id is None:
            return

        if tag == "span" and "titleline" in classes:
            self._in_titleline = True
            return

        if (
            self._in_titleline
            and tag == "a"
            and not self._capture_title
            and not self._captured_title_link
        ):
            self._capture_title = True
            self._captured_title_link = True
            self._current_href = attr_map.get("href", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            self._capture_title = False
            return

        if tag == "span" and self._in_titleline:
            self._in_titleline = False
            title = collapse_whitespace("".join(self._current_title_parts))
            if self._current_item_id and title:
                discussion_url = f"{HN_FRONT_PAGE_URL}item?id={self._current_item_id}"
                article_url = urljoin(HN_FRONT_PAGE_URL, self._current_href)
                self.posts.append(
                    HNPost(
                        item_id=self._current_item_id,
                        rank=self._current_rank,
                        title=title,
                        article_url=article_url,
                        discussion_url=discussion_url,
                    )
                )
            self._current_item_id = None
            self._current_title_parts = []
            self._current_href = ""
            self._captured_title_link = False

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._current_title_parts.append(data)


class HTMLTextExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe", "head"}
    _BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = collapse_whitespace(data)
        if text:
            self._chunks.append(text)

    def get_text(self, max_chars: int) -> str:
        text = "".join(self._chunks)
        lines = [collapse_whitespace(line) for line in text.splitlines()]
        normalized = "\n".join(line for line in lines if line)
        if len(normalized) > max_chars:
            return normalized[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."
        return normalized


class HNDiscussionLinkParser(HTMLParser):
    _SCOPE_PRIORITY = ("toptext", "commtext", "titleline")

    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[dict[str, str]] = []
        self._active_scope_depths = {scope: 0 for scope in self._SCOPE_PRIORITY}
        self._open_tags: list[tuple[str, tuple[str, ...]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())
        opened_scopes = tuple(scope for scope in self._SCOPE_PRIORITY if scope in classes)
        self._open_tags.append((tag, opened_scopes))

        for scope in opened_scopes:
            self._active_scope_depths[scope] += 1

        active_scope = next(
            (scope for scope in self._SCOPE_PRIORITY if self._active_scope_depths[scope] > 0),
            None,
        )
        if tag == "a" and active_scope:
            href = attr_map.get("href", "")
            if href:
                self.candidates.append(
                    {
                        "href": urljoin(HN_FRONT_PAGE_URL, href),
                        "scope": active_scope,
                    }
                )

    def handle_endtag(self, tag: str) -> None:
        while self._open_tags:
            open_tag, opened_scopes = self._open_tags.pop()
            for scope in opened_scopes:
                if self._active_scope_depths[scope] > 0:
                    self._active_scope_depths[scope] -= 1
            if open_tag == tag:
                break


def collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def workspace_state_key(workspace: str | None) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", (workspace or "default").strip())
    return normalized or "default"


def build_bloom_filter_path(state_dir: str, workspace: str | None) -> Path:
    return Path(state_dir) / workspace_state_key(workspace) / "seen_pages.bloom.json"


def compute_bloom_bits(size_mb: int) -> int:
    return max(size_mb, 1) * 1024 * 1024 * 8


def normalize_api_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("LightRAG API base URL cannot be empty")
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    return normalized


def build_api_base_url(explicit_base_url: str | None) -> str:
    if explicit_base_url:
        return normalize_api_base_url(explicit_base_url)

    env_base_url = settings.lightrag_api_url
    if env_base_url:
        return normalize_api_base_url(env_base_url)

    host = settings.host.strip() or "127.0.0.1"
    port = str(settings.port).strip() or "9621"
    return normalize_api_base_url(f"http://{host}:{port}")


def build_page_headers() -> dict[str, str]:
    return {
        "User-Agent": settings.hn_loader_user_agent or DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def build_api_headers(api_key: str | None, workspace: str | None) -> dict[str, str]:
    headers = {
        "User-Agent": "LightRAG-HN-Loader/1.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["X-API-Key"] = api_key
    if workspace:
        headers["LIGHTRAG-WORKSPACE"] = workspace
    return headers


def build_endpoint_url(api_base_url: str, path: str) -> str:
    normalized_base_url = normalize_api_base_url(api_base_url)
    normalized_path = path if path.startswith("/") else f"/{path}"
    if normalized_base_url.endswith("/documents") and normalized_path.startswith("/documents/"):
        normalized_path = normalized_path[len("/documents") :]
    return f"{normalized_base_url}{normalized_path}"


def decode_json_response(response) -> dict[str, Any]:
    raw_body = response.read()
    if not raw_body:
        return {}

    charset = response.headers.get_content_charset() or "utf-8"
    return json.loads(raw_body.decode(charset, errors="replace"))


def request_json(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    timeout: int,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_data = None
    if payload is not None:
        request_data = json.dumps(payload).encode("utf-8")

    request = Request(url, headers=headers, data=request_data, method=method)

    try:
        with urlopen(request, timeout=timeout) as response:
            return decode_json_response(response)
    except HTTPError as exc:
        error_body = exc.read().decode(
            exc.headers.get_content_charset() or "utf-8",
            errors="replace",
        )
        try:
            detail = json.loads(error_body)
        except json.JSONDecodeError:
            detail = error_body.strip() or exc.reason
        raise RuntimeError(
            f"API {method} {url} failed with HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"API {method} {url} failed: {exc.reason}") from exc


def fetch_html(url: str, timeout: int, max_bytes: int) -> tuple[str, str, str | None]:
    request = Request(url, headers=build_page_headers(), method="GET")
    with urlopen(request, timeout=timeout) as response:
        content_type = (response.headers.get_content_type() or "").lower()
        final_url = response.geturl()
        if content_type and content_type not in HTML_CONTENT_TYPES and not content_type.startswith("text/"):
            return "", final_url, content_type

        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            payload = payload[:max_bytes]
        charset = response.headers.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace"), final_url, content_type


async def fetch_page_html(
    url: str,
    *,
    timeout: int,
    max_bytes: int,
    fetcher: PlaywrightWebKitFetcher,
) -> tuple[str, str, str | None]:
    try:
        return await fetcher.fetch_html(url, timeout=timeout, max_bytes=max_bytes)
    except RuntimeError:
        raise
    except PlaywrightError as exc:
        raise RuntimeError(f"Playwright failed to fetch {url}: {exc}") from exc


def parse_front_page(html: str, max_posts: int) -> list[HNPost]:
    parser = HNFrontPageParser()
    parser.feed(html)
    return parser.posts[:max_posts]


def extract_text_from_html(html: str, max_chars: int) -> str:
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text(max_chars=max_chars)


def normalize_hn_external_link_candidates(candidates: list[dict[str, str]]) -> list[str]:
    unique_links: list[str] = []
    seen_links: set[str] = set()

    for candidate in candidates:
        link = candidate.get("href", "")
        normalized_link = normalize_url_identity(link)
        if not normalized_link or normalized_link in seen_links:
            continue

        parsed = urlparse(normalized_link)
        if parsed.scheme not in {"http", "https"}:
            continue

        if parsed.netloc.lower() in {"news.ycombinator.com", "hn.algolia.com"}:
            continue

        seen_links.add(normalized_link)
        unique_links.append(normalized_link)
        if len(unique_links) >= MAX_EXTERNAL_SOURCE_LINKS:
            break

    return unique_links


def extract_hn_external_links(html: str) -> list[str]:
    parser = HNDiscussionLinkParser()
    parser.feed(html)
    return normalize_hn_external_link_candidates(parser.candidates)


async def extract_hn_external_links_from_page(page) -> list[str]:
    candidates = await page.evaluate("""
        () => {
            const selectors = [
                ['toptext', 'div.toptext a[href]'],
                ['commtext', '.commtext a[href]'],
                ['titleline', 'span.titleline a[href]'],
            ];
            return selectors.flatMap(([scope, selector]) =>
                Array.from(document.querySelectorAll(selector)).map((anchor) => ({
                    href: anchor.href || anchor.getAttribute('href') || '',
                    scope,
                }))
            );
        }
    """)
    if not isinstance(candidates, list):
        return []
    normalized_candidates = [
        candidate for candidate in candidates if isinstance(candidate, dict)
    ]
    return normalize_hn_external_link_candidates(normalized_candidates)


async def fetch_hn_discussion_page(
    url: str,
    *,
    timeout: int,
    max_bytes: int,
    fetcher: PlaywrightWebKitFetcher,
) -> HNDiscussionPage:
    try:
        return await fetcher.fetch_hn_discussion_page(
            url,
            timeout=timeout,
            max_bytes=max_bytes,
        )
    except RuntimeError:
        raise
    except PlaywrightError as exc:
        raise RuntimeError(f"Playwright failed to fetch {url}: {exc}") from exc


def filter_external_source_urls(
    source_urls: list[str], *, exclude_urls: list[str] | None = None
) -> list[str]:
    excluded_urls = {
        normalized
        for normalized in (
            normalize_url_identity(url) for url in (exclude_urls or [])
        )
        if normalized
    }
    filtered_urls: list[str] = []

    for source_url in source_urls:
        normalized_url = normalize_url_identity(source_url)
        if not normalized_url or normalized_url in excluded_urls:
            continue
        excluded_urls.add(normalized_url)
        filtered_urls.append(source_url)
        if len(filtered_urls) >= MAX_EXTERNAL_SOURCE_LINKS:
            break

    return filtered_urls


async def load_external_source_texts(
    source_urls: list[str],
    *,
    timeout: int,
    max_html_bytes: int,
    max_content_chars: int,
    fetcher: PlaywrightWebKitFetcher,
) -> list[tuple[str, str]]:
    if not source_urls:
        return []

    per_source_chars = max(
        2000,
        max_content_chars // max(1, min(len(source_urls), MAX_EXTERNAL_SOURCE_LINKS)),
    )
    external_sources: list[tuple[str, str]] = []

    for index, source_url in enumerate(source_urls, start=1):
        logging.info(
            "Following linked source %d/%d: %s",
            index,
            len(source_urls),
            source_url,
        )
        try:
            html, resolved_url, _ = await fetch_page_html(
                source_url,
                timeout=timeout,
                max_bytes=max_html_bytes,
                fetcher=fetcher,
            )
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            logging.warning("Failed to fetch linked source %s: %s", source_url, exc)
            continue

        if not html:
            logging.info("Linked source returned no readable HTML: %s", source_url)
            continue

        extracted_text = extract_text_from_html(html, max_chars=per_source_chars)
        if not extracted_text:
            logging.info("Linked source text extraction was empty: %s", resolved_url)
            continue

        external_sources.append((resolved_url, extracted_text))
        logging.info(
            "Loaded linked source resolved=%s chars=%d",
            resolved_url,
            len(extracted_text),
        )

    return external_sources


def should_use_discussion_page(post: HNPost) -> bool:
    article_url = post.article_url.lower()
    parsed = urlparse(article_url)
    if parsed.path == "/from":
        return False
    return (
        parsed.netloc in {"news.ycombinator.com", ""}
        or parsed.path == "/item"
        or article_url.startswith("item?id=")
    )


def predicted_file_source(post: HNPost) -> str:
    if should_use_discussion_page(post):
        return post.discussion_url
    return post.article_url or post.discussion_url


def normalize_url_identity(url: str) -> str:
    normalized_url = (url or "").strip()
    if not normalized_url:
        return normalized_url

    parsed = urlparse(normalized_url)
    if not parsed.scheme and not parsed.netloc:
        return normalized_url

    return parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        fragment="",
    ).geturl()


def bloom_key_for_post(post: HNPost) -> str:
    return normalize_url_identity(predicted_file_source(post)) or f"hn-item:{post.item_id}"


def bloom_key_for_document(document: LoadedDocument) -> str:
    return normalize_url_identity(document.file_path) or document.doc_id


def build_document_text(
    post: HNPost,
    *,
    source_url: str,
    source_kind: str,
    extracted_text: str,
    external_sources: list[tuple[str, str]] | None = None,
) -> str:
    sections = [
        f"Title: {post.title}",
        f"Hacker News Rank: {post.rank}",
        f"Hacker News Item ID: {post.item_id}",
        f"Article URL: {post.article_url}",
        f"Discussion URL: {post.discussion_url}",
        f"Loaded From: {source_kind}",
        f"Resolved Source URL: {source_url}",
    ]
    if extracted_text:
        sections.append(f"Content:\n{extracted_text}")

    for index, (external_url, external_text) in enumerate(external_sources or [], start=1):
        sections.append(
            f"Linked Source {index} URL: {external_url}\n\n"
            f"Linked Source {index} Content:\n{external_text}"
        )

    if not extracted_text and not external_sources:
        sections.append(
            "Content:\nUnable to extract readable page text, so this document contains post metadata only."
        )
    return "\n\n".join(sections)


async def load_post_document(
    post: HNPost,
    *,
    timeout: int,
    max_html_bytes: int,
    max_content_chars: int,
    fetcher: PlaywrightWebKitFetcher,
) -> LoadedDocument:
    primary_url = post.discussion_url if should_use_discussion_page(post) else post.article_url
    primary_kind = "hn_discussion" if primary_url == post.discussion_url else "article"

    extracted_text = ""
    external_sources: list[tuple[str, str]] = []
    resolved_url = primary_url
    discussion_html = ""
    discussion_resolved_url = post.discussion_url
    discussion_links: list[str] = []

    logging.info(
        "Loading HN post rank=%d item=%s primary=%s url=%s",
        post.rank,
        post.item_id,
        primary_kind,
        primary_url,
    )

    try:
        if primary_url == post.discussion_url:
            discussion_page = await fetch_hn_discussion_page(
                primary_url,
                timeout=timeout,
                max_bytes=max_html_bytes,
                fetcher=fetcher,
            )
            html = discussion_page.html
            resolved_url = discussion_page.resolved_url
            discussion_html = discussion_page.html
            discussion_resolved_url = discussion_page.resolved_url
            discussion_links = discussion_page.external_links
        else:
            html, resolved_url, _ = await fetch_page_html(
                primary_url,
                timeout=timeout,
                max_bytes=max_html_bytes,
                fetcher=fetcher,
            )
        if html:
            extracted_text = extract_text_from_html(html, max_chars=max_content_chars)
            logging.info(
                "Loaded primary source for item=%s resolved=%s chars=%d",
                post.item_id,
                resolved_url,
                len(extracted_text),
            )
    except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
        logging.warning(
            "Failed to fetch primary source for item=%s url=%s: %s",
            post.item_id,
            primary_url,
            exc,
        )
        extracted_text = ""

    if discussion_html:
        if not discussion_links:
            discussion_links = extract_hn_external_links(discussion_html)
        discussion_links = filter_external_source_urls(
            discussion_links,
            exclude_urls=[
                post.article_url,
                primary_url,
                resolved_url,
                post.discussion_url,
                discussion_resolved_url,
            ],
        )
        if discussion_links:
            logging.info(
                "Following %d linked source page(s) for HN item=%s",
                len(discussion_links),
                post.item_id,
            )
            external_sources = await load_external_source_texts(
                discussion_links,
                timeout=timeout,
                max_html_bytes=max_html_bytes,
                max_content_chars=max_content_chars,
                fetcher=fetcher,
            )
        else:
            logging.info("No extra linked source pages found for HN item=%s", post.item_id)

    if primary_url == post.discussion_url and external_sources:
        primary_kind = "hn_discussion_with_external"

    file_path = post.article_url if post.article_url else post.discussion_url
    if primary_kind.startswith("hn_discussion"):
        file_path = post.discussion_url

    return LoadedDocument(
        doc_id=f"hn-{post.item_id}",
        file_path=file_path,
        title=post.title,
        content=build_document_text(
            post,
            source_url=resolved_url,
            source_kind=primary_kind,
            extracted_text=extracted_text,
            external_sources=external_sources,
        ),
    )


async def collect_documents(
    posts: list[HNPost],
    *,
    timeout: int,
    max_html_bytes: int,
    max_content_chars: int,
    concurrency: int,
    fetcher: PlaywrightWebKitFetcher,
) -> list[LoadedDocument]:
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(post: HNPost) -> LoadedDocument:
        async with semaphore:
            return await load_post_document(
                post,
                timeout=timeout,
                max_html_bytes=max_html_bytes,
                max_content_chars=max_content_chars,
                fetcher=fetcher,
            )

    return await asyncio.gather(*(worker(post) for post in posts))


def submit_document(
    document: LoadedDocument,
    *,
    api_base_url: str,
    api_key: str | None,
    workspace: str | None,
    timeout: int,
) -> dict[str, Any]:
    headers = build_api_headers(api_key=api_key, workspace=workspace)

    return request_json(
        build_endpoint_url(api_base_url, "/documents/text"),
        method="POST",
        headers=headers,
        timeout=timeout,
        payload={
            "text": document.content,
            "file_source": document.file_path,
        },
    )


def submit_documents_batch(
    documents: list[LoadedDocument],
    *,
    api_base_url: str,
    api_key: str | None,
    workspace: str | None,
    timeout: int,
) -> dict[str, Any]:
    headers = build_api_headers(api_key=api_key, workspace=workspace)
    return request_json(
        build_endpoint_url(api_base_url, "/documents/texts"),
        method="POST",
        headers=headers,
        timeout=timeout,
        payload={
            "texts": [document.content for document in documents],
            "file_sources": [document.file_path for document in documents],
        },
    )


def submit_documents_resilient(
    documents: list[LoadedDocument],
    *,
    api_base_url: str,
    api_key: str | None,
    workspace: str | None,
    timeout: int,
) -> list[SubmissionRecord]:
    if not documents:
        return []

    if len(documents) == 1:
        return [
            SubmissionRecord(
                document=documents[0],
                response=submit_document(
                    documents[0],
                    api_base_url=api_base_url,
                    api_key=api_key,
                    workspace=workspace,
                    timeout=timeout,
                ),
            )
        ]

    batch_response = submit_documents_batch(
        documents,
        api_base_url=api_base_url,
        api_key=api_key,
        workspace=workspace,
        timeout=timeout,
    )
    if batch_response.get("status") != "duplicated":
        return [
            SubmissionRecord(document=document, response=batch_response)
            for document in documents
        ]

    logging.info(
        "Batch submission reported duplicates; retrying %d documents individually.",
        len(documents),
    )
    return [
        SubmissionRecord(
            document=document,
            response=submit_document(
                document,
                api_base_url=api_base_url,
                api_key=api_key,
                workspace=workspace,
                timeout=timeout,
            ),
        )
        for document in documents
    ]


def normalize_doc_status(status: Any) -> str:
    normalized = str(status or "unknown").strip().lower()
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]
    return normalized


def wait_for_track_completion(
    *,
    api_base_url: str,
    api_key: str | None,
    workspace: str | None,
    track_id: str,
    request_timeout: int,
    poll_interval: float,
    wait_timeout: int,
) -> dict[str, Any]:
    headers = build_api_headers(api_key=api_key, workspace=workspace)
    endpoint_url = build_endpoint_url(api_base_url, f"/documents/track_status/{track_id}")
    deadline = time.monotonic() + wait_timeout
    last_snapshot: str | None = None

    while True:
        response = request_json(
            endpoint_url,
            method="GET",
            headers=headers,
            timeout=request_timeout,
        )
        documents = response.get("documents") or []
        status_summary = response.get("status_summary") or {}
        normalized_summary = {
            normalize_doc_status(status): count
            for status, count in status_summary.items()
        }
        snapshot = json.dumps(normalized_summary, sort_keys=True)
        if snapshot != last_snapshot:
            logging.info("Track %s status: %s", track_id, normalized_summary or {"pending": len(documents)})
            last_snapshot = snapshot

        document_statuses = [normalize_doc_status(document.get("status")) for document in documents]
        if documents and all(status in TERMINAL_DOC_STATUSES for status in document_statuses):
            return response

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for LightRAG track_id {track_id} after {wait_timeout} seconds"
            )

        time.sleep(max(poll_interval, 0.1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape the Hacker News front page, fetch each post's article or discussion text, "
            "and insert one document per post into LightRAG through the document insertion API."
        )
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="LightRAG API base URL. Defaults to LIGHTRAG_API_URL or http://HOST:PORT from .env.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override for the X-API-Key header. Defaults to LIGHTRAG_API_KEY from .env.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Optional LightRAG workspace header value.",
    )
    parser.add_argument(
        "--state-dir",
        default=DEFAULT_STATE_DIR,
        help="Directory for persistent local loader state, including the Bloom filter. Default: %(default)s",
    )
    parser.add_argument(
        "--bloom-size-mb",
        type=int,
        default=DEFAULT_BLOOM_SIZE_MB,
        help="Approximate Bloom filter size in MiB for seen-page tracking. Default: %(default)s",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Ignore the local Bloom filter for this run and resubmit current posts. Server-side duplicates still apply.",
    )
    parser.add_argument(
        "--page-url",
        default=HN_FRONT_PAGE_URL,
        help="Front page URL to scrape. Default: %(default)s",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=30,
        help="Maximum number of posts to load from the first page. Default: %(default)s",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout in seconds for page fetches and API requests. Default: %(default)s",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Maximum concurrent page fetches. Default: %(default)s",
    )
    parser.add_argument(
        "--max-html-bytes",
        type=int,
        default=2_000_000,
        help="Maximum bytes to read from each fetched page. Default: %(default)s",
    )
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=15_000,
        help="Maximum extracted text length per post. Default: %(default)s",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Polling interval in seconds for /documents/track_status. Default: %(default)s",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=DEFAULT_WAIT_TIMEOUT,
        help="Maximum seconds to wait for document processing to finish. Default: %(default)s",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return after enqueuing documents instead of polling track status to completion.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for the loader script.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


async def main_async(args: argparse.Namespace) -> int:
    api_base_url = build_api_base_url(args.api_base_url)
    api_key = args.api_key or settings.lightrag_api_key or getenv("LIGHTRAG_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API key is required but not set. Provide it through --api-key or LIGHTRAG_API_KEY environment variable."
        )
    bloom_filter = PersistentBloomFilter.load_or_create(
        build_bloom_filter_path(args.state_dir, args.workspace),
        requested_num_bits=compute_bloom_bits(args.bloom_size_mb),
        requested_num_hashes=DEFAULT_BLOOM_HASH_COUNT,
    )

    try:
        try:
            async with PlaywrightWebKitFetcher() as fetcher:
                try:
                    front_page_html, resolved_url, _ = await fetch_page_html(
                        args.page_url,
                        timeout=args.timeout,
                        max_bytes=args.max_html_bytes,
                        fetcher=fetcher,
                    )
                except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
                    logging.error("Failed to fetch Hacker News front page: %s", exc)
                    return 1

                posts = parse_front_page(front_page_html, max_posts=args.max_posts)
                if not posts:
                    logging.error("No posts were parsed from %s", resolved_url)
                    return 1

                logging.info("Parsed %d posts from %s", len(posts), resolved_url)

                skipped_posts: list[HNPost] = []
                candidate_posts: list[HNPost] = []
                if args.reindex:
                    candidate_posts = posts
                    logging.info("Reindex mode enabled; bypassing local Bloom filter for %d posts", len(posts))
                else:
                    for post in posts:
                        if bloom_filter.contains(bloom_key_for_post(post)):
                            skipped_posts.append(post)
                        else:
                            candidate_posts.append(post)
                    if skipped_posts:
                        logging.info(
                            "Skipped %d already-seen posts from local Bloom filter state %s",
                            len(skipped_posts),
                            bloom_filter.file_path,
                        )

                if not candidate_posts:
                    logging.info("No new posts to index after local duplicate filtering.")
                    return 0

                documents = await collect_documents(
                    candidate_posts,
                    timeout=args.timeout,
                    max_html_bytes=args.max_html_bytes,
                    max_content_chars=args.max_content_chars,
                    concurrency=max(1, args.concurrency),
                    fetcher=fetcher,
                )
        except RuntimeError as exc:
            logging.error("Failed to initialize Playwright WebKit fetcher: %s", exc)
            return 1

        try:
            submission_records = await asyncio.to_thread(
                submit_documents_resilient,
                documents,
                api_base_url=api_base_url,
                api_key=api_key,
                workspace=args.workspace,
                timeout=args.timeout,
            )
        except RuntimeError as exc:
            logging.error("Failed to submit documents to LightRAG API: %s", exc)
            return 1

        track_documents: dict[str, list[LoadedDocument]] = {}
        queued_count = 0
        submission_failed = False
        for submission in submission_records:
            status = normalize_doc_status(submission.response.get("status"))
            track_id = str(submission.response.get("track_id") or "")
            logging.info(
                "API enqueue file=%s status=%s track_id=%s message=%s",
                submission.document.file_path,
                status,
                track_id or "-",
                submission.response.get("message", ""),
            )

            if status == "duplicated":
                bloom_filter.add(bloom_key_for_document(submission.document))
                continue

            if status in {"success", "partial_success"}:
                queued_count += 1
                if args.no_wait:
                    logging.info(
                        "Queued %s -> %s track_id=%s",
                        submission.document.doc_id,
                        submission.document.title,
                        track_id or "-",
                    )
                    continue
                if not track_id:
                    logging.error(
                        "Submission for %s succeeded without a track_id; cannot verify indexing.",
                        submission.document.file_path,
                    )
                    submission_failed = True
                    continue
                track_documents.setdefault(track_id, []).append(submission.document)
                continue

            submission_failed = True
            logging.error(
                "Failed to enqueue %s [%s]: %s",
                submission.document.file_path,
                status,
                submission.response.get("message", "unknown error"),
            )

        if args.no_wait:
            logging.info(
                "Queued %d Hacker News post(s) via API %s and returned without waiting for pipeline completion.",
                queued_count,
                api_base_url,
            )
            return 1 if submission_failed else 0

        failed_documents: list[str] = []
        timed_out_tracks: list[str] = []
        for track_id, queued_documents in track_documents.items():
            try:
                track_status = await asyncio.to_thread(
                    wait_for_track_completion,
                    api_base_url=api_base_url,
                    api_key=api_key,
                    workspace=args.workspace,
                    track_id=track_id,
                    request_timeout=args.timeout,
                    poll_interval=args.poll_interval,
                    wait_timeout=args.wait_timeout,
                )
            except TimeoutError as exc:
                timed_out_tracks.append(track_id)
                logging.warning(
                    "Timed out while waiting for LightRAG track status %s: %s",
                    track_id,
                    exc,
                )
                logging.warning(
                    "Track %s remains queued in LightRAG; check /documents/pipeline_status or rerun with --no-wait for fire-and-forget mode.",
                    track_id,
                )
                continue
            except RuntimeError as exc:
                logging.error(
                    "Failed while waiting for LightRAG track status %s: %s",
                    track_id,
                    exc,
                )
                return 1

            expected_file_paths = {
                bloom_key_for_document(document): document.file_path
                for document in queued_documents
            }
            observed_file_paths: set[str] = set()
            for tracked_document in track_status.get("documents") or []:
                file_path = tracked_document.get("file_path") or "unknown_source"
                tracked_status = normalize_doc_status(tracked_document.get("status"))
                bloom_key = normalize_url_identity(file_path) or file_path
                observed_file_paths.add(bloom_key)
                if tracked_status in SUCCESS_DOC_STATUSES:
                    bloom_filter.add(bloom_key)
                    logging.info("Indexed %s [%s]", file_path, tracked_status)
                    continue

                failed_documents.append(file_path)
                logging.error(
                    "Failed %s [%s]: %s",
                    file_path,
                    tracked_status,
                    tracked_document.get("error_msg") or "no error message",
                )

            missing_paths = sorted(set(expected_file_paths) - observed_file_paths)
            for missing_path in missing_paths:
                failed_documents.append(expected_file_paths[missing_path])
                logging.error(
                    "Track %s did not return final status for %s",
                    track_id,
                    expected_file_paths[missing_path],
                )

        logging.info(
            "Indexed %d candidate Hacker News posts via API %s",
            len(documents),
            api_base_url,
        )
        if timed_out_tracks:
            logging.warning(
                "Timed out while waiting for %d track(s); ingestion is still running in the LightRAG background pipeline.",
                len(timed_out_tracks),
            )
        for document in documents:
            logging.info("Loaded %s -> %s", document.doc_id, document.title)
        return 1 if submission_failed or failed_documents else 0
    finally:
        bloom_filter.save()


def main() -> int:
    load_dotenv(dotenv_path=".env", override=False)
    args = parse_args()
    configure_logging(args.verbose)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())