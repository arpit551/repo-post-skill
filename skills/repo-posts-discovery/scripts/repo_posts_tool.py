#!/usr/bin/env python3
"""Skill utility for syncing and searching repo_posts without MCP.

Usage examples:
  repo_posts_tool.py sync --bootstrap-limit 1000
  repo_posts_tool.py latest --limit 10
  repo_posts_tool.py grep --pattern 'mcp|model context protocol' --limit 20
  repo_posts_tool.py semantic --query 'open source ai agent framework' --limit 15
  repo_posts_tool.py hybrid --query 'self-hosted security' --pattern 'waf|threat' --limit 15
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import html
import inspect
import json
import math
import os
import pickle
import re
import subprocess
import sys
from array import array
from collections import Counter
from dataclasses import asdict, dataclass as _dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

DEFAULT_FEED_URL = "https://tom-doerr.github.io/repo_posts/feed.xml"
DEFAULT_SEARCH_INDEX_URL = "https://tom-doerr.github.io/repo_posts/assets/search-index.json"
DEFAULT_BASE_URL = "https://tom-doerr.github.io/repo_posts"
DEFAULT_AUTO_SYNC_MAX_AGE_MINUTES = 1440

CACHE_FILE = "posts.jsonl"
STATE_FILE = "state.json"
SEM_FILE = "semantic_index.pkl"
GITHUB_CACHE_FILE = "github_repo_cache.json"

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+_.-]{1,}")
GITHUB_LINK_RE = re.compile(r"https?://github\.com/([^/]+)/([^/#?]+)", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://github\.com/[^)]+)\)", re.IGNORECASE)

SYNONYM_CANONICAL: dict[str, str] = {
    "repos": "repo",
    "repository": "repo",
    "repositories": "repo",
    "projects": "project",
    "agents": "agent",
    "agentic": "agent",
    "llms": "llm",
    "models": "model",
    "frameworks": "framework",
    "searching": "search",
    "searched": "search",
    "indexing": "index",
    "indexed": "index",
    "opensource": "open-source",
    "oss": "open-source",
    "selfhosted": "self-hosted",
    "self-host": "self-hosted",
    "security": "secure",
    "devtool": "tool",
    "devtools": "tool",
}

# Python 3.9 doesn't support dataclass(slots=...); ignore that kwarg there.
if "slots" in inspect.signature(_dataclass).parameters:
    dataclass = _dataclass
else:
    def dataclass(*args, **kwargs):
        kwargs.pop("slots", None)
        return _dataclass(*args, **kwargs)


@dataclass(slots=True)
class RepoPost:
    post_url: str
    repo_url: str
    owner: str
    repo: str
    title: str
    summary: str
    updated_at: str
    published_date: str
    image_url: str
    search_text: str
    source: str


@dataclass(slots=True)
class SemanticArtifact:
    signature: str
    post_urls: list[str]
    idf: dict[str, float]
    dim: int
    projections_per_token: int
    matrix: array


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "sync":
        result = sync_cache(
            cache_dir=cache_dir,
            feed_url=args.feed_url,
            search_index_url=args.search_index_url,
            base_url=args.base_url,
            bootstrap_limit=args.bootstrap_limit,
            feed_limit=args.feed_limit,
            force_full=args.force_full,
        )
        _print_json(result)
        return 0

    auto_sync_result: dict[str, Any] | None = None
    if not args.no_auto_sync:
        auto_sync_result = maybe_auto_sync_cache(
            cache_dir=cache_dir,
            max_age_minutes=max(0, int(args.auto_sync_max_age_minutes)),
            bootstrap_limit=max(0, int(args.auto_sync_bootstrap_limit)),
            feed_limit=max(0, int(args.auto_sync_feed_limit)),
        )

    posts = load_posts(cache_dir / CACHE_FILE)
    if not posts:
        fail = {
            "error": "cache-empty",
            "message": "No cached posts found. Run sync first.",
            "hint": "repo_posts_tool.py sync",
        }
        _print_json(fail)
        return 2

    if args.command == "latest":
        rows = latest_posts(posts, limit=args.limit, since=args.since)
        rows, github_meta = finalize_rows(rows, args=args, cache_dir=cache_dir, default_sort="updated")
        _print_json({"results": rows, "count": len(rows), "github": github_meta, "auto_sync": auto_sync_result})
        return 0

    if args.command == "grep":
        rows = grep_search(
            posts,
            pattern=args.pattern,
            limit=args.limit,
            case_sensitive=args.case_sensitive,
            fields=args.fields,
        )
        rows, github_meta = finalize_rows(rows, args=args, cache_dir=cache_dir, default_sort="updated")
        _print_json({"results": rows, "count": len(rows), "github": github_meta, "auto_sync": auto_sync_result})
        return 0

    if args.command == "semantic":
        rows = semantic_search(
            posts,
            cache_dir=cache_dir,
            query=args.query,
            limit=args.limit,
            recency_days=args.recency_days,
        )
        rows, github_meta = finalize_rows(rows, args=args, cache_dir=cache_dir, default_sort="relevance")
        _print_json({"results": rows, "count": len(rows), "github": github_meta, "auto_sync": auto_sync_result})
        return 0

    if args.command == "hybrid":
        rows = hybrid_search(
            posts,
            cache_dir=cache_dir,
            query=args.query,
            pattern=args.pattern,
            limit=args.limit,
        )
        rows, github_meta = finalize_rows(rows, args=args, cache_dir=cache_dir, default_sort="relevance")
        _print_json({"results": rows, "count": len(rows), "github": github_meta, "auto_sync": auto_sync_result})
        return 0

    _print_json({"error": "unknown-command", "command": args.command})
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync and search repo_posts dataset for skill workflows")
    default_cache_dir = str(Path(__file__).resolve().parent.parent / "data")

    sub = parser.add_subparsers(dest="command")
    
    def add_cache_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--cache-dir",
            default=default_cache_dir,
            help="Directory for cache/state/index files",
        )
        p.add_argument(
            "--no-auto-sync",
            action="store_true",
            help="Do not auto-sync when cache is missing or stale",
        )
        p.add_argument(
            "--auto-sync-max-age-minutes",
            type=int,
            default=DEFAULT_AUTO_SYNC_MAX_AGE_MINUTES,
            help="Auto-sync when last sync is older than this many minutes (default: 1440 = 1 day)",
        )
        p.add_argument(
            "--auto-sync-bootstrap-limit",
            type=int,
            default=1200,
            help="Bootstrap limit used by auto-sync when cache is empty",
        )
        p.add_argument(
            "--auto-sync-feed-limit",
            type=int,
            default=300,
            help="Feed limit used by auto-sync incremental updates",
        )
        p.add_argument(
            "--with-github",
            action="store_true",
            help="Fetch GitHub metadata (stars, latest commit date) for returned repos",
        )
        p.add_argument(
            "--github-limit",
            type=int,
            default=30,
            help="Maximum repos to enrich from GitHub API per command",
        )
        p.add_argument(
            "--github-refresh-hours",
            type=float,
            default=24.0,
            help="Hours before cached GitHub metadata is refreshed",
        )
        p.add_argument(
            "--sort-by",
            choices=["relevance", "updated", "stars", "latest_commit"],
            default="relevance",
            help="Final sort for output rows",
        )
        p.add_argument(
            "--allow-duplicate-repos",
            action="store_true",
            help="Keep multiple posts for the same owner/repo in results",
        )

    p_sync = sub.add_parser("sync", help="Sync local append-oriented cache")
    p_sync.add_argument(
        "--cache-dir",
        default=default_cache_dir,
        help="Directory for cache/state/index files",
    )
    p_sync.add_argument("--feed-url", default=DEFAULT_FEED_URL)
    p_sync.add_argument("--search-index-url", default=DEFAULT_SEARCH_INDEX_URL)
    p_sync.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p_sync.add_argument("--bootstrap-limit", type=int, default=0, help="Initial load cap from search-index (0=all)")
    p_sync.add_argument("--feed-limit", type=int, default=0, help="Incremental feed cap (0=all)")
    p_sync.add_argument("--force-full", action="store_true", help="Ignore incremental marker and rebuild from full sources")

    p_latest = sub.add_parser("latest", help="List latest repositories")
    add_cache_arg(p_latest)
    p_latest.add_argument("--limit", type=int, default=20)
    p_latest.add_argument("--since", default="", help="ISO timestamp filter on updated_at")
    p_latest.set_defaults(sort_by="updated")

    p_grep = sub.add_parser("grep", help="Regex search over fields")
    add_cache_arg(p_grep)
    p_grep.add_argument("--pattern", required=True)
    p_grep.add_argument("--limit", type=int, default=20)
    p_grep.add_argument("--case-sensitive", action="store_true")
    p_grep.add_argument(
        "--fields",
        nargs="*",
        default=["title", "summary", "owner", "repo", "repo_url", "search_text"],
        help="Fields to scan",
    )
    p_grep.set_defaults(sort_by="updated")

    p_sem = sub.add_parser("semantic", help="Semantic search")
    add_cache_arg(p_sem)
    p_sem.add_argument("--query", required=True)
    p_sem.add_argument("--limit", type=int, default=20)
    p_sem.add_argument("--recency-days", type=int, default=-1, help="Filter results older than this many days")
    p_sem.set_defaults(sort_by="relevance")

    p_hybrid = sub.add_parser("hybrid", help="Hybrid semantic + keyword + optional regex")
    add_cache_arg(p_hybrid)
    p_hybrid.add_argument("--query", required=True)
    p_hybrid.add_argument("--pattern", default="", help="Optional regex hard-filter/boost")
    p_hybrid.add_argument("--limit", type=int, default=20)
    p_hybrid.set_defaults(sort_by="relevance")

    return parser


def sync_cache(
    *,
    cache_dir: Path,
    feed_url: str,
    search_index_url: str,
    base_url: str,
    bootstrap_limit: int,
    feed_limit: int,
    force_full: bool,
) -> dict[str, Any]:
    posts_path = cache_dir / CACHE_FILE
    state_path = cache_dir / STATE_FILE
    sem_path = cache_dir / SEM_FILE

    state = _load_state(state_path)
    existing = load_posts(posts_path)

    by_url: dict[str, RepoPost] = {p.post_url: p for p in existing}
    before_count = len(by_url)

    latest_known = "" if force_full else str(state.get("latest_post_url") or "")

    bootstrapped = False
    added_from_index = 0
    added_from_feed = 0

    if force_full or not by_url:
        idx_payload = fetch_json(search_index_url)
        if not isinstance(idx_payload, list):
            raise RuntimeError("search-index payload must be a list")
        items = idx_payload if bootstrap_limit <= 0 else idx_payload[:bootstrap_limit]
        parsed = [from_search_index_item(x, base_url=base_url) for x in items]
        for post in parsed:
            if post is None:
                continue
            if post.post_url not in by_url:
                added_from_index += 1
            by_url[post.post_url] = post
        bootstrapped = True

    feed_xml = fetch_text(feed_url)
    feed_rows = parse_feed_entries(
        feed_xml,
        stop_after_post_url=latest_known,
        limit=feed_limit,
    )

    for post in feed_rows:
        if post.post_url not in by_url:
            added_from_feed += 1
        by_url[post.post_url] = post

    ordered = sorted(by_url.values(), key=lambda p: (_ts_key(p.updated_at), p.post_url), reverse=True)
    save_posts(posts_path, ordered)

    if sem_path.exists():
        sem_path.unlink()

    now = utc_now_iso()
    state.update(
        {
            "last_sync_at": now,
            "latest_post_url": ordered[0].post_url if ordered else "",
            "post_count": len(ordered),
            "added_from_index": added_from_index,
            "added_from_feed": added_from_feed,
            "full_rebuild": bool(force_full),
            "bootstrapped": bootstrapped,
            "feed_url": feed_url,
            "search_index_url": search_index_url,
            "base_url": base_url,
        }
    )
    _save_state(state_path, state)

    return {
        "synced": True,
        "last_sync_at": now,
        "post_count_before": before_count,
        "post_count_after": len(ordered),
        "new_or_updated": max(len(ordered) - before_count, 0),
        "added_from_index": added_from_index,
        "added_from_feed": added_from_feed,
        "feed_entries_processed": len(feed_rows),
        "latest_post_url": state["latest_post_url"],
        "cache_file": str(posts_path),
        "state_file": str(state_path),
    }


def maybe_auto_sync_cache(
    *,
    cache_dir: Path,
    max_age_minutes: int,
    bootstrap_limit: int,
    feed_limit: int,
) -> dict[str, Any]:
    posts_path = cache_dir / CACHE_FILE
    state_path = cache_dir / STATE_FILE
    state = _load_state(state_path)

    missing_cache = not posts_path.exists()
    last_sync_at = str(state.get("last_sync_at") or "")
    last_sync_ts = _ts_key(last_sync_at)
    now_ts = datetime.now(tz=timezone.utc).timestamp()
    stale = (last_sync_ts <= 0) or ((now_ts - last_sync_ts) > (max_age_minutes * 60.0))

    if not missing_cache and not stale:
        return {
            "triggered": False,
            "reason": "fresh-cache",
            "last_sync_at": last_sync_at,
            "max_age_minutes": max_age_minutes,
        }

    reason = "missing-cache" if missing_cache else "stale-cache"
    sync_result = sync_cache(
        cache_dir=cache_dir,
        feed_url=str(state.get("feed_url") or DEFAULT_FEED_URL),
        search_index_url=str(state.get("search_index_url") or DEFAULT_SEARCH_INDEX_URL),
        base_url=str(state.get("base_url") or DEFAULT_BASE_URL),
        bootstrap_limit=bootstrap_limit,
        feed_limit=feed_limit,
        force_full=False,
    )

    return {
        "triggered": True,
        "reason": reason,
        "max_age_minutes": max_age_minutes,
        "sync": sync_result,
    }


def latest_posts(posts: list[RepoPost], *, limit: int, since: str) -> list[dict[str, Any]]:
    rows = posts
    if since:
        rows = [p for p in rows if _ts_key(p.updated_at) >= _ts_key(since)]
    return [post_to_dict(p) for p in rows[: max(1, limit)]]


def grep_search(
    posts: list[RepoPost],
    *,
    pattern: str,
    limit: int,
    case_sensitive: bool,
    fields: list[str],
) -> list[dict[str, Any]]:
    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(pattern, flags)

    out: list[dict[str, Any]] = []
    for post in posts:
        haystack = "\n".join(getattr(post, f, "") for f in fields)
        match = regex.search(haystack)
        if not match:
            continue
        row = post_to_dict(post)
        row["match"] = {
            "pattern": pattern,
            "snippet": snippet(haystack, match.start(), match.end()),
            "fields": fields,
        }
        out.append(row)
        if len(out) >= max(1, limit):
            break

    return out


def semantic_search(
    posts: list[RepoPost],
    *,
    cache_dir: Path,
    query: str,
    limit: int,
    recency_days: int,
) -> list[dict[str, Any]]:
    index = get_semantic_index(posts, cache_dir / SEM_FILE)
    sem_hits = index_search(index, query=query, limit=max(50, limit * 4))

    by_url = {u: s for u, s in sem_hits}
    by_post = {p.post_url: p for p in posts}

    scored: list[tuple[float, RepoPost, str]] = []
    for url, sem in sem_hits:
        post = by_post.get(url)
        if post is None:
            continue
        rec = recency_score(post.updated_at, half_life_days=180)
        age = age_days(post.updated_at)
        if recency_days >= 0 and age is not None and age > recency_days:
            continue
        score = (0.85 * sem) + (0.15 * rec)
        reason = f"semantic={sem:.3f}, recency={rec:.3f}"
        scored.append((score, post, reason))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: list[dict[str, Any]] = []
    for score, post, reason in scored[: max(1, limit)]:
        row = post_to_dict(post)
        row["score"] = round(score, 6)
        row["reason"] = reason
        out.append(row)

    return out


def hybrid_search(
    posts: list[RepoPost],
    *,
    cache_dir: Path,
    query: str,
    pattern: str,
    limit: int,
) -> list[dict[str, Any]]:
    index = get_semantic_index(posts, cache_dir / SEM_FILE)

    sem_hits = index_search(index, query=query, limit=max(80, limit * 6))
    sem_map = {u: s for u, s in sem_hits}

    q_tokens = tokenize(query)

    kw_scores: list[tuple[float, str]] = []
    for post in posts:
        score = keyword_score(q_tokens, post.search_text)
        if score > 0:
            kw_scores.append((score, post.post_url))
    kw_scores.sort(reverse=True)
    kw_map = {u: s for s, u in kw_scores[: max(80, limit * 6)]}

    regex = re.compile(pattern, re.IGNORECASE) if pattern else None

    by_post = {p.post_url: p for p in posts}
    candidates = set(sem_map) | set(kw_map)

    scored: list[tuple[float, RepoPost, str]] = []
    for url in candidates:
        post = by_post.get(url)
        if post is None:
            continue

        sem = sem_map.get(url, 0.0)
        kw = kw_map.get(url, 0.0)
        overlap = token_overlap(q_tokens, post.search_text)
        rec = recency_score(post.updated_at, half_life_days=120)

        score = (0.6 * sem) + (0.3 * kw) + (0.05 * overlap) + (0.05 * rec)
        reasons = [f"semantic={sem:.3f}", f"keyword={kw:.3f}", f"overlap={overlap:.3f}", f"recency={rec:.3f}"]

        if regex:
            if regex.search(post.search_text):
                score += 0.2
                reasons.append("grep-boost")
            else:
                continue

        scored.append((score, post, ", ".join(reasons)))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: list[dict[str, Any]] = []
    for score, post, reason in scored[: max(1, limit)]:
        row = post_to_dict(post)
        row["score"] = round(score, 6)
        row["reason"] = reason
        out.append(row)

    return out


def fetch_text(url: str, timeout: int = 30) -> str:
    ua = "repo-posts-discovery-skill/1.0"
    try:
        result = subprocess.run(
            ["curl", "-fsSL", "--max-time", str(timeout), "-A", ua, url],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except Exception:
        req = Request(url, headers={"User-Agent": ua, "Accept": "*/*"})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")


def fetch_json(url: str, timeout: int = 30) -> Any:
    return json.loads(fetch_text(url, timeout=timeout))


def parse_feed_entries(xml_text: str, *, stop_after_post_url: str, limit: int) -> list[RepoPost]:
    root = ET.fromstring(xml_text)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "media": "http://search.yahoo.com/mrss/",
    }

    out: list[RepoPost] = []
    for entry in root.findall("atom:entry", ns):
        post = from_feed_entry(entry, ns)
        if post is None:
            continue
        if stop_after_post_url and post.post_url == stop_after_post_url:
            break
        out.append(post)
        if limit > 0 and len(out) >= limit:
            break

    return out


def from_search_index_item(item: dict[str, Any], *, base_url: str) -> RepoPost | None:
    rel_url = str(item.get("u") or "").strip()
    if not rel_url:
        return None

    title = str(item.get("title") or "").strip()
    summary = str(item.get("s") or "").strip()
    search_text = str(item.get("t") or "").strip() or " ".join(x for x in [title, summary] if x)
    published_date = str(item.get("d") or "").strip()

    post_url = absolute_url(rel_url, base_url)
    image_url = absolute_url(str(item.get("img") or ""), base_url)

    repo_url = ""
    owner = ""
    repo = ""

    md_link = MARKDOWN_LINK_RE.search(title) or MARKDOWN_LINK_RE.search(search_text)
    if md_link:
        repo_url = md_link.group(2)
        owner, repo = owner_repo(repo_url)
        if md_link.group(1).strip():
            title = md_link.group(1).strip()

    if not repo_url:
        gh = GITHUB_LINK_RE.search(search_text)
        if gh:
            owner, repo = gh.group(1), gh.group(2)
            repo_url = f"https://github.com/{owner}/{repo}"

    if not title:
        title = f"{owner}/{repo}" if owner and repo else post_url.rsplit("/", 1)[-1]

    searchable = " ".join(x for x in [title, summary, owner, repo, repo_url, search_text] if x)

    return RepoPost(
        post_url=post_url,
        repo_url=repo_url,
        owner=owner,
        repo=repo,
        title=title,
        summary=summary,
        updated_at=to_iso(published_date),
        published_date=published_date,
        image_url=image_url,
        search_text=searchable,
        source="search-index",
    )


def from_feed_entry(entry: ET.Element, ns: dict[str, str]) -> RepoPost | None:
    post_url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
    if not post_url:
        link_el = entry.find("atom:link", ns)
        post_url = (link_el.attrib.get("href") if link_el is not None else "") or ""
        post_url = post_url.strip()
    if not post_url:
        return None

    updated_at = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
    title_raw = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()

    repo_url = ""
    for link in entry.findall("atom:link", ns):
        if link.attrib.get("rel") == "related":
            href = (link.attrib.get("href") or "").strip()
            if "github.com" in href:
                repo_url = href
                break

    content_html = (entry.findtext("atom:content", default="", namespaces=ns) or "").strip()
    if not repo_url:
        gh = GITHUB_LINK_RE.search(content_html)
        if gh:
            repo_url = f"https://github.com/{gh.group(1)}/{gh.group(2)}"

    owner, repo = owner_repo(repo_url)
    title = f"{owner}/{repo}" if owner and repo else title_raw
    summary = extract_summary(content_html)

    image_url = ""
    thumb = entry.find("media:thumbnail", ns)
    if thumb is not None:
        image_url = (thumb.attrib.get("url") or "").strip()
    if not image_url:
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("rel") == "enclosure" and link.attrib.get("type", "").startswith("image/"):
                image_url = (link.attrib.get("href") or "").strip()
                break

    published_date = date_from_post_url(post_url)
    searchable = " ".join(x for x in [title, summary, owner, repo, repo_url, title_raw] if x)

    return RepoPost(
        post_url=post_url,
        repo_url=repo_url,
        owner=owner,
        repo=repo,
        title=title,
        summary=summary,
        updated_at=updated_at or to_iso(published_date),
        published_date=published_date,
        image_url=image_url,
        search_text=searchable,
        source="feed",
    )


def load_posts(path: Path) -> list[RepoPost]:
    if not path.exists():
        return []
    rows: list[RepoPost] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(RepoPost(**obj))
    return rows


def save_posts(path: Path, posts: list[RepoPost]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for post in posts:
            fh.write(json.dumps(asdict(post), ensure_ascii=False) + "\n")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_semantic_index(posts: list[RepoPost], sem_path: Path, *, dim: int = 192, projections: int = 4) -> SemanticArtifact:
    signature = semantic_signature(posts, dim=dim, projections=projections)
    if sem_path.exists():
        try:
            artifact = pickle.loads(sem_path.read_bytes())
            if isinstance(artifact, SemanticArtifact) and artifact.signature == signature:
                return artifact
        except Exception:
            pass

    artifact = build_index(posts, signature=signature, dim=dim, projections=projections)
    sem_path.write_bytes(pickle.dumps(artifact))
    return artifact


def semantic_signature(posts: list[RepoPost], *, dim: int, projections: int) -> str:
    first = posts[0].post_url if posts else ""
    last = posts[-1].post_url if posts else ""
    latest = posts[0].updated_at if posts else ""
    raw = f"count={len(posts)}|first={first}|last={last}|latest={latest}|d={dim}|p={projections}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_index(posts: list[RepoPost], *, signature: str, dim: int, projections: int) -> SemanticArtifact:
    token_docs = [tokenize(p.search_text) for p in posts]
    df: Counter[str] = Counter()
    for toks in token_docs:
        for tok in set(toks):
            df[tok] += 1

    n_docs = max(len(posts), 1)
    idf = {tok: math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0 for tok, freq in df.items()}

    matrix = array("f")
    for toks in token_docs:
        vec = vector_from_tokens(toks, idf=idf, dim=dim, projections=projections)
        matrix.extend(vec)

    return SemanticArtifact(
        signature=signature,
        post_urls=[p.post_url for p in posts],
        idf=idf,
        dim=dim,
        projections_per_token=projections,
        matrix=matrix,
    )


def index_search(artifact: SemanticArtifact, *, query: str, limit: int) -> list[tuple[str, float]]:
    tokens = tokenize(query)
    if not tokens:
        return []

    q_vec = vector_from_tokens(
        tokens,
        idf=artifact.idf,
        dim=artifact.dim,
        projections=artifact.projections_per_token,
    )

    top: list[tuple[float, str]] = []
    for idx, url in enumerate(artifact.post_urls):
        start = idx * artifact.dim
        score = 0.0
        for j in range(artifact.dim):
            score += artifact.matrix[start + j] * q_vec[j]
        if score <= 0:
            continue
        if len(top) < limit:
            heapq.heappush(top, (score, url))
        else:
            heapq.heappushpop(top, (score, url))

    top.sort(reverse=True)
    return [(url, float(score)) for score, url in top]


def tokenize(text: str) -> list[str]:
    raw = TOKEN_RE.findall((text or "").lower())
    out: list[str] = []
    for token in raw:
        token = normalize_token(token)
        if len(token) >= 2:
            out.append(token)
    return out


def normalize_token(token: str) -> str:
    if token in SYNONYM_CANONICAL:
        return SYNONYM_CANONICAL[token]

    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("es") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 3:
        token = token[:-1]

    return SYNONYM_CANONICAL.get(token, token)


def vector_from_tokens(tokens: list[str], *, idf: dict[str, float], dim: int, projections: int) -> list[float]:
    if not tokens:
        return [0.0] * dim

    tf = Counter(tokens)
    vec = [0.0] * dim

    for token, count in tf.items():
        weight = (1.0 + math.log(count)) * idf.get(token, 1.0)
        for i in range(projections):
            digest = hashlib.blake2b(f"{token}:{i}".encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += weight * sign

    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0:
        return vec
    inv = 1.0 / norm
    return [x * inv for x in vec]


def keyword_score(tokens: list[str], text: str) -> float:
    if not tokens:
        return 0.0
    lower = text.lower()
    score = 0.0
    for token in tokens:
        cnt = lower.count(token)
        if cnt:
            score += 1.0 + math.log(cnt + 1.0)
    return score / len(tokens)


def token_overlap(tokens: list[str], text: str) -> float:
    if not tokens:
        return 0.0
    lower = text.lower()
    found = sum(1 for t in tokens if t in lower)
    return found / len(tokens)


def post_to_dict(post: RepoPost) -> dict[str, Any]:
    return {
        "post_url": post.post_url,
        "repo_url": post.repo_url,
        "owner": post.owner,
        "repo": post.repo,
        "title": post.title,
        "summary": post.summary,
        "updated_at": post.updated_at,
        "published_date": post.published_date,
        "image_url": post.image_url,
    }


def absolute_url(maybe_relative: str, base_url: str) -> str:
    if not maybe_relative:
        return ""
    if maybe_relative.startswith("http://") or maybe_relative.startswith("https://"):
        return maybe_relative
    return f"{base_url.rstrip('/')}/{maybe_relative.lstrip('/')}"


def owner_repo(repo_url: str) -> tuple[str, str]:
    m = GITHUB_LINK_RE.search(repo_url or "")
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def date_from_post_url(post_url: str) -> str:
    path = urlparse(post_url).path
    parts = [p for p in path.split("/") if p]
    for i in range(len(parts) - 3):
        y, m, d = parts[i : i + 3]
        if len(y) == 4 and y.isdigit() and len(m) == 2 and m.isdigit() and len(d) == 2 and d.isdigit():
            return f"{y}-{m}-{d}"
    return ""


def extract_summary(content_html: str) -> str:
    paragraphs = re.findall(r"<p>(.*?)</p>", content_html or "", flags=re.IGNORECASE | re.DOTALL)
    for para in paragraphs:
        text = strip_html(para)
        if text and "screenshot" not in text.lower():
            return text
    return ""


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def to_iso(date_str: str) -> str:
    if not date_str:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        return f"{date_str}T00:00:00+00:00"
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return date_str


def parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _ts_key(value: str) -> float:
    dt = parse_iso(value)
    if dt is None:
        return 0.0
    return dt.timestamp()


def age_days(value: str) -> float | None:
    dt = parse_iso(value)
    if dt is None:
        return None
    delta = datetime.now(tz=timezone.utc) - dt.astimezone(timezone.utc)
    return max(0.0, delta.total_seconds() / 86400.0)


def recency_score(value: str, *, half_life_days: float) -> float:
    age = age_days(value)
    if age is None:
        return 0.0
    return math.exp(-math.log(2) * (age / max(half_life_days, 1.0)))


def snippet(text: str, start: int, end: int, window: int = 60) -> str:
    left = max(start - window, 0)
    right = min(end + window, len(text))
    out = text[left:right].replace("\n", " ")
    if left > 0:
        out = "..." + out
    if right < len(text):
        out = out + "..."
    return out


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def finalize_rows(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    cache_dir: Path,
    default_sort: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    github_meta: dict[str, Any] = {"enabled": False, "enriched": 0, "attempted": 0, "errors": []}

    if not args.allow_duplicate_repos:
        rows = dedupe_rows(rows, mode=default_sort)

    if args.with_github:
        rows, github_meta = enrich_with_github(
            rows,
            cache_dir=cache_dir,
            repo_limit=max(0, int(args.github_limit)),
            refresh_hours=max(0.0, float(args.github_refresh_hours)),
        )

    sort_by = (args.sort_by or default_sort).strip().lower()
    rows = sort_rows(rows, sort_by=sort_by)
    return rows, github_meta


def enrich_with_github(
    rows: list[dict[str, Any]],
    *,
    cache_dir: Path,
    repo_limit: int,
    refresh_hours: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_path = cache_dir / GITHUB_CACHE_FILE
    cache = _load_state(cache_path)
    repos: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        owner = str(row.get("owner") or "").strip()
        repo = str(row.get("repo") or "").strip()
        if not owner or not repo:
            continue
        key = (owner, repo)
        if key in seen:
            continue
        seen.add(key)
        repos.append(key)

    repos = repos[:repo_limit] if repo_limit > 0 else []
    attempted = len(repos)
    enriched = 0
    errors: list[str] = []
    refresh_seconds = refresh_hours * 3600.0

    for owner, repo in repos:
        repo_key = f"{owner}/{repo}"
        cached = cache.get(repo_key, {})
        fetched_at = _ts_key(str(cached.get("fetched_at") or ""))
        fresh = fetched_at > 0 and (datetime.now(tz=timezone.utc).timestamp() - fetched_at) <= refresh_seconds
        data = cached.get("data") if fresh else None

        if data is None:
            data, err = fetch_github_repo(owner, repo)
            if err:
                errors.append(f"{repo_key}: {err}")
                continue
            cache[repo_key] = {
                "fetched_at": utc_now_iso(),
                "data": data,
            }

        stars = data.get("stargazers_count")
        pushed_at = data.get("pushed_at")
        updated_at = data.get("updated_at")
        default_branch = data.get("default_branch")
        html_url = data.get("html_url")

        for row in rows:
            if row.get("owner") == owner and row.get("repo") == repo:
                row["github_stars"] = stars
                row["github_latest_commit_date"] = pushed_at
                row["github_repo_updated_at"] = updated_at
                row["github_default_branch"] = default_branch
                row["github_repo_url"] = html_url or row.get("repo_url")
                enriched += 1

    _save_state(cache_path, cache)

    return rows, {
        "enabled": True,
        "attempted": attempted,
        "enriched": enriched,
        "errors": errors[:10],
        "cache_file": str(cache_path),
    }


def fetch_github_repo(owner: str, repo: str) -> tuple[dict[str, Any] | None, str | None]:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    token = (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "repo-posts-discovery-skill/1.1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace")), None
    except HTTPError as e:
        return None, f"HTTP {e.code}"
    except URLError as e:
        return None, str(e.reason)
    except Exception as e:  # pragma: no cover
        return None, str(e)


def sort_rows(rows: list[dict[str, Any]], *, sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "stars":
        return sorted(
            rows,
            key=lambda r: (
                int(r.get("github_stars") or -1),
                float(r.get("score") or -1.0),
                _ts_key(str(r.get("github_latest_commit_date") or r.get("updated_at") or "")),
            ),
            reverse=True,
        )
    if sort_by == "latest_commit":
        return sorted(
            rows,
            key=lambda r: (
                _ts_key(str(r.get("github_latest_commit_date") or "")),
                int(r.get("github_stars") or -1),
                float(r.get("score") or -1.0),
            ),
            reverse=True,
        )
    if sort_by == "updated":
        return sorted(
            rows,
            key=lambda r: (
                _ts_key(str(r.get("updated_at") or "")),
                float(r.get("score") or -1.0),
            ),
            reverse=True,
        )
    # relevance
    return sorted(
        rows,
        key=lambda r: (
            float(r.get("score") or -1.0),
            _ts_key(str(r.get("updated_at") or "")),
        ),
        reverse=True,
    )


def dedupe_rows(rows: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    best_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _repo_key(row)
        if not key:
            key = str(row.get("post_url") or "")
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = row
            continue
        if _prefer_row(row, existing, mode=mode):
            best_by_key[key] = row
    return list(best_by_key.values())


def _repo_key(row: dict[str, Any]) -> str:
    owner = str(row.get("owner") or "").strip().lower()
    repo = str(row.get("repo") or "").strip().lower()
    if owner and repo:
        return f"{owner}/{repo}"
    repo_url = str(row.get("repo_url") or "").strip().lower()
    return repo_url


def _prefer_row(candidate: dict[str, Any], current: dict[str, Any], *, mode: str) -> bool:
    if mode == "updated":
        return _ts_key(str(candidate.get("updated_at") or "")) > _ts_key(str(current.get("updated_at") or ""))
    c_score = float(candidate.get("score") or -1.0)
    x_score = float(current.get("score") or -1.0)
    if c_score != x_score:
        return c_score > x_score
    return _ts_key(str(candidate.get("updated_at") or "")) > _ts_key(str(current.get("updated_at") or ""))


def _print_json(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
