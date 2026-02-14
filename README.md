# repo-post-skill

A production-ready, **skill-first** repository discovery workflow for [tom-doerr/repo_posts](https://github.com/tom-doerr/repo_posts).

It is designed for Codex users who want high-signal open-source recommendations with:
- append-oriented sync
- semantic search
- regex/grep filtering
- hybrid ranking
- optional GitHub enrichment (stars and latest commit date)

No MCP server required.

## What this solves

`repo_posts` is a great source, but manual browsing is noisy and slow. This skill gives an agentic workflow that can:
- keep a local cache fresh
- search conceptually (semantic)
- enforce strict constraints (regex with word boundaries)
- rank and return top repositories for user-facing recommendations
- optionally enrich and sort by stars/latest commit

## Repository layout

```text
repo-post-skill-mcp/
├── README.md
└── skills/
    └── repo-posts-discovery/
        ├── SKILL.md
        ├── agents/openai.yaml
        ├── scripts/repo_posts_tool.py
        ├── references/query-playbook.md
        └── data/
            └── .gitkeep
```

## Core components

- Skill definition: `skills/repo-posts-discovery/SKILL.md`
- Tooling script: `skills/repo-posts-discovery/scripts/repo_posts_tool.py`
- Agent metadata: `skills/repo-posts-discovery/agents/openai.yaml`
- Query presets: `skills/repo-posts-discovery/references/query-playbook.md`

## Data flow

1. Bootstrap from `search-index.json` (bulk historical posts).
2. Incremental updates from `feed.xml` (newest posts).
3. Append-oriented merge keyed by `post_url` (no destructive overwrite logic).
4. Persist local artifacts in `data/`.
5. Build semantic index lazily on first semantic/hybrid query.

Primary data sources:
- `https://tom-doerr.github.io/repo_posts/assets/search-index.json`
- `https://tom-doerr.github.io/repo_posts/feed.xml`

## Freshness model (important)

Every read/search command (`latest`, `grep`, `semantic`, `hybrid`) can auto-sync before searching.

Default policy:
- `--auto-sync-max-age-minutes 1440` (1 day)
- if cache is missing, sync triggers
- if cache is older than 1 day, sync triggers
- otherwise, cached data is used

Controls:
- disable auto-sync: `--no-auto-sync`
- tune stale threshold: `--auto-sync-max-age-minutes <minutes>`
- tune auto-sync fetch sizes:
  - `--auto-sync-bootstrap-limit` (default `1200`)
  - `--auto-sync-feed-limit` (default `300`)

## Quick start

```bash
cd /Users/arpit/Dev/repo-post-skill-mcp/skills/repo-posts-discovery

# Initial cache warmup
scripts/repo_posts_tool.py sync --bootstrap-limit 1200 --feed-limit 300

# Hybrid discovery
scripts/repo_posts_tool.py hybrid --query 'open source ai agent framework' --limit 10
```

## Command reference

### 1) Sync cache

```bash
# Normal incremental sync
scripts/repo_posts_tool.py sync

# Fast bounded sync
scripts/repo_posts_tool.py sync --bootstrap-limit 1200 --feed-limit 300

# Full rebuild
scripts/repo_posts_tool.py sync --force-full
```

### 2) Latest repositories

```bash
scripts/repo_posts_tool.py latest --limit 20
scripts/repo_posts_tool.py latest --limit 20 --with-github --sort-by stars
```

### 3) Regex/grep search

```bash
scripts/repo_posts_tool.py grep --pattern 'mcp|model context protocol' --limit 20
scripts/repo_posts_tool.py grep --pattern '\b(india|indian|nse|bse|nifty|sensex)\b' --limit 30
```

### 4) Semantic search

```bash
scripts/repo_posts_tool.py semantic --query 'open source local ai coding agent' --limit 20
scripts/repo_posts_tool.py semantic --query 'self hosted security tools' --limit 20 --recency-days 365
```

### 5) Hybrid search (recommended default)

```bash
scripts/repo_posts_tool.py hybrid --query 'India stock market analysis using AI' \
  --pattern '\b(india|indian|nse|bse|nifty|sensex)\b' \
  --limit 30 \
  --with-github \
  --sort-by stars
```

## Ranking and result quality

### Semantic mode

Score uses:
- semantic similarity
- recency weight

### Hybrid mode

Score combines:
- semantic similarity
- keyword score
- token overlap
- recency
- optional regex boost when `--pattern` matches

### Deduplication

By default, multiple posts for the same repo are deduped by `owner/repo`.

Use `--allow-duplicate-repos` to keep all matching posts.

## GitHub enrichment and sorting

Enable with `--with-github`:
- fetches repo metadata from GitHub API
- adds `github_stars`
- adds `github_latest_commit_date`
- adds `github_repo_updated_at`, `github_default_branch`

Sort options:
- `--sort-by relevance`
- `--sort-by updated`
- `--sort-by stars`
- `--sort-by latest_commit`

GitHub cache:
- stored at `data/github_repo_cache.json`
- refresh window controlled by `--github-refresh-hours` (default `24`)
- maximum API-enriched repos controlled by `--github-limit` (default `30`)

Auth (optional but recommended for higher rate limits):
- set `GITHUB_TOKEN` or `GH_TOKEN`

## Output format

All commands return JSON. Main shape:

```json
{
  "results": [
    {
      "owner": "example",
      "repo": "project",
      "summary": "short summary",
      "repo_url": "https://github.com/example/project",
      "post_url": "https://tom-doerr.github.io/repo_posts/....html",
      "updated_at": "2026-02-14T00:00:00+00:00",
      "score": 0.91,
      "reason": "semantic=..., keyword=..., overlap=..., recency=...",
      "github_stars": 1234,
      "github_latest_commit_date": "2026-02-10T08:12:30Z"
    }
  ],
  "count": 10,
  "github": { "...": "metadata" },
  "auto_sync": { "...": "metadata" }
}
```

## How to use this as a skill (not raw script-only)

In Codex, invoke the skill by asking for `repo-posts-discovery` tasks, for example:
- "Use repo-posts-discovery to find top open-source MCP repos."
- "Use repo-posts-discovery and show top 10 AI finance repos sorted by stars and latest commit."

Expected skill behavior:
- syncs when needed (or stale by policy)
- runs hybrid/semantic/grep based on query type
- returns concise recommendations first
- does not dump raw command logs unless explicitly asked

## Local artifacts

Generated under `skills/repo-posts-discovery/data/`:
- `posts.jsonl` (canonical cached posts)
- `state.json` (sync metadata and freshness state)
- `semantic_index.pkl` (local semantic index)
- `github_repo_cache.json` (GitHub enrichment cache)

## Skill-level test checklist

1. Run a fresh sync and confirm `post_count_after > 0`.
2. Run `hybrid` query with `--with-github --sort-by stars`.
3. Verify each result has valid `repo_url` and `post_url`.
4. Verify `github_stars` and `github_latest_commit_date` exist when enriched.
5. Re-run the same query and verify GitHub cache reuse (fewer API fetches).
6. Set `--auto-sync-max-age-minutes 1`, wait 1+ minute, run query, and verify `auto_sync.triggered=true`.

## Notes

- This repo is intentionally skill-first and lightweight.
- The dataset updates continuously; freshness is controlled via auto-sync policy.
- For strict regional/market detection, prefer word-boundary regex patterns to avoid false positives.
