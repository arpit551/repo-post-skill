---
name: repo-posts-discovery
description: Discover high-signal open-source repositories from tom-doerr/repo_posts using local skill scripts with append-oriented syncing, semantic search, regex/grep filtering, and hybrid ranking. Use when users ask for latest repositories, curated repo recommendations, niche OSS discovery, or targeted searches like "find MCP repos" and "grep for self-hosted security tools".
---

# Repo Posts Discovery

## Overview

Use this skill to fetch and search `repo_posts` data without any MCP server dependency.

Run the script at `scripts/repo_posts_tool.py` for all operations:
- sync cache (append-oriented)
- list latest repos
- regex/grep search
- semantic search
- hybrid search
- optional GitHub enrichment (stars + latest commit date) with sortable output

## Workflow

1. Sync first when freshness matters.
2. Run hybrid search for broad discovery prompts.
3. Run semantic search for conceptual intent.
4. Run grep search for hard constraints.
5. Return 5-10 repositories with direct links and match rationale.

## Commands

Use these commands from the skill directory:

```bash
cd /Users/arpit/Dev/repo-post-skill-mcp/skills/repo-posts-discovery
```

### Sync (append-oriented)

```bash
scripts/repo_posts_tool.py sync
```

Fast limited sync:

```bash
scripts/repo_posts_tool.py sync --bootstrap-limit 1200 --feed-limit 300
```

Full rebuild:

```bash
scripts/repo_posts_tool.py sync --force-full
```

### Latest

```bash
scripts/repo_posts_tool.py latest --limit 15
```

With GitHub enrichment and star sorting:

```bash
scripts/repo_posts_tool.py latest --limit 15 --with-github --sort-by stars
```

### Grep / regex

```bash
scripts/repo_posts_tool.py grep --pattern 'mcp|model context protocol' --limit 20
```

```bash
scripts/repo_posts_tool.py grep --pattern 'self-hosted|on-prem|docker' --limit 20
```

### Semantic

```bash
scripts/repo_posts_tool.py semantic --query 'open source ai agent framework' --limit 15
```

### Hybrid

```bash
scripts/repo_posts_tool.py hybrid --query 'local ai coding agent tools' --pattern 'github' --limit 15
```

Rank by stars with latest commit metadata:

```bash
scripts/repo_posts_tool.py hybrid --query 'India stock market analysis using AI' --limit 20 --with-github --sort-by stars
```

## Response Format

For each recommended repo, include:
- `owner/repo`
- 1-line summary
- why it matched (`semantic`, `keyword`, `grep`, `recency`)
- `github_stars` (when `--with-github`)
- `github_latest_commit_date` (when `--with-github`)
- `repo_url`
- `post_url`

## Data Files

The script writes local artifacts in `data/`:
- `data/posts.jsonl`
- `data/state.json`
- `data/semantic_index.pkl`

## Quality Rules

- Prefer repositories with clear summary and direct GitHub link.
- Prefer newer entries unless the user asks for historical/niche results.
- If semantic results are weak, fallback to hybrid + grep constraints.
- If cache is missing, run sync before search.
- For strict region/market matching, use word boundaries in regex (for example: `\\b(india|nse|bse|nifty|sensex)\\b`).

For reusable query sets, load `references/query-playbook.md`.
