# repo-post-skill-mcp

This workspace now focuses on a **skill-first** implementation for `repo_posts` discovery.

## Main deliverable

- Skill: `/Users/arpit/Dev/repo-post-skill-mcp/skills/repo-posts-discovery`
- Core script: `/Users/arpit/Dev/repo-post-skill-mcp/skills/repo-posts-discovery/scripts/repo_posts_tool.py`

The script supports:
- append-oriented sync from `search-index.json` + `feed.xml`
- latest listing
- regex/grep filtering
- semantic search
- hybrid ranking

## Quick start

```bash
cd /Users/arpit/Dev/repo-post-skill-mcp/skills/repo-posts-discovery
scripts/repo_posts_tool.py sync --bootstrap-limit 1200 --feed-limit 300
scripts/repo_posts_tool.py hybrid --query 'open source ai agent framework' --limit 10
```

## Data files

Generated under skill `data/`:
- `posts.jsonl`
- `state.json`
- `semantic_index.pkl`
