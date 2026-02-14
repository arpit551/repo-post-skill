# repo-post-skill

Find proven open-source repos fast, so you can reuse patterns and avoid reinventing the wheel.

## Install

### Claude Code

```text
/plugin marketplace add https://github.com/arpit551/repo-post-skill
/plugin install repo-post-skill@arpit-repo-posts-marketplace
```

For local testing:

```text
/plugin marketplace add /Users/arpit/Dev/repo-post-skill-mcp
/plugin install repo-post-skill@arpit-repo-posts-marketplace
```

### Codex

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R /Users/arpit/Dev/repo-post-skill-mcp/skills/repo-posts-discovery "${CODEX_HOME:-$HOME/.codex}/skills/"
```

Then ask Codex naturally:
- "Use repo-posts-discovery to find top open-source repos for AI agents."
- "Use repo-posts-discovery and return 10 repos sorted by stars and latest commit."

### Other agents and tools

You can use the same workflow in Cursor, Windsurf, terminal agents, or CI by running:

```bash
cd /Users/arpit/Dev/repo-post-skill-mcp/skills/repo-posts-discovery
scripts/repo_posts_tool.py hybrid --query "your query" --limit 20 --with-github --sort-by stars
```

Detailed multi-platform guide: `docs/PLATFORM_GUIDE.md`

## Why use this skill

- Skip noisy browsing and get curated repo shortlists quickly.
- Discover references by intent (not just exact keywords).
- Add hard filters when needed (for domain, market, language, stack).
- Return ranked repos with useful proof signals like stars and activity.

## How teams use it

- Before building a new feature, find 5-10 reference repos first.
- Compare existing OSS approaches before finalizing architecture.
- Build “inspiration lists” for hackathons, prototypes, or production ideas.
- Run targeted discovery for niche asks (for example: AI finance, MCP tools, self-hosted security).

## Prompt examples

Use these directly after install:

- "Use repo-post-skill and find top open-source repos for AI agents. Sort by stars and latest commit."
- "Use repo-post-skill to find repos similar to Cursor/Codex workflows."
- "Use repo-post-skill to find India stock analysis repos using AI with strict India market terms."
- "Use repo-post-skill and give me 10 production-grade self-hosted security tools."

## What you get back

For each recommendation:

- `owner/repo`
- one-line summary
- reason it matched your query
- `github_stars`
- latest commit date
- repo URL
- source post URL

## Good prompt patterns

- Be clear on your intent: "production-ready", "beginner-friendly", "research-heavy", "framework".
- Add constraints: language, domain, deployment style, geography, or market.
- Ask for a ranked list with 5-10 items and short rationale per item.

## Example workflow

1. Ask for a broad shortlist.
2. Refine with constraints (language/domain/region).
3. Ask for final ranking by stars + latest commit.
4. Pick 2-3 repos to use as implementation references.
