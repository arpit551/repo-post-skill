# repo-post-skill

Discover high-signal open-source repositories from `repo_posts` so teams can reuse proven patterns instead of starting from scratch.

## Install (Claude Code)

Use your published marketplace repo:

```text
/plugin marketplace add <owner>/<repo>
/plugin install repo-post-skill@repo-posts-marketplace
```

Local development install:

```text
/plugin marketplace add ./repo-post-skill-mcp
/plugin install repo-post-skill@repo-posts-marketplace
```

## Install (Codex)

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R ./skills/repo-posts-discovery "${CODEX_HOME:-$HOME/.codex}/skills/"
```

Then prompt:
- "Use repo-posts-discovery to find top open-source repos for AI agents."
- "Use repo-posts-discovery and return 10 repos sorted by stars and latest commit."

## Install (Other Agents/CLI)

```bash
cd ./skills/repo-posts-discovery
scripts/repo_posts_tool.py hybrid --query "your query" --limit 20 --with-github --sort-by stars
```

## Why this exists

Most teams waste time re-building solutions that already exist in OSS.  
This skill makes reference-first development easy:

- find credible repos quickly
- compare approaches before coding
- narrow by domain or constraints
- rank results using activity and popularity signals

## Typical use cases

- "We need references before building this feature."
- "Show production-grade repos for this stack."
- "Find niche projects for a specific market/domain."
- "Create a shortlist for architecture review."

## Prompt examples

- "Use repo-post-skill and find 10 open-source AI agent frameworks sorted by stars and latest commit."
- "Use repo-post-skill and find self-hosted security tooling with clear maintenance signals."
- "Use repo-post-skill and check if India stock market AI repos exist with strict India terms."
- "Use repo-post-skill and give alternatives to an existing repo approach."

## What the output includes

- `owner/repo`
- summary
- why it matched
- stars
- latest commit date
- repo link
- source post link

## Recommended workflow

1. Start broad and ask for top 10.
2. Add strict constraints (language, market, domain, deployment).
3. Re-rank by stars and latest commit.
4. Pick 2-3 repos as implementation references.

## Included plugin files

- `.claude-plugin/plugin.json`
- `.claude-plugin/marketplace.json`
