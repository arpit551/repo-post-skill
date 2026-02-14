# Platform Guide

Use this guide to run `repo-post-skill` from different AI environments.

## Claude Code

Install:

```text
/plugin marketplace add https://github.com/arpit551/repo-post-skill
/plugin install repo-post-skill@arpit-repo-posts-marketplace
```

Example prompts:
- "Use repo-post-skill to find the best open-source MCP repos."
- "Use repo-post-skill and show 10 AI finance repos ranked by stars and latest commit."

## Codex

Install skill locally:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R /Users/arpit/Dev/repo-post-skill-mcp/skills/repo-posts-discovery "${CODEX_HOME:-$HOME/.codex}/skills/"
```

Example prompts:
- "Use repo-posts-discovery to find production-ready agent frameworks."
- "Use repo-posts-discovery and check whether India stock AI repos exist."

## Other agent environments

If your tool can run shell commands, use the script directly:

```bash
cd /Users/arpit/Dev/repo-post-skill-mcp/skills/repo-posts-discovery
scripts/repo_posts_tool.py hybrid --query "self-hosted security tools" --limit 20 --with-github --sort-by stars
```

Add strict constraints when needed:

```bash
scripts/repo_posts_tool.py hybrid --query "India stock market analysis using AI" \
  --pattern "\\b(india|indian|nse|bse|nifty|sensex)\\b" \
  --limit 30 \
  --with-github \
  --sort-by stars
```

## Best practices

- Start broad, then narrow with constraints.
- Ask for ranked output with stars + latest commit.
- Ask for top 5-10 repos with one-line rationale each.
