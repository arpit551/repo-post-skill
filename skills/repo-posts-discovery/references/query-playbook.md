# Query Playbook

## Daily discovery flow

1. `scripts/repo_posts_tool.py sync`
2. `scripts/repo_posts_tool.py hybrid --query 'interesting developer tools' --limit 20`
3. Refine with grep if needed.

## Semantic-first prompts

- `scripts/repo_posts_tool.py semantic --query 'open source ai agent framework' --limit 20`
- `scripts/repo_posts_tool.py semantic --query 'self hosted observability stack' --limit 20`
- `scripts/repo_posts_tool.py semantic --query 'local first productivity apps' --limit 20`

## Hard filters (regex)

- MCP: `mcp|model context protocol`
- Security: `waf|threat|siem|malware|pentest`
- Language: `\brust\b|\bgo\b|\bpython\b`
- Deployment: `self-hosted|on-prem|docker|kubernetes`
- India market strict: `\b(india|indian|nse|bse|nifty|sensex)\b`

Example:

```bash
scripts/repo_posts_tool.py hybrid --query 'security tooling' --pattern 'waf|threat|siem' --limit 20
```

```bash
scripts/repo_posts_tool.py hybrid --query 'India stock market analysis using AI' --pattern '\b(india|indian|nse|bse|nifty|sensex)\b' --limit 30 --with-github --sort-by stars
```

## Latest-only sweep

```bash
scripts/repo_posts_tool.py latest --limit 25
```

If user asks "today", include exact date in output (for example `2026-02-14`) and mention timezone context.

## Troubleshooting

- Empty cache: run `scripts/repo_posts_tool.py sync`
- Weak semantic output: use hybrid mode + regex constraint
- Slow first semantic call: expected; index file is built at `data/semantic_index.pkl`
