"""Plain-language analysis agent — bring your own model (Claude via API key, or a local
OpenAI-compatible model), drive the SMILE MSI engine through the same tools the MCP server
exposes, create new tools and flows on the fly, and keep a complete audit log.

Run it with ``smile-msi chat`` (or ``python -m smile_msi.agent``). Modules:

* :mod:`.tools`     — tool registry (built-ins from :mod:`smile_msi.mcpserver`) + result rendering
* :mod:`.custom`    — user-created, versioned tools and saved flows
* :mod:`.providers` — Claude (official ``anthropic`` SDK) and OpenAI-compatible local servers
* :mod:`.core`      — the agent loop (approvals, events, logging)
* :mod:`.log`       — the append-only session log and its Markdown report
* :mod:`.server`    — the local browser chat
"""
