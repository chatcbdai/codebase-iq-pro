#!/bin/bash

# Add all MCP servers from Claude Desktop to Claude Code
# Use --scope user to make them available in all Claude Code sessions

echo "Adding MCP servers to Claude Code..."

# Everything server
claude mcp add --scope user everything \
  /opt/homebrew/bin/npx -- -y @modelcontextprotocol/server-everything

# Brave Search
claude mcp add --scope user brave-search \
  /opt/homebrew/bin/npx -- -y @modelcontextprotocol/server-brave-search \
  --env BRAVE_API_KEY=BSAjp4akiPCJrf1M8KpTvvP1XcVz37_

# Everart
claude mcp add --scope user everart \
  /opt/homebrew/bin/npx -- -y @modelcontextprotocol/server-everart \
  --env EVERART_API_KEY=everart-t04gasfXPhaG5f0Sp43POU1ruCKitMps3GZBQErGWCw

# Memory
claude mcp add --scope user memory \
  /opt/homebrew/bin/npx -- -y @modelcontextprotocol/server-memory

# Puppeteer
claude mcp add --scope user puppeteer \
  /opt/homebrew/bin/npx -- -y @modelcontextprotocol/server-puppeteer

# Sequential Thinking
claude mcp add --scope user sequential-thinking \
  /opt/homebrew/bin/npx -- -y @modelcontextprotocol/server-sequential-thinking

# Filesystem
claude mcp add --scope user filesystem \
  /opt/homebrew/bin/npx -- -y @modelcontextprotocol/server-filesystem /Users/chrisryviss

# GitHub
claude mcp add --scope user github \
  /opt/homebrew/bin/npx -- -y @modelcontextprotocol/server-github

# Playwright
claude mcp add --scope user playwright \
  /opt/homebrew/bin/npx -- -y github:executeautomation/mcp-playwright \
  --env PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

# Postgres
claude mcp add --scope user postgres \
  /opt/homebrew/bin/npx -- -y @modelcontextprotocol/server-postgres postgresql://localhost/mydb

# SQLite
claude mcp add --scope user sqlite \
  /opt/homebrew/bin/uvx -- mcp-server-sqlite --db-path /Users/chrisryviss/databases/example.db

# Slack (with placeholder tokens)
claude mcp add --scope user slack \
  /opt/homebrew/bin/npx -- -y @modelcontextprotocol/server-slack \
  --env SLACK_BOT_TOKEN=xoxb-your-token \
  --env SLACK_TEAM_ID=your-team-id

echo "Done! All MCP servers have been added to Claude Code."
echo "List servers with: claude mcp list"