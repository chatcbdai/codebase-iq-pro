# Migration Guide: CodebaseIQ Pro v2.0

## Overview

Version 2.0 solves three critical issues:
1. **25K Token Limit**: MCP responses exceeding protocol limits
2. **Zero Knowledge Problem**: AI starting fresh each conversation
3. **Performance**: 4-5 minute analysis time

## What's Changed

### New Tools

1. **`get_codebase_context`** (PRIMARY TOOL)
   - Always use this FIRST in any conversation
   - Returns optimized data under 25K tokens
   - Loads from persistent cache instantly

2. **`check_understanding`** (RED FLAG SYSTEM)
   - REQUIRED before ANY code implementation
   - Prevents overconfident AI changes
   - Returns approval score (must be >= 8/10)

3. **`get_impact_analysis`**
   - Detailed analysis for specific files
   - Shows dependencies and risks
   - Suggests safer alternatives

### Updated Tools

- **`analyze_codebase`**: Now saves to persistent storage
- **`get_ai_knowledge_package`**: Redirects to `get_codebase_context`
- **`get_business_context`**: Returns optimized summaries

### New Configuration

**.claude/config.md**
- Automatically instructs Claude to load context first
- Enforces verification before implementation
- Cannot be overlooked by AI

## Migration Steps

### For Existing Users

1. **Update to v2.0**
   ```bash
   git pull origin main
   pip install -r requirements.txt
   ```

2. **Run One-Time Analysis**
   ```bash
   # This saves to ~/.codebaseiq/analysis_cache.json
   analyze_codebase /path/to/your/project
   ```

3. **Create Configuration**
   ```bash
   # Copy the template
   cp .claude/config.md /path/to/your/project/.claude/config.md
   ```

4. **Update Your Workflow**
   
   **Old Workflow:**
   ```
   1. analyze_codebase (every conversation)
   2. get_ai_knowledge_package
   3. Make changes
   ```
   
   **New Workflow:**
   ```
   1. get_codebase_context (instant)
   2. check_understanding (verification)
   3. get_impact_analysis (if needed)
   4. Make changes (only after approval)
   ```

### For New Users

1. **Install CodebaseIQ Pro v2.0**
2. **Run Initial Analysis** (one-time, 4-5 minutes)
3. **Use `get_codebase_context`** at conversation start

## New Features in Detail

### 1. Persistent Storage

Analysis is saved to `~/.codebaseiq/analysis_cache.json`
- Survives between conversations
- Auto-loaded by `get_codebase_context`
- Refresh with `get_codebase_context(refresh=true)`

### 2. Red Flag System

Before ANY implementation:
```python
check_understanding(
    implementation_plan="What you plan to do",
    files_to_modify=["file1.py", "file2.js"],
    understanding_points=[
        "Key insight 1",
        "Risk awareness",
        "Business impact"
    ]
)
```

### 3. Progressive Disclosure

- Initial tools return summaries
- Details available on demand
- Always under 25K token limit

### 4. Forced Verification

The `.claude/config.md` file ensures:
- Context loaded before code changes
- Understanding verified
- Cutting-edge tech handled safely

## Performance Improvements

| Operation | v1.0 | v2.0 |
|-----------|------|------|
| Initial Analysis | 4-5 min | 4-5 min (one-time) |
| New Conversation | 4-5 min | < 1 second |
| Context Loading | Full analysis | Optimized cache |
| Response Size | 1.2M tokens | < 25K tokens |

## Breaking Changes

1. **Tool Names**: Some tools renamed for clarity
2. **Response Format**: Optimized for size
3. **Workflow**: Verification now required

## Troubleshooting

### "No analysis available"
- Run `analyze_codebase` once
- Check `~/.codebaseiq/analysis_cache.json` exists

### "Understanding score too low"
- Provide more detailed plan
- Show risk awareness
- Include testing strategy

### "Response still too large"
- Use `get_codebase_context` instead of old tools
- Contact support if issue persists

## Support

- GitHub Issues: [Report bugs](https://github.com/chatcbdai/codebase-iq-pro/issues)
- Email: hi@chatcbd.com

## Summary

v2.0 transforms CodebaseIQ Pro from a powerful but token-heavy tool into an efficient, safety-first system that:
- ✅ Respects MCP's 25K token limit
- ✅ Solves the zero knowledge problem
- ✅ Forces AI to verify understanding
- ✅ Loads instantly from cache
- ✅ Prevents breaking changes

The future of AI-assisted coding is here - and it's safe by design! 🚀