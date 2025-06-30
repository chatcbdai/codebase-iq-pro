# CodebaseIQ Pro - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Prerequisites

- Python 3.9 or higher
- OpenAI API key (required)
- Optional: Voyage AI key (better code embeddings)
- Optional: Pinecone API key (cloud vector database)

### 2. Installation

```bash
# Clone or download the CodebaseIQ Pro files
cd codebase_iq_pro

# Run the setup script
python setup.py

# Install dependencies if prompted
# The setup script will guide you through everything
```

### 3. Configuration

Create a `.env` file with your API keys:

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your keys
nano .env  # or use your favorite editor
```

**Minimum configuration** (only OpenAI required):
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Premium configuration** (all features):
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
VOYAGE_API_KEY=your-voyage-api-key      # Better code embeddings
PINECONE_API_KEY=your-pinecone-api-key  # Cloud vector DB
REDIS_URL=redis://localhost:6379        # Distributed cache
```

### 4. Verify Installation

```bash
claude mcp list
# Should show: codebase-iq-pro
```

### 5. Start Using CodebaseIQ Pro

In VS Code with Claude:

```
// Analyze your entire codebase
analyze_codebase path: "."

// Quick analysis
analyze_codebase path: ".", analysis_type: "quick"

// Security-focused analysis
analyze_codebase path: ".", analysis_type: "security_focus"

// Search for code semantically
semantic_code_search query: "authentication logic"

// Find similar code
find_similar_code entity_path: "src/auth/login.py"

// Get analysis summary
get_analysis_summary

// Check security issues
get_danger_zones

// View dependencies
get_dependencies
```

## 🎯 Common Use Cases

### 1. First-Time Codebase Analysis
```
analyze_codebase path: "/path/to/project"
```
This creates embeddings and analyzes everything. Takes 30-60 seconds for most projects.

### 2. Find Code by Description
```
semantic_code_search query: "function that handles user login"
```

### 3. Security Audit
```
analyze_codebase path: ".", analysis_type: "security_focus"
get_danger_zones
```

### 4. Check Test Coverage
```
get_analysis_summary
// Look for test_coverage in the results
```

### 5. Find Dependencies
```
get_dependencies
// Shows all external packages and internal dependencies
```

## 💡 Tips

1. **Free vs Premium**:
   - Free: Uses OpenAI embeddings + local Qdrant database
   - Premium: Voyage AI (better for code) + Pinecone (faster, cloud-based)

2. **Performance**:
   - First analysis takes longer (building embeddings)
   - Subsequent searches are instant
   - Results are cached for 1 hour

3. **Large Codebases**:
   - Set `MAX_FILE_SIZE_MB=20` in .env for larger files
   - Use `analysis_type: "quick"` for faster initial scan

4. **Best Practices**:
   - Run full analysis once per session
   - Use semantic search instead of grep
   - Check danger zones before making changes

## 🆘 Troubleshooting

### "OPENAI_API_KEY is required"
- Make sure your .env file exists and contains: `OPENAI_API_KEY=sk-...`
- Check the key is valid at https://platform.openai.com/api-keys

### "Vector search not available"
- Run `analyze_codebase` first to create embeddings
- Make sure `enable_embeddings: true` (default)

### "Module not found" errors
- Run `python setup.py` and install missing packages
- Make sure you're using Python 3.9+

### Slow analysis
- Normal for first run (building embeddings)
- Use `analysis_type: "quick"` for faster results
- Reduce `MAX_FILE_SIZE_MB` to skip large files

## 📊 Understanding Results

### Analysis Summary
- **security_score**: 0-10 (higher is better)
- **code_quality_score**: 0-10 (higher is better)
- **documentation_score**: 0-1 (percentage)
- **test_coverage**: Percentage of files with tests

### Danger Zones
Files with security risks, sorted by severity:
- Level 7-10: Critical security issues
- Level 4-6: Important to review
- Level 1-3: Minor concerns

### Semantic Search Results
- **score**: 0-1 (similarity score, higher is better)
- **path**: File location
- **type**: Entity type (function, class, etc.)

## 🚀 Advanced Features

### Custom Analysis Types
- `"full"`: Complete analysis (default)
- `"security_focus"`: Security and dependencies only
- `"performance_focus"`: Performance and architecture
- `"quick"`: Fast overview

### Filtering Search Results
```
semantic_code_search query: "database", filters: {"type": "class"}
```

### Similarity Threshold
```
find_similar_code entity_path: "src/main.py", similarity_threshold: 0.8
```

## 📚 Next Steps

1. Read the [Usage Guide](USAGE.md) for detailed examples
2. Check the [MCP Setup Guide](MCP_SETUP_GUIDE.md) for configuration
3. Review [technical architecture](enhanced-architecture.md)
4. See [implementation details](implementation-guide.md)

Happy coding with CodebaseIQ Pro! 🎉