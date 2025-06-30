# CodebaseIQ Pro Usage Guide

## VS Code Integration

CodebaseIQ Pro works seamlessly with Claude Code in VS Code. When you open a project in VS Code and start Claude Code, the MCP server analyzes YOUR project, not the server installation directory.

### Quick Start

1. **Open your project in VS Code**
   ```bash
   code /path/to/your/project
   ```

2. **Start Claude Code in the terminal**
   ```bash
   claude
   ```

3. **Analyze your workspace**
   ```
   > analyze_codebase path: "."
   ```
   This analyzes YOUR project directory!

### Path Resolution

- `path: "."` → Your VS Code workspace
- `path: "./src"` → The src folder in your workspace
- `path: "/absolute/path"` → Any absolute path

### Common Usage Patterns

#### Analyze current workspace
```
analyze_codebase path: "."
```

#### Analyze a subdirectory
```
analyze_codebase path: "./backend"
```

#### Search for code
```
semantic_code_search query: "user authentication logic"
```

#### Find similar implementations
```
find_similar_code entity_path: "./src/auth/login.js"
```

#### Get security insights
```
get_danger_zones
```

#### Check dependencies
```
get_dependencies
```

## Analysis Types

- **`full`** (default): Complete analysis with all agents
- **`security_focus`**: Focus on security vulnerabilities
- **`performance_focus`**: Focus on performance issues
- **`quick`**: Fast overview for large codebases

Example:
```
analyze_codebase path: ".", analysis_type: "security_focus"
```

## Working with Results

### Analysis Summary
```
get_analysis_summary
```

Returns:
- Security score (0-10)
- Code quality metrics
- Documentation coverage
- Test coverage
- Key insights from all agents

### Danger Zones
```
get_danger_zones
```

Lists files with security risks, sorted by severity:
- Level 7-10: Critical issues
- Level 4-6: Important to review
- Level 1-3: Minor concerns

### Semantic Search
```
semantic_code_search query: "database connections", top_k: 20
```

Parameters:
- `query`: Natural language description
- `top_k`: Number of results (default: 10)
- `filters`: Optional filters like `{"type": "function"}`
- `search_type`: "semantic", "keyword", or "hybrid"

## Pro Tips

1. **First analysis takes longer** (building embeddings), subsequent searches are instant
2. **Results are cached** for 1 hour for performance
3. **For large codebases**, use `analysis_type: "quick"` first
4. **Use relative paths** to make commands portable
5. **The server is stateless** - each analysis is independent

## Examples by Project Type

### React/Frontend Project
```
> analyze_codebase path: "."
> semantic_code_search query: "useState hooks"
> semantic_code_search query: "API calls"
> find_similar_code entity_path: "./src/components/Button.jsx"
```

### Python/Backend Project
```
> analyze_codebase path: "."
> semantic_code_search query: "database models"
> semantic_code_search query: "authentication decorators"
> get_dependencies
```

### Full-Stack Application
```
> analyze_codebase path: "./frontend"
> analyze_codebase path: "./backend"
> semantic_code_search query: "API endpoints"
> get_danger_zones
```

## Advanced Features

### Disable Embeddings
For faster analysis without semantic search:
```
analyze_codebase path: ".", enable_embeddings: false
```

### Focus Areas
Specify what to analyze:
```
analyze_codebase path: ".", focus_areas: ["security", "performance"]
```

### Cross-Project Analysis
Analyze other projects using absolute paths:
```
analyze_codebase path: "/Users/chrisryviss/other-project"
```

## Common Misconceptions

❌ **Wrong**: "The server only analyzes its own code"
✅ **Right**: The server can analyze ANY codebase you point it to

❌ **Wrong**: "I need to cd to my project before using analyze_codebase"
✅ **Right**: Just open your project in VS Code, Claude Code inherits the workspace context

❌ **Wrong**: "The path is relative to the server installation"
✅ **Right**: The path is relative to your current working directory (VS Code workspace)