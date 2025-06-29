# CodebaseIQ Pro

<p align="center">
  <strong>Advanced MCP Server for Intelligent Codebase Analysis</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#api-reference">API</a> •
  <a href="#contributing">Contributing</a>
</p>

---

CodebaseIQ Pro is an advanced Model Context Protocol (MCP) server that provides instant, comprehensive codebase understanding through AI-powered analysis. It solves the "zero knowledge" problem by offering semantic code search, multi-agent orchestration, and enterprise-grade security features.

## 🌟 Features

### Core Capabilities
- **⚡ Lightning-Fast Analysis** - Analyze any codebase in 30-60 seconds
- **🔍 Semantic Code Search** - Find code using natural language queries
- **🤖 Multi-Agent System** - 11 specialized agents working in parallel
- **🔒 Enterprise Security** - Sandbox execution, credential detection, audit logging
- **📊 Comprehensive Insights** - Architecture, dependencies, security, performance

### Adaptive Service Selection
- **Free Tier** (Default)
  - OpenAI embeddings (required)
  - Qdrant local vector database
  - In-memory caching
  
- **Premium Tier** (Optional)
  - Voyage AI embeddings (optimized for code)
  - Pinecone cloud vector database
  - Redis distributed caching

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- OpenAI API key (required)
- VS Code with Claude extension

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chatcbdai/codebase-iq-pro.git
   cd codebase-iq-pro
   ```

2. **Run setup**
   ```bash
   python setup.py
   ```

3. **Configure API key** (One-time setup)
   - Edit the `.env` file
   - Replace `your-openai-api-key-here` with your actual OpenAI API key
   - That's it! The key will be automatically loaded from the .env file

4. **Configure Claude Code** (Already done!)
   
   If you're using Claude Desktop:
   - CodebaseIQ Pro is already configured in your claude_desktop_config.json
   - Just restart Claude Desktop to use it
   
   For Claude Code in VS Code:
   - Run: `claude mcp list` to verify it's installed
   - Start a new Claude Code session to access the tools

5. **Start using with Claude**
   ```
   analyze_codebase path: "."
   semantic_code_search query: "authentication logic"
   get_analysis_summary
   ```

## 📖 Documentation

### Available MCP Tools

#### `analyze_codebase`
Perform comprehensive codebase analysis with multi-agent orchestration.

**Parameters:**
- `path` (string, required): Path to analyze
- `analysis_type` (string): "full", "security_focus", "performance_focus", "quick"
- `enable_embeddings` (boolean): Create vector embeddings (default: true)
- `focus_areas` (list): Specific areas to focus on

**Example:**
```
analyze_codebase path: "/my/project", analysis_type: "security_focus"
```

#### `semantic_code_search`
Search code using natural language queries.

**Parameters:**
- `query` (string, required): Natural language search query
- `top_k` (integer): Number of results (default: 10)
- `filters` (object): Optional filters
- `search_type` (string): "semantic", "keyword", or "hybrid"

**Example:**
```
semantic_code_search query: "function that handles user authentication"
```

#### `find_similar_code`
Find code similar to a given file or function.

**Parameters:**
- `entity_path` (string, required): Path to code entity
- `top_k` (integer): Number of results (default: 5)
- `similarity_threshold` (float): Minimum similarity (0-1, default: 0.7)

#### `get_analysis_summary`
Get a summary of the current analysis results.

#### `get_danger_zones`
Get security-sensitive areas requiring attention.

#### `get_dependencies`
View dependency graph and package information.

## 🏗️ Architecture

```
codebase-iq-pro/
├── src/
│   └── codebaseiq/
│       ├── __init__.py
│       ├── server.py          # Main MCP server
│       ├── agents/            # Analysis agents
│       │   ├── analysis_agents.py
│       │   ├── embedding_agent.py
│       │   └── ...
│       ├── core/             # Core components
│       │   ├── adaptive_config.py
│       │   ├── analysis_base.py
│       │   └── simple_orchestrator.py
│       └── services/         # External services
│           ├── vector_db.py
│           ├── embedding_service.py
│           └── cache_service.py
├── docs/                     # Documentation
├── tests/                    # Test suite
├── codebaseiq_pro.py        # Entry point
├── setup.py                 # Setup script
├── requirements.txt         # Dependencies
└── .env                     # Configuration
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install in development mode
   ```bash
   pip install -r requirements.txt
   python -m pip install -e .
   ```
4. Run tests
   ```bash
   python tests/test_setup.py
   ```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Analysis Speed | < 60s for 10K files |
| Search Latency | < 100ms |
| Memory Usage | < 2GB |
| Accuracy | > 95% relevance |

## 🔒 Security

- All operations run in sandboxed environments
- Automatic credential detection and masking
- Comprehensive audit logging
- No data leaves your local environment (unless using cloud services)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for the MCP (Model Context Protocol) ecosystem
- Powered by OpenAI embeddings and optional Voyage AI
- Vector search by Qdrant (local) or Pinecone (cloud)

---

<p align="center">
  Made with ❤️ for developers who value intelligent code analysis
</p>