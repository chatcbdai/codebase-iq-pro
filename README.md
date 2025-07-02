# CodebaseIQ Pro: Solving the AI "Zero Knowledge" Problem

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://github.com/modelcontextprotocol)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## The Challenge

Every time you start a new conversation with an AI coding assistant (Claude, ChatGPT, Copilot, etc.), it begins with **zero knowledge** of your codebase. This "fresh start" problem creates critical issues:

### 📁 Small Codebases (< 10 files)
✅ AI can quickly understand the entire structure  
✅ Changes are generally safe and accurate  
✅ Minimal risk of breaking existing functionality  

### 📚 Large Codebases (100+ files)
❌ AI makes **overconfident changes** without understanding interconnected systems  
❌ Creates **duplicate files** instead of updating existing ones  
❌ Causes **cascading errors** by breaking hidden dependencies  
❌ Wastes time exploring files one-by-one (and still misses critical context)  

## The Solution

CodebaseIQ Pro provides a **one-time comprehensive analysis** that gives AI assistants everything they need to know about your codebase upfront:

- 🧠 **Deep Understanding**: What each file does and WHY it exists
- 🔗 **Dependency Mapping**: How files connect and impact each other  
- ⚠️ **Danger Zones**: Critical files that shouldn't be modified
- 💼 **Business Logic**: The actual purpose behind the technical implementation
- 📋 **Safety Instructions**: Specific rules for making changes without breaking things

## The Result

Instead of AI assistants blindly making changes that seem correct but break your system, they now have:

- **Complete context** before suggesting any modifications
- **Awareness of ripple effects** from changes
- **Understanding of existing solutions** (no more duplicate implementations)
- **Knowledge of critical paths** that must be preserved

This transforms AI from a confident-but-dangerous helper into an informed collaborator that truly understands your codebase architecture.

---

### Why This Matters

Every secondary error from incomplete AI understanding costs:
- 🕐 **Time**: Debugging issues AI created
- 💰 **Money**: Token usage exploring files repeatedly  
- 😤 **Frustration**: Fixing "fixes" that broke other systems

CodebaseIQ Pro eliminates these issues with one comprehensive analysis that serves as your AI's complete knowledge base for every future conversation.

## ✨ Key Features

### 🧠 Enhanced Understanding (v2.0)
- **Deep Understanding Agent**: Semantic code analysis with purpose extraction
- **Cross-File Intelligence**: Impact analysis and circular dependency detection
- **Business Logic Extraction**: Translates code into business terms
- **AI Knowledge Packaging**: Instant context with safety instructions

### 🛡️ Safety First
- **Danger Zone Identification**: Critical files marked with clear warnings
- **Impact Analysis**: See how changes ripple through your codebase
- **Risk Assessment**: Every file rated (CRITICAL/HIGH/MEDIUM/LOW)
- **Safe Modification Guide**: Step-by-step checklists for changes

### 🎯 Multi-Agent Analysis
- **9 Specialized Agents**: Each focused on a specific aspect
- **Parallel Processing**: Fast analysis with intelligent orchestration
- **Adaptive Configuration**: Works with free or premium services
- **Language Support**: Python, JavaScript, TypeScript, Java, Go, and more

### 🔧 Flexible Infrastructure
- **Vector Search**: Qdrant (free/local) or Pinecone (premium)
- **Embeddings**: OpenAI (required) or Voyage AI (premium)
- **Caching**: In-memory or Redis for large codebases
- **Performance**: Handles codebases of any size

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key (required)
- MCP-compatible client (Claude Desktop, Cline VSCode, etc.)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/chatcbdai/codebase-iq-pro.git
cd codebase-iq-pro
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment**
```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Optional premium features
echo "VOYAGE_API_KEY=your-key-here" >> .env
echo "PINECONE_API_KEY=your-key-here" >> .env
```

4. **Configure MCP client**

For Claude Desktop, add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "codebase-iq-pro": {
      "command": "python",
      "args": ["/path/to/codebaseiq-pro/src/codebaseiq/server.py"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

## 📖 Usage

### Basic Analysis
```
analyze_codebase /path/to/your/project
```

### Get AI Knowledge Package (NEW!)
```
get_ai_knowledge_package
```
Returns comprehensive understanding with:
- Instant context (read in seconds)
- Danger zones with warnings
- Safe modification guidelines
- AI-specific instructions

### Check Before Modifying (NEW!)
```
get_modification_guidance /src/critical/auth.py
```
Returns:
- Risk level assessment
- Impact analysis (affected files)
- Safety checklist
- Safer alternatives

### Understand Business Logic (NEW!)
```
get_business_context
```
Returns:
- Domain entities and relationships
- User journeys
- Business rules
- Compliance requirements

### Semantic Search
```
semantic_code_search "authentication logic"
```

### Find Similar Code
```
find_similar_code /src/services/auth.py
```

## 🏗️ Architecture

```
CodebaseIQ Pro
├── Core Engine
│   ├── Adaptive Configuration
│   ├── Simple Orchestrator
│   └── Analysis Base
├── Analysis Agents (Phase 1)
│   ├── Dependency Analysis
│   ├── Security Audit
│   ├── Pattern Detection
│   ├── Architecture Analysis
│   └── Performance Analysis
├── Enhanced Agents (Phase 2-4)
│   ├── Deep Understanding Agent
│   ├── Cross-File Intelligence
│   ├── Business Logic Extractor
│   └── AI Knowledge Packager
└── Services
    ├── Vector Database (Qdrant/Pinecone)
    ├── Embeddings (OpenAI/Voyage)
    └── Cache (Memory/Redis)
```

## 📊 Performance

- **Analysis Speed**: < 60s for 10K files
- **Search Latency**: < 100ms
- **Memory Usage**: < 2GB
- **Parallel Processing**: Uses all CPU cores
- **Smart Caching**: Reduces redundant analysis

## 🔐 Security & Compliance

- **Never stores credentials**: All secrets via environment variables
- **Secure file handling**: Respects .gitignore patterns
- **Compliance detection**: Identifies HIPAA, PCI, SOC2 requirements
- **Access control**: Configurable file/directory restrictions

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Format code
black src/
```

## 📝 Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [MCP Setup Guide](docs/MCP_SETUP_GUIDE.md)
- [Technical Deep Dive](docs/technical-deep-dive.md)
- [Enhanced Architecture](docs/enhanced-architecture.md)
- [Implementation Guide](docs/implementation-guide.md)

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'codebaseiq'"**
   - Run: `pip install -e .`

2. **"OpenAI API key not found"**
   - Ensure `.env` file exists with `OPENAI_API_KEY`

3. **"Analysis takes too long"**
   - Try `analysis_type="quick"` for faster results
   - Reduce file size limit in config

## 📈 Roadmap

- [x] **v2.0**: Enhanced understanding with AI safety
- [ ] **v2.1**: Language-specific security rules
- [ ] **v2.2**: Real-time file watching
- [ ] **v2.3**: Git history analysis
- [ ] **v2.4**: Team knowledge sharing
- [ ] **v3.0**: Cloud-based analysis

## 📄 License

MIT License with Attribution - see [LICENSE](LICENSE) file for details.

**Important**: While this is open source software, we require attribution when using or building upon CodebaseIQ Pro. This helps us track adoption and build a community of contributors.

## 🤝 Join Our Team

Are you passionate about code analysis and AI-assisted development? We're actively looking for experienced developers to join our core team! 

**We're especially interested in:**
- Language analyzer experts (Go, Rust, C++, Ruby, etc.)
- Security researchers
- AI/ML engineers
- DevOps and infrastructure specialists

**To join:** Email us at hi@chatcbd.com with:
- Your GitHub profile
- Areas of expertise
- Why you're excited about CodebaseIQ Pro

## 🙏 Acknowledgments

- Built for the [Model Context Protocol](https://github.com/modelcontextprotocol)
- Inspired by the need to reduce cost, headaches, and unnecessary time wasted.
- Thanks to all contributors and early adopters

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/chatcbdai/codebase-iq-pro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chatcbdai/codebase-iq-pro/discussions)

## 🚀 CodebaseIQ Pro Team

- 🦉 **Idea Guy**: @cannavis - [Christopher Visser](https://www.thcgummies.com/auther/christopher-visser)
- ⚙️ **Developer**: [Claude Code Opus 4](https://claude.ai/code)  
- 🤖 **Coordinator**: @chatcbdai - [ChatCBD](https://chatcbd.com)
- 🏧 **Sponsor**: Cannabidiol Life - [Stress & Headache Reliever](https://cbdoilsandedibles.com)
- 📧 **Give Us A Shout**: hi@chatcbd.com

---

**Transform your codebase understanding. Make AI-assisted coding less stressful. Spend more time improving your project and less time explaining and retraining for every new conversation. Check it out: Try CodebaseIQ Pro today!** 🚀
