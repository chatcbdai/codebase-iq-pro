# Contributing to CodebaseIQ Pro

First off, thank you for considering contributing to CodebaseIQ Pro! It's people like you that make this tool better for everyone.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, etc.)
- Any relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- A clear description of the enhancement
- Why this enhancement would be useful
- Possible implementation approach
- Any potential drawbacks

### Pull Requests

1. Fork the repo and create your branch from `main`
2. Make your changes following our coding standards
3. Add tests for any new functionality
4. Ensure all tests pass
5. Update documentation as needed
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/codebase-iq-pro.git
cd codebase-iq-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install dev dependencies
pip install pytest black flake8
```

## Coding Standards

### Python Style Guide

- Follow PEP 8
- Use Black for formatting: `black src/`
- Use type hints where possible
- Write docstrings for all public functions/classes

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests

Example:
```
Add deep understanding agent for semantic analysis

- Implement AST parsing for Python and JavaScript
- Add risk assessment for file modifications
- Create comprehensive test suite

Fixes #123
```

### Testing

- Write tests for all new features
- Maintain or improve code coverage
- Run tests before submitting PR: `pytest tests/`

### Documentation

- Update README.md if needed
- Add docstrings to new functions/classes
- Update relevant documentation in docs/

## Project Structure

```
src/codebaseiq/
├── agents/          # Analysis agents
├── core/            # Core components
├── services/        # External services
└── server.py        # Main MCP server

tests/               # Test suite
docs/                # Documentation
```

## Agent Development

When creating new agents:

1. Inherit from `AnalysisAgent` base class
2. Implement required methods
3. Follow the existing agent patterns
4. Add comprehensive tests
5. Document the agent's purpose and capabilities

Example:
```python
class MyNewAgent(AnalysisAgent):
    """Agent for analyzing specific aspect of code."""
    
    def __init__(self):
        super().__init__(
            name="my_new_agent",
            role=AgentRole.ANALYSIS,
            priority=5
        )
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis."""
        # Your implementation here
        pass
```

## Questions?

Feel free to open an issue for any questions or join our discussions!

Thank you for contributing! 🎉