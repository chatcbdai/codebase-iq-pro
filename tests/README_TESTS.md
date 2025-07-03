# CodebaseIQ Pro Test Suite

This directory contains comprehensive tests for the CodebaseIQ Pro MCP Server, designed to validate all functionality against a real-world codebase.

## Test Codebase

The tests use the `codebase_tester_for_codebaseIQ-pro` directory as a test subject. This is a Node.js/Express/Prisma application that provides a realistic codebase for testing analysis capabilities.

## Test Files

### 1. `test_mcp_server_full.py`
Comprehensive test suite that validates all MCP tools:
- Complete codebase analysis (4-phase process)
- State persistence and restoration
- Semantic code search
- Find similar code functionality
- All individual analysis tools (25K token tools)
- AI knowledge package generation
- Modification guidance
- Understanding verification
- Error handling and graceful degradation

### 2. `test_persistence.py`
Focused tests for state persistence functionality:
- Initial analysis creates persistence files
- New server instances restore state automatically
- Performance validation (<5 second restoration)
- Cache expiration (7-day limit)
- Git-aware caching
- Embeddings are not regenerated
- Search functionality works after restore

### 3. `run_tests.py`
Convenient test runner with multiple options.

## Running Tests

### Prerequisites
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Set required environment variable
export OPENAI_API_KEY=your-key-here
```

### Quick Validation
```bash
# Run essential tests only (fastest)
python tests/run_tests.py quick
```

### Persistence Tests
```bash
# Test state persistence specifically
python tests/run_tests.py persistence
```

### Full Test Suite
```bash
# Run all MCP server tests (comprehensive)
python tests/run_tests.py full
```

### All Tests
```bash
# Run both persistence and full test suites
python tests/run_tests.py all
```

### Verbose Mode
```bash
# Add --verbose for detailed logging
python tests/run_tests.py full --verbose
```

### Direct Test Execution
```bash
# Run tests directly
PYTHONPATH=/path/to/codebase_iq_pro python tests/test_persistence.py
PYTHONPATH=/path/to/codebase_iq_pro python tests/test_mcp_server_full.py
```

## Test Coverage

The test suite validates:

### Core Functionality
- ✅ Complete 4-phase analysis process
- ✅ Multi-agent orchestration
- ✅ Vector embeddings generation
- ✅ Caching mechanisms

### State Management
- ✅ Automatic state restoration on startup
- ✅ Performance (<5 second requirement)
- ✅ Cache expiration handling
- ✅ Git-aware cache invalidation

### Search Capabilities
- ✅ Semantic code search
- ✅ Find similar code
- ✅ Search works without full analysis

### Analysis Tools
- ✅ Dependency analysis
- ✅ Security analysis
- ✅ Architecture analysis
- ✅ Business logic analysis
- ✅ Technical stack analysis
- ✅ Code intelligence analysis

### AI Integration
- ✅ AI knowledge package generation
- ✅ Modification guidance
- ✅ Understanding verification
- ✅ Danger zone identification

### Error Handling
- ✅ Invalid paths
- ✅ Missing files
- ✅ Graceful degradation

## Expected Results

### Successful Test Run
- All tests should pass with a >90% success rate
- State restoration should complete in <5 seconds
- Embeddings should not be regenerated on restart
- Search functionality should work immediately after restore

### Performance Benchmarks
- Initial analysis: 30-60 seconds (depending on codebase size)
- State restoration: <5 seconds
- Semantic search: <1 second
- Individual analysis tools: 5-15 seconds each

## Troubleshooting

### Common Issues

1. **"No module named 'codebaseiq'"**
   - Ensure PYTHONPATH is set correctly
   - Run from the project root directory

2. **"OpenAI API key not found"**
   - Set OPENAI_API_KEY environment variable
   - Check .env file exists with the key

3. **Tests timing out**
   - Increase timeout values in test files
   - Check network connectivity for API calls

4. **State not persisting**
   - Check .codebaseiq_cache directory permissions
   - Verify _needs_state_restoration flag is True

## Adding New Tests

To add new tests:

1. Add test method to appropriate test class
2. Follow naming convention: `test_feature_name()`
3. Use `self.test_tool()` helper for MCP tool testing
4. Log results with `self.log_test()` in persistence tests
5. Update this README with new test coverage

## CI/CD Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run CodebaseIQ Pro Tests
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    pip install -r requirements.txt
    python tests/run_tests.py all
```