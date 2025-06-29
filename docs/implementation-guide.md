# CodebaseIQ Pro : Implementation Guide

## Quick Start Guide

### Prerequisites

```bash
# Python 3.9+ required
python --version

# Install core dependencies
pip install mcp aiofiles tree-sitter networkx cryptography watchdog redis msgpack numpy

# Install vector database client (choose one)
pip install pinecone-client  # Recommended
# OR
pip install qdrant-client
# OR
pip install weaviate-client

# Install orchestration frameworks
pip install langgraph crewai
```

### Environment Setup

Create a `.env` file in your project root:

```bash
# Vector Database Configuration
VECTOR_DB_TYPE=pinecone
VECTOR_DB_URL=https://api.pinecone.io
VECTOR_DB_API_KEY=your-pinecone-api-key

# Embedding Service
EMBEDDING_MODEL=voyage-3-lite
EMBEDDING_API_KEY=your-voyage-api-key
# Alternative: Use OpenAI
# EMBEDDING_MODEL=text-embedding-3-large
# EMBEDDING_API_KEY=your-openai-api-key

# Security
VAULT_URL=http://localhost:8200
VAULT_TOKEN=your-vault-token
ENABLE_SANDBOX=true

# Cache
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600

# Performance
MAX_WORKERS=10
BATCH_SIZE=100
```

### VS Code Configuration

Update your `.vscode/mcp.json`:

```json
{
  "mcpServers": {
    "codebase-iq-pro-": {
      "command": "python",
      "args": ["/path/to/codebase-iq-pro-.py"],
      "env": {
        "PYTHONPATH": ".",
        "CODEBASE_IQ_CACHE": "${workspaceFolder}/.codebase-iq-cache"
      }
    }
  }
}
```

### Docker Deployment (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  codebase-iq-pro:
    build: .
    environment:
      - VECTOR_DB_API_KEY=${VECTOR_DB_API_KEY}
      - EMBEDDING_API_KEY=${EMBEDDING_API_KEY}
      - REDIS_URL=redis://redis:6379
    volumes:
      - ${PROJECT_PATH}:/workspace:ro
      - ./cache:/app/cache
    depends_on:
      - redis
      - vault

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  vault:
    image: hashicorp/vault:latest
    cap_add:
      - IPC_LOCK
    environment:
      VAULT_DEV_ROOT_TOKEN_ID: dev-token
      VAULT_DEV_LISTEN_ADDRESS: 0.0.0.0:8200
    ports:
      - "8200:8200"

volumes:
  redis-data:
```

## Usage Examples

### Example 1: Initial Full Analysis with Embeddings

```
You: Analyze this codebase with full semantic search capabilities

Claude: I'll perform a comprehensive analysis with semantic embeddings...

[Uses analyze_codebase_ tool with enable_embeddings=true]

✅ Analysis Complete:
- 1,847 files analyzed across 12 modules
- 1,847 semantic embeddings created
- Vector search index ready
- Security score: 8.7/10
- Documentation coverage: 72%
- Test coverage: 68%

Key Findings:
1. **Architecture**: Microservices with event-driven communication
2. **Stack**: Node.js 18.17.0, React 17.0.2, PostgreSQL 14
3. **Danger Zones**: 
   - `/src/payments` - Payment processing (level 9/10)
   - `/src/auth` - Authentication system (level 8/10)
4. **Code Quality**: Well-structured with consistent patterns

You can now use semantic search to find any code pattern instantly!
```

### Example 2: Semantic Code Search

```
You: Find all code that handles user authentication

Claude: [Uses semantic_code_search tool]

Found 12 semantically related files:

1. **src/auth/authService.ts** (similarity: 0.95)
   - Main authentication service
   - Handles login, logout, token management
   
2. **src/middleware/authMiddleware.ts** (similarity: 0.89)
   - Express middleware for route protection
   - JWT token validation
   
3. **src/components/LoginForm.tsx** (similarity: 0.86)
   - React component for user login
   - Form validation and submission

4. **src/utils/tokenManager.ts** (similarity: 0.84)
   - JWT token utilities
   - Token refresh logic

[Additional results...]

Would you like me to analyze any of these files in detail?
```

### Example 3: Finding Similar Code

```
You: What code is similar to our PaymentProcessor class?

Claude: [Uses find_similar_code tool]

Files similar to `src/services/PaymentProcessor.ts`:

1. **src/services/OrderProcessor.ts** (similarity: 0.87)
   - Similar transaction handling pattern
   - Uses same error handling approach
   - Suggestion: Consider extracting shared transaction logic

2. **src/services/RefundService.ts** (similarity: 0.82)
   - Related payment operations
   - Could share validation utilities

3. **src/services/SubscriptionManager.ts** (similarity: 0.78)
   - Recurring payment logic
   - Similar state machine pattern

These files could benefit from a shared base class or common utilities.
```

### Example 4: Security-Focused Analysis

```
You: Run a security audit on the authentication system

Claude: [Uses analyze_with_focus tool with focus_areas=["security"]]

🔒 Security Analysis Complete:

**Critical Findings:**

1. **Hardcoded Secrets** (CRITICAL)
   - File: `src/config/auth.config.ts`
   - Line 23: JWT secret in source code
   - Fix: Move to environment variables immediately

2. **SQL Injection Risk** (HIGH)
   - File: `src/auth/userQuery.ts`
   - Line 45: String concatenation in SQL query
   - Fix: Use parameterized queries

3. **Weak Password Policy** (MEDIUM)
   - No minimum length enforcement
   - No complexity requirements
   - Fix: Implement OWASP password guidelines

**Positive Security Measures:**
✓ bcrypt for password hashing (good!)
✓ Rate limiting on login endpoint
✓ CSRF protection enabled

**Recommended Actions:**
1. Rotate all secrets immediately
2. Implement security headers (HSTS, CSP)
3. Add 2FA support
4. Enable audit logging for all auth events
```

### Example 5: Multi-Agent Insights

```
You: Show me insights from the documentation agent

Claude: [Uses get_agent_insights tool]

📚 Documentation Agent Insights:

**Overall Documentation Score: 6.8/10**

**Well-Documented Files (10):**
- src/api/README.md - Comprehensive API documentation
- src/components/Button/Button.tsx - Complete JSDoc
- src/utils/validators.ts - All functions documented

**Poorly Documented Files (47):**
- src/services/CacheService.ts - No documentation
- src/models/User.ts - Missing property descriptions
- src/middleware/errorHandler.ts - No usage examples

**Recommendations:**
1. Add module-level documentation to all services
2. Document all public APIs with examples
3. Add type documentation for complex interfaces
4. Consider using documentation generation tools

**Documentation Patterns Detected:**
- JSDoc style in JavaScript files
- Docstrings in Python files
- Inline comments for complex logic
```

## Advanced Features

### Custom Analysis Workflows

Create custom analysis workflows by combining agents:

```python
# In your code
async def custom_security_analysis(codebase_path):
    # 1. Run security-focused analysis
    security_results = await analyze_with_focus(
        path=codebase_path,
        focus_areas=["security", "compliance"],
        depth="deep"
    )
    
    # 2. Search for specific vulnerable patterns
    vulnerable_patterns = await semantic_code_search(
        query="eval exec subprocess shell injection SQL concatenation",
        top_k=20
    )
    
    # 3. Check similar code to vulnerable files
    for file in vulnerable_patterns:
        similar = await find_similar_code(file['path'], top_k=3)
        # Analyze if vulnerabilities are widespread
    
    return comprehensive_security_report
```

### Integration with CI/CD

```yaml
# .github/workflows/codebase-analysis.yml
name: CodebaseIQ Analysis

on:
  pull_request:
    branches: [main, develop]

jobs:
  analyze:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup CodebaseIQ Pro
        run: |
          docker-compose up -d
          sleep 10  # Wait for services
          
      - name: Run Analysis
        run: |
          docker exec codebase-iq-pro python -c "
          import asyncio
          from codebase_iq_client import analyze_codebase_
          
          async def main():
              result = await analyze_codebase_(
                  path='/workspace',
                  analysis_type='security_focus'
              )
              
              # Check security score
              security_score = result['results']['security']['security_score']
              if security_score < 7.0:
                  print(f'Security score too low: {security_score}')
                  exit(1)
                  
              # Check for critical vulnerabilities
              vulns = result['results']['security']['vulnerabilities']
              critical = [v for v in vulns if v['severity'] >= 9]
              if critical:
                  print(f'Critical vulnerabilities found: {len(critical)}')
                  exit(1)
                  
          asyncio.run(main())
          "
          
      - name: Upload Analysis Report
        uses: actions/upload-artifact@v3
        with:
          name: codebase-analysis
          path: .codebase-iq-cache/analysis-report.json
```

### Semantic Search API

Build a semantic search API for your team:

```python
from fastapi import FastAPI, Query
from codebase_iq_client import semantic_code_search

app = FastAPI()

@app.get("/search")
async def search_code(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Number of results"),
    type_filter: Optional[str] = Query(None, description="Filter by entity type")
):
    """Search codebase using natural language"""
    
    filters = {"type": type_filter} if type_filter else None
    
    results = await semantic_code_search(
        query=q,
        top_k=limit,
        filters=filters
    )
    
    return {
        "query": q,
        "count": len(results),
        "results": results
    }

# Example queries:
# GET /search?q=authentication+logic
# GET /search?q=find+payment+processing&type_filter=service
# GET /search?q=React+components+with+state+management
```

## Performance Tuning

### Optimizing Analysis Speed

1. **Adjust Worker Count**
   ```bash
   export MAX_WORKERS=20  # For machines with many cores
   ```

2. **Increase Batch Size**
   ```bash
   export BATCH_SIZE=200  # For better throughput
   ```

3. **Enable Shallow Analysis**
   ```python
   # For quick overview
   result = await analyze_with_focus(
       path="/project",
       focus_areas=["architecture"],
       depth="shallow"
   )
   ```

### Optimizing Vector Search

1. **Use Appropriate Embedding Model**
   - `voyage-3-lite`: Best balance of speed and quality
   - `text-embedding-3-small`: Fastest, lower quality
   - `voyage-3-large`: Highest quality, slower

2. **Configure Vector Index**
   ```python
   # Pinecone optimization
   index = pinecone.create_index(
       "codebase-iq-pro",
       dimension=1024,
       metric="cosine",
       pod_type="p1.x1"  # Performance pod
   )
   ```

3. **Implement Caching**
   ```python
   # Cache frequent searches
   @lru_cache(maxsize=1000)
   async def cached_search(query: str):
       return await semantic_code_search(query)
   ```

## Troubleshooting

### Common Issues

1. **"Vector search not initialized"**
   - Run `analyze_codebase_` with `enable_embeddings=true` first
   - Check VECTOR_DB_API_KEY is set correctly

2. **"Analysis taking too long"**
   - Reduce MAX_FILE_SIZE_MB for large codebases
   - Use `analysis_type="performance_focus"` for faster analysis
   - Enable caching with Redis

3. **"Security sandbox blocking operations"**
   - Check audit logs for blocked operations
   - Ensure paths don't contain sensitive patterns
   - Verify ENABLE_SANDBOX setting

4. **"Embedding API rate limit"**
   - Implement exponential backoff
   - Reduce BATCH_SIZE
   - Consider using local embedding models

### Debug Mode

Enable detailed logging:

```bash
export CODEBASE_IQ_DEBUG=true
export LOG_LEVEL=DEBUG
```

Check logs:
```bash
tail -f ~/.codebase-iq-pro/logs/analysis.log
```


## Conclusion

CodebaseIQ Pro  transforms how AI understands your code. With semantic search, multi-agent analysis, and enterprise-grade security, it's the definitive solution for AI-assisted development in 2025.
