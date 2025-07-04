# CodebaseIQ Pro - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying CodebaseIQ Pro MCP Server in a production environment with proper monitoring, security, and reliability.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.8 or higher
- **Memory**: Minimum 2GB RAM (4GB+ recommended for large codebases)
- **Storage**: 10GB+ free space for embeddings and cache
- **CPU**: 2+ cores recommended

### Required Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Production extras
pip install psutil  # For resource monitoring
```

### Environment Variables
Create a `.env` file with the following:

```bash
# REQUIRED
OPENAI_API_KEY=your-openai-api-key

# PRODUCTION MODE
CODEBASEIQ_PRODUCTION=true

# Optional Premium Services
VOYAGE_API_KEY=your-voyage-key          # Better embeddings
PINECONE_API_KEY=your-pinecone-key     # Cloud vector DB
PINECONE_ENVIRONMENT=your-env
REDIS_URL=redis://localhost:6379        # Distributed cache

# Performance Settings
MAX_WORKERS=10                          # Parallel processing
CACHE_TTL=3600                         # Cache duration (seconds)
MAX_FILE_SIZE=1048576                  # 1MB limit per file
LOG_LEVEL=INFO                         # Logging verbosity
```

## Pre-Deployment Checklist

### 1. Run Production Tests
```bash
# Run critical tests
python tests/test_critical_manually.py

# Run full test suite (if pytest is configured)
python run_production_tests.py
```

Expected output:
```
✅ ALL CRITICAL TESTS PASSED - Server is production ready!
```

### 2. Verify Configuration
```bash
# Check environment
python -c "from src.codebaseiq.server import CodebaseIQProServer; print('✅ Server imports successfully')"

# Verify API keys
python -c "import os; assert os.getenv('OPENAI_API_KEY'), 'Missing OPENAI_API_KEY'"
```

### 3. Test Server Startup
```bash
# Test in standard mode first
python src/codebaseiq/server.py

# Then test in production mode
CODEBASEIQ_PRODUCTION=true python src/codebaseiq/server.py
```

## Production Deployment

### Option 1: Systemd Service (Linux)

Create `/etc/systemd/system/codebaseiq-pro.service`:

```ini
[Unit]
Description=CodebaseIQ Pro MCP Server
After=network.target

[Service]
Type=simple
User=codebaseiq
WorkingDirectory=/opt/codebaseiq-pro
Environment="CODEBASEIQ_PRODUCTION=true"
EnvironmentFile=/opt/codebaseiq-pro/.env
ExecStart=/usr/bin/python3 /opt/codebaseiq-pro/src/codebaseiq/server.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable codebaseiq-pro
sudo systemctl start codebaseiq-pro
sudo systemctl status codebaseiq-pro
```

### Option 2: Docker Container

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt psutil

# Copy application
COPY src/ src/
COPY .env .

# Set production mode
ENV CODEBASEIQ_PRODUCTION=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Run server
CMD ["python", "src/codebaseiq/server.py"]
```

Build and run:
```bash
docker build -t codebaseiq-pro .
docker run -d --name codebaseiq-pro \
  -p 8080:8080 \
  -v $(pwd)/cache:/app/cache \
  --restart unless-stopped \
  codebaseiq-pro
```

### Option 3: Process Manager (PM2)

```bash
# Install PM2
npm install -g pm2

# Create ecosystem file
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'codebaseiq-pro',
    script: 'python',
    args: 'src/codebaseiq/server.py',
    cwd: '/opt/codebaseiq-pro',
    env: {
      CODEBASEIQ_PRODUCTION: 'true'
    },
    error_file: 'logs/error.log',
    out_file: 'logs/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G'
  }]
}
EOF

# Start with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

## Monitoring

### Health Checks

The production server provides three health endpoints:

1. **Health Check** - Overall system health
   ```bash
   curl http://localhost:8080/health
   ```

2. **Readiness Check** - Ready to accept requests
   ```bash
   curl http://localhost:8080/ready
   ```

3. **Liveness Check** - Server is alive
   ```bash
   curl http://localhost:8080/alive
   ```

### Metrics Monitoring

Production mode tracks:
- Request count
- Error rate
- Memory usage
- CPU usage
- Uptime

### Log Monitoring

```bash
# View logs (systemd)
journalctl -u codebaseiq-pro -f

# View logs (Docker)
docker logs -f codebaseiq-pro

# View logs (PM2)
pm2 logs codebaseiq-pro
```

### Resource Monitoring

The server automatically monitors resources and logs warnings when:
- Memory usage exceeds 1GB
- CPU usage exceeds 80%

## Security Best Practices

### 1. API Key Security
- Never commit `.env` files
- Use environment variables or secrets management
- Rotate API keys regularly

### 2. Network Security
- Run behind a reverse proxy (nginx/Apache)
- Enable TLS/SSL
- Implement rate limiting
- Use firewall rules

### 3. File System Security
- Restrict file access permissions
- Run as non-root user
- Use read-only file systems where possible

### 4. Access Control
Example nginx configuration:
```nginx
server {
    listen 443 ssl;
    server_name codebaseiq.yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Basic auth
        auth_basic "CodebaseIQ Pro";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
}
```

## Performance Tuning

### 1. Optimize Worker Count
```bash
# Set based on CPU cores
export MAX_WORKERS=$(nproc)
```

### 2. Configure Caching
```bash
# Increase cache TTL for stable codebases
export CACHE_TTL=7200  # 2 hours

# Use Redis for distributed caching
export REDIS_URL=redis://localhost:6379
```

### 3. Vector Database Optimization
- Use Pinecone for cloud deployment
- Configure appropriate index settings
- Monitor vector count and query performance

### 4. Memory Management
- Set memory limits in deployment
- Monitor for memory leaks
- Configure garbage collection

## Troubleshooting

### Common Issues

1. **Server Won't Start**
   - Check Python version: `python --version`
   - Verify all dependencies: `pip list`
   - Check logs for errors

2. **High Memory Usage**
   - Reduce MAX_WORKERS
   - Decrease embedding batch size
   - Clear old cache files

3. **Slow Performance**
   - Check vector database performance
   - Verify network connectivity to APIs
   - Monitor CPU usage

4. **API Errors**
   - Verify API keys are valid
   - Check rate limits
   - Monitor error logs

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python src/codebaseiq/server.py
```

## Backup and Recovery

### Regular Backups
```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backup/codebaseiq-pro"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup cache and state
tar -czf "$BACKUP_DIR/cache_$DATE.tar.gz" .codebaseiq_cache/
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" logs/

# Cleanup old backups (keep 7 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
```

### State Recovery
The server automatically recovers state on startup from:
- `.codebaseiq_cache/` directory
- Vector database (if persistent)

## Scaling Considerations

### Horizontal Scaling
- Use load balancer for multiple instances
- Share vector database (Pinecone)
- Use distributed cache (Redis)

### Vertical Scaling
- Increase memory for larger codebases
- Add CPU cores for parallel processing
- Use SSD storage for faster I/O

## Maintenance

### Regular Tasks
1. **Weekly**: Check logs for errors
2. **Monthly**: Update dependencies
3. **Quarterly**: Rotate API keys
4. **As needed**: Clear old cache files

### Updates
```bash
# Update CodebaseIQ Pro
git pull origin main
pip install -r requirements.txt

# Restart service
sudo systemctl restart codebaseiq-pro
```

## Support

For production support:
- Check logs first
- Run diagnostic tests
- Review this guide
- Contact: hi@chatcbd.com

---

## Quick Start Commands

```bash
# 1. Clone and setup
git clone <repository>
cd codebase_iq_pro
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your keys

# 3. Test
python tests/test_critical_manually.py

# 4. Deploy (systemd)
sudo cp codebaseiq-pro.service /etc/systemd/system/
sudo systemctl enable codebaseiq-pro
sudo systemctl start codebaseiq-pro

# 5. Verify
curl http://localhost:8080/health
```

---

🚀 **Production Checklist**
- [ ] Environment variables configured
- [ ] Tests passing
- [ ] Monitoring setup
- [ ] Backups configured
- [ ] Security measures in place
- [ ] Documentation updated
- [ ] Team trained on operations

**Remember**: Always test changes in a staging environment before deploying to production!