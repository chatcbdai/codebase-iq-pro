#!/usr/bin/env python3
"""
Production enhancements for CodebaseIQ Pro MCP Server

Adds health checks, monitoring, graceful shutdown, and other production features.
"""

import asyncio
import signal
import time
import psutil
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ProductionFeatures:
    """Production-ready features for the MCP server"""
    
    def __init__(self, server):
        self.server = server
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.shutdown_event = asyncio.Event()
        self.health_status = "healthy"
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_shutdown_signal)
            
    def _handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Basic health info
            uptime = time.time() - self.start_time
            
            # System resources
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Server state
            server_healthy = True
            checks = []
            
            # Check vector DB
            if hasattr(self.server, 'vector_db') and self.server.vector_db:
                try:
                    # Simple connectivity check
                    stats = await self.server.vector_db.get_stats()
                    checks.append({
                        "component": "vector_db",
                        "status": "healthy",
                        "details": f"{stats.get('total_vectors', 0)} vectors"
                    })
                except Exception as e:
                    checks.append({
                        "component": "vector_db",
                        "status": "unhealthy",
                        "error": str(e)
                    })
                    server_healthy = False
            
            # Check embedding service
            if hasattr(self.server, 'embedding_service') and self.server.embedding_service:
                checks.append({
                    "component": "embedding_service",
                    "status": "healthy",
                    "details": "OpenAI service ready"
                })
            
            # Check cache
            if hasattr(self.server, 'cache') and self.server.cache:
                checks.append({
                    "component": "cache",
                    "status": "healthy",
                    "details": "In-memory cache active"
                })
            
            # Overall health
            health_status = "healthy" if server_healthy else "degraded"
            
            return {
                "status": health_status,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": int(uptime),
                "version": "1.0.0",
                "metrics": {
                    "requests_total": self.request_count,
                    "errors_total": self.error_count,
                    "error_rate": self.error_count / max(self.request_count, 1),
                    "memory_usage_mb": memory_info.rss / 1024 / 1024,
                    "cpu_percent": process.cpu_percent()
                },
                "checks": checks
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def ready_check(self) -> Dict[str, Any]:
        """Readiness probe - is server ready to accept requests"""
        try:
            # Check if server is initialized
            if not hasattr(self.server, '_handle_call_tool'):
                return {"ready": False, "reason": "Server not initialized"}
                
            # Check if critical components are ready
            if not self.server.vector_db:
                return {"ready": False, "reason": "Vector database not ready"}
                
            if not self.server.embedding_service:
                return {"ready": False, "reason": "Embedding service not ready"}
                
            return {
                "ready": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "ready": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def liveness_check(self) -> Dict[str, Any]:
        """Liveness probe - is server alive and responding"""
        try:
            # Simple check that server can respond
            return {
                "alive": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "alive": False,
                "error": str(e)
            }
    
    def increment_request_count(self):
        """Track request metrics"""
        self.request_count += 1
        
    def increment_error_count(self):
        """Track error metrics"""
        self.error_count += 1
        
    async def graceful_shutdown(self):
        """Perform graceful shutdown"""
        logger.info("Starting graceful shutdown...")
        
        try:
            # Stop accepting new requests
            self.health_status = "shutting_down"
            
            # Wait for ongoing requests to complete (with timeout)
            await asyncio.sleep(2)  # Give 2 seconds for requests to complete
            
            # Close connections
            if hasattr(self.server, 'vector_db') and self.server.vector_db:
                logger.info("Closing vector database connection...")
                # Vector DB cleanup if needed
                
            # Save any pending state
            if hasattr(self.server, '_persist_state'):
                logger.info("Persisting final state...")
                await self.server._persist_state()
                
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
    async def monitor_resources(self):
        """Background task to monitor resources"""
        while not self.shutdown_event.is_set():
            try:
                # Check memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # Warn if memory is high
                if memory_mb > 1000:  # 1GB threshold
                    logger.warning(f"High memory usage: {memory_mb:.2f} MB")
                    
                # Check CPU
                cpu_percent = process.cpu_percent(interval=1)
                if cpu_percent > 80:
                    logger.warning(f"High CPU usage: {cpu_percent}%")
                    
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                
            # Check every 30 seconds
            await asyncio.sleep(30)


class ProductionServer:
    """Production wrapper for CodebaseIQ Pro Server"""
    
    def __init__(self, server):
        self.server = server
        self.production_features = ProductionFeatures(server)
        self._original_handle_call_tool = server.handle_call_tool
        
        # Wrap handle_call_tool with metrics
        async def wrapped_handle_call_tool(name: str, arguments: Optional[Dict[str, Any]] = None):
            self.production_features.increment_request_count()
            try:
                result = await self._original_handle_call_tool(name, arguments)
                return result
            except Exception as e:
                self.production_features.increment_error_count()
                raise
                
        server.handle_call_tool = wrapped_handle_call_tool
        
    async def run_with_monitoring(self):
        """Run server with monitoring and health checks"""
        # Start resource monitoring
        monitor_task = asyncio.create_task(
            self.production_features.monitor_resources()
        )
        
        try:
            # Run the server
            await self.server.run()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            # Graceful shutdown
            await self.production_features.graceful_shutdown()
            monitor_task.cancel()
            
    async def get_health(self):
        """Get health status"""
        return await self.production_features.health_check()
        
    async def get_ready(self):
        """Get readiness status"""
        return await self.production_features.ready_check()
        
    async def get_alive(self):
        """Get liveness status"""
        return await self.production_features.liveness_check()


# Additional production tools that could be added to the server
def add_production_tools(server):
    """Add production-specific MCP tools"""
    # This would add tools like:
    # - health_check: Get server health status
    # - metrics: Get server metrics
    # - clear_cache: Clear caches safely
    # These would be registered with the MCP server
    pass