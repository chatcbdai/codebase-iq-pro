#!/usr/bin/env python3
"""
Base server functionality for CodebaseIQ Pro
Contains initialization, service setup, and core server operations
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Core imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# Import our modules
from .core import (
    get_config, SimpleOrchestrator, EnhancedCodeEntity, AgentRole,
    TokenManager, TokenBudget, CacheManager
)
from .services import create_vector_db, create_embedding_service, create_cache_service
from .agents import (
    DependencyAnalysisAgent,
    SecurityAuditAgent,
    PatternDetectionAgent,
    VersionCompatibilityAgent,
    ArchitectureAnalysisAgent,
    PerformanceAnalysisAgent,
    EmbeddingAgent,
    DocumentationAgent,
    TestCoverageAgent
)

logger = logging.getLogger(__name__)


class CodebaseIQProServerBase:
    """Base class with initialization and core server functionality"""
    
    def __init__(self):
        self.server = Server("codebase-iq-pro")
        
        # Initialize adaptive configuration
        try:
            self.config = get_config()
            logger.info("✅ Adaptive configuration initialized")
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            print("\n🔑 Required: Set OPENAI_API_KEY environment variable")
            print("📦 Optional: Set VOYAGE_API_KEY for premium embeddings")
            print("📦 Optional: Set PINECONE_API_KEY for premium vector database")
            sys.exit(1)
            
        # Initialize services based on configuration
        self._initialize_services()
        
        # Initialize orchestrator and agents
        self._initialize_orchestrator()
        
        # State management
        self.current_analysis = None
        self.analysis_cache = {}
        
        # Initialize token and cache managers
        self.token_manager = TokenManager()
        self.cache_manager = CacheManager()
        
        # Flag for state restoration
        self._needs_state_restoration = True
        
    def _initialize_services(self):
        """Initialize all services based on configuration"""
        # Vector Database
        try:
            self.vector_db = create_vector_db(self.config.vector_db_config)
            # Will initialize later in async context
            logger.info(f"✅ Vector database created: {self.config.vector_db_config['type']}")
        except Exception as e:
            logger.error(f"Failed to create vector database: {e}")
            self.vector_db = None
            
        # Embedding Service
        try:
            self.embedding_service = create_embedding_service(self.config.embedding_config)
            logger.info(f"✅ Embedding service initialized: {self.config.embedding_config['service']}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise  # Embedding service is required
            
        # Cache Service
        try:
            self.cache = create_cache_service(self.config.cache_config)
            logger.info(f"✅ Cache service initialized: {self.config.cache_config['type']}")
        except Exception as e:
            logger.warning(f"Failed to initialize cache service: {e}")
            self.cache = None
            
    def _initialize_orchestrator(self):
        """Initialize the orchestrator and register all agents"""
        self.orchestrator = SimpleOrchestrator()
        
        # Register core analysis agents
        self.orchestrator.register_agent(DependencyAnalysisAgent())
        self.orchestrator.register_agent(SecurityAuditAgent())
        self.orchestrator.register_agent(PatternDetectionAgent())
        self.orchestrator.register_agent(VersionCompatibilityAgent())
        self.orchestrator.register_agent(ArchitectureAnalysisAgent())
        self.orchestrator.register_agent(PerformanceAnalysisAgent())
        
        # Register documentation and test agents
        self.orchestrator.register_agent(DocumentationAgent())
        self.orchestrator.register_agent(TestCoverageAgent())
        
        # Register embedding agent if vector DB is available
        if self.vector_db:
            self.orchestrator.register_agent(
                EmbeddingAgent(self.vector_db, self.embedding_service)
            )
            
        logger.info(f"✅ Registered {len(self.orchestrator.agents)} analysis agents")
        
    async def _auto_restore_state(self):
        """Automatically restore state from persistence on startup"""
        if not self._needs_state_restoration:
            return
            
        try:
            logger.info("🔄 Checking for persisted state to restore...")
            start_time = datetime.now()
            
            # Check cache directory
            cache_dir = Path.home() / ".codebaseiq" / "cache"
            if not cache_dir.exists():
                logger.info("No cached state found - fresh start")
                return
                
            # Look for recent analysis files
            codebase_path = Path(os.getcwd())
            restored_count = 0
            
            # Analysis types to restore
            analysis_types = [
                'dependency', 'security', 'architecture', 
                'business_logic', 'technical_stack', 'code_intelligence'
            ]
            
            for analysis_type in analysis_types:
                cache_file = cache_dir / f"{codebase_path.name}_{analysis_type}.json"
                if cache_file.exists():
                    # Check if cache is recent (within last 7 days)
                    cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if cache_age.days <= 7:
                        try:
                            # Load the cached analysis
                            cached_data = await self.cache_manager.load_analysis(codebase_path, analysis_type)
                            if cached_data:
                                # Store in memory cache
                                cache_key = f"{codebase_path}_{analysis_type}"
                                self.analysis_cache[cache_key] = cached_data['analysis']
                                restored_count += 1
                                logger.info(f"✅ Restored {analysis_type} analysis from cache")
                        except Exception as e:
                            logger.warning(f"Failed to restore {analysis_type}: {e}")
                            
            # Restore current_analysis if available
            if restored_count > 0:
                # Try to load the most recent comprehensive analysis
                comprehensive_cache = cache_dir / f"{codebase_path.name}_comprehensive.json"
                if comprehensive_cache.exists():
                    try:
                        cached_data = await self.cache_manager.load_analysis(codebase_path, "comprehensive")
                        if cached_data:
                            self.current_analysis = cached_data['analysis']
                            logger.info("✅ Restored comprehensive analysis")
                    except Exception as e:
                        logger.warning(f"Failed to restore comprehensive analysis: {e}")
                        
            # Initialize vector DB if embeddings exist
            if self.vector_db:
                try:
                    await self.vector_db.initialize()
                    stats = await self.vector_db.get_stats()
                    if stats and stats.get('total_vectors', 0) > 0:
                        logger.info(f"✅ Vector database ready with {stats['total_vectors']} embeddings")
                except Exception as e:
                    logger.warning(f"Failed to initialize vector DB: {e}")
                    
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"✨ State restoration completed in {elapsed:.2f}s - restored {restored_count} analyses")
            
            # Clear the flag
            self._needs_state_restoration = False
            
        except Exception as e:
            logger.error(f"State restoration failed: {e}", exc_info=True)
            # Continue anyway - don't let restoration failure stop the server
            self._needs_state_restoration = False
            
    async def _initialize_async_services(self):
        """Initialize async services like vector DB"""
        if self.vector_db:
            try:
                await self.vector_db.initialize()
                logger.info("✅ Vector database initialized")
            except Exception as e:
                logger.error(f"Failed to initialize vector database: {e}")
                self.vector_db = None
                
    async def run(self):
        """Start the MCP server"""
        print("🚀 CodebaseIQ Pro Server starting...")
        print("✨ Adaptive configuration loaded:")
        
        config_summary = self.config.get_config_summary()
        print(f"  • Vector DB: {config_summary['vector_db']['type']} ({config_summary['vector_db']['tier']})")
        print(f"  • Embeddings: {config_summary['embeddings']['service']} ({config_summary['embeddings']['tier']})")
        print(f"  • Cache: {config_summary['cache']['type']}")
        print(f"  • Workers: {config_summary['performance']['max_workers']}")
        
        if any(config_summary['premium_features'].values()):
            print("\n✨ Premium features enabled:")
            for feature, enabled in config_summary['premium_features'].items():
                if enabled:
                    print(f"  • {feature.replace('_', ' ').title()}")
                    
        print("\n📡 Ready to analyze codebases!")
        
        # Initialize async services
        await self._initialize_async_services()
        
        # Restore state from persistence
        await self._auto_restore_state()
        
        from mcp.server.models import InitializationOptions
        from mcp.server.lowlevel.server import NotificationOptions
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="codebase-iq-pro",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )