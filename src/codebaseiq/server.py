#!/usr/bin/env python3
"""
CodebaseIQ Pro - Advanced MCP Server with Adaptive Service Selection
Automatically uses free/local services by default, upgrades to premium when available

Features:
- Adaptive vector database: Qdrant (free) or Pinecone (premium)
- Adaptive embeddings: OpenAI (required) or Voyage AI (premium)
- Local caching with optional Redis
- Multi-agent analysis system
- Enterprise-grade security
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
import logging

# Core imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# Import our modules
try:
    from codebaseiq.core import get_config, SimpleOrchestrator, EnhancedCodeEntity, AgentRole
    from codebaseiq.services import create_vector_db, create_embedding_service, create_cache_service
    from codebaseiq.agents import (
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
    
except ImportError as e:
    print(f"⚠️  Missing required modules: {e}")
    print("Please ensure all module files are properly installed.")
    print("Try running: python -m pip install -e .")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodebaseIQProServer:
    """Enhanced MCP server with adaptive service selection"""
    
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
        
        # Setup MCP tools
        self._setup_tools()
        
        # State management
        self.current_analysis = None
        self.analysis_cache = {}
        
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
        
    def _setup_tools(self):
        """Setup MCP tools"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools"""
            return [
                types.Tool(
                    name="analyze_codebase",
                    description="Analyze a codebase with intelligent multi-agent orchestration.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the codebase directory"},
                            "analysis_type": {"type": "string", "enum": ["full", "security_focus", "performance_focus", "quick"], "default": "full"},
                            "enable_embeddings": {"type": "boolean", "default": True},
                            "focus_areas": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["path"]
                    }
                ),
                types.Tool(
                    name="semantic_code_search",
                    description="Search for code using natural language queries.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Natural language search query"},
                            "top_k": {"type": "integer", "default": 10},
                            "filters": {"type": "object"},
                            "search_type": {"type": "string", "enum": ["semantic", "keyword", "hybrid"], "default": "semantic"}
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="find_similar_code",
                    description="Find code similar to a given file or function.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_path": {"type": "string", "description": "Path to the code entity"},
                            "top_k": {"type": "integer", "default": 5},
                            "similarity_threshold": {"type": "number", "default": 0.7}
                        },
                        "required": ["entity_path"]
                    }
                ),
                types.Tool(
                    name="get_analysis_summary",
                    description="Get a summary of the current analysis results.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_danger_zones",
                    description="Get list of danger zones (high-risk code areas) from security analysis.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_dependencies",
                    description="Get dependency analysis results.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool calls"""
            
            if name == "analyze_codebase":
                result = await self._analyze_codebase(
                    path=arguments.get("path"),
                    analysis_type=arguments.get("analysis_type", "full"),
                    enable_embeddings=arguments.get("enable_embeddings", True),
                    focus_areas=arguments.get("focus_areas")
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "semantic_code_search":
                result = await self._semantic_code_search(
                    query=arguments.get("query"),
                    top_k=arguments.get("top_k", 10),
                    filters=arguments.get("filters"),
                    search_type=arguments.get("search_type", "semantic")
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "find_similar_code":
                result = await self._find_similar_code(
                    entity_path=arguments.get("entity_path"),
                    top_k=arguments.get("top_k", 5),
                    similarity_threshold=arguments.get("similarity_threshold", 0.7)
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_analysis_summary":
                result = await self._get_analysis_summary()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_danger_zones":
                result = await self._get_danger_zones()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_dependencies":
                result = await self._get_dependencies()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            else:
                raise ValueError(f"Unknown tool: {name}")
        
    async def _analyze_codebase(
            self,
            path: str,
            analysis_type: str = "full",
            enable_embeddings: bool = True,
            focus_areas: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """Internal method to analyze a codebase"""
            try:
                root_path = Path(path).resolve()
                if not root_path.exists():
                    return {'error': f'Path does not exist: {path}'}
                    
                if not root_path.is_dir():
                    return {'error': f'Path is not a directory: {path}'}
                    
                logger.info(f"Starting {analysis_type} analysis of {root_path}")
                
                # Check cache
                cache_key = f"{root_path}:{analysis_type}:{enable_embeddings}"
                if self.cache:
                    cached_result = await self.cache.get(cache_key)
                    if cached_result:
                        logger.info("Returning cached analysis result")
                        return cached_result
                        
                # Discover files
                file_map = await self._discover_files(root_path)
                
                # Create initial entities
                entities = {}
                for rel_path in file_map:
                    entities[rel_path] = EnhancedCodeEntity(
                        path=rel_path,
                        type=self._determine_entity_type(rel_path),
                        name=Path(rel_path).stem
                    )
                    
                # Prepare context
                context = {
                    'root_path': root_path,
                    'file_map': file_map,
                    'entities': entities,
                    'enable_embeddings': enable_embeddings and self.vector_db is not None,
                    'focus_areas': focus_areas or []
                }
                
                # Execute orchestrated analysis
                results = await self.orchestrator.execute(context, analysis_type)
                
                # Store in cache
                if self.cache:
                    await self.cache.set(cache_key, results, ttl=3600)  # 1 hour
                    
                # Store current analysis
                self.current_analysis = results
                
                return {
                    'status': 'success',
                    'path': str(root_path),
                    'files_analyzed': len(file_map),
                    'analysis_type': analysis_type,
                    'results': results,
                    'config': self.config.get_config_summary()
                }
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}", exc_info=True)
                return {
                    'status': 'error',
                    'error': str(e),
                    'path': path
                }
                
    async def _semantic_code_search(
            self,
            query: str,
            top_k: int = 10,
            filters: Optional[Dict[str, Any]] = None,
            search_type: str = "semantic"
        ) -> Dict[str, Any]:
            """Internal method to search for code using natural language queries"""
            if not self.vector_db:
                return {
                    'error': 'Vector search not available. Set PINECONE_API_KEY or use Qdrant.',
                    'suggestion': 'Run analyze_codebase with enable_embeddings=true first'
                }
                
            if not self.current_analysis:
                return {
                    'error': 'No codebase analyzed yet. Run analyze_codebase first.'
                }
                
            try:
                # Generate query embedding
                query_embedding = await self.embedding_service.embed_text(query)
                
                # Search vector database
                results = await self.vector_db.search(
                    query_vector=query_embedding,
                    top_k=top_k,
                    filter=filters
                )
                
                # Enhance results with code snippets
                enhanced_results = []
                for result in results:
                    metadata = result.get('metadata', {})
                    enhanced_results.append({
                        'path': metadata.get('path', ''),
                        'type': metadata.get('type', ''),
                        'name': metadata.get('name', ''),
                        'score': result.get('score', 0),
                        'importance': metadata.get('importance', 0),
                        'danger_level': metadata.get('danger_level', 0)
                    })
                    
                return {
                    'query': query,
                    'results': enhanced_results,
                    'total_results': len(enhanced_results),
                    'search_type': search_type,
                    'embedding_model': self.config.embedding_config['model']
                }
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return {
                    'error': str(e),
                    'query': query
                }
                
    async def _find_similar_code(
            self,
            entity_path: str,
            top_k: int = 5,
            similarity_threshold: float = 0.7
        ) -> Dict[str, Any]:
            """Internal method to find code similar to a given file or function"""
            if not self.vector_db:
                return {
                    'error': 'Vector search not available.',
                    'suggestion': 'Enable vector database for similarity search'
                }
                
            if not self.current_analysis:
                return {
                    'error': 'No codebase analyzed yet. Run analyze_codebase first.'
                }
                
            try:
                # Find entity
                entities = self.current_analysis.get('agent_results', {}).get('entities', {})
                entity = entities.get(entity_path)
                
                if not entity or not hasattr(entity, 'embedding') or entity.embedding is None:
                    return {
                        'error': f'Entity not found or has no embedding: {entity_path}'
                    }
                    
                # Search for similar
                results = await self.vector_db.search(
                    query_vector=entity.embedding,
                    top_k=top_k + 1  # +1 to exclude self
                )
                
                # Filter out self and apply threshold
                similar = []
                for result in results:
                    metadata = result.get('metadata', {})
                    if metadata.get('path') != entity_path and result.get('score', 0) >= similarity_threshold:
                        similar.append({
                            'path': metadata.get('path', ''),
                            'type': metadata.get('type', ''),
                            'name': metadata.get('name', ''),
                            'similarity_score': result.get('score', 0)
                        })
                        
                return {
                    'entity_path': entity_path,
                    'similar_entities': similar[:top_k],
                    'threshold': similarity_threshold
                }
                
            except Exception as e:
                logger.error(f"Similarity search failed: {e}")
                return {
                    'error': str(e),
                    'entity_path': entity_path
                }
                
    async def _get_analysis_summary(self) -> Dict[str, Any]:
            """Internal method to get a summary of the current analysis results"""
            if not self.current_analysis:
                return {
                    'error': 'No analysis performed yet. Run analyze_codebase first.'
                }
                
            summary = self.current_analysis.get('summary', {})
            
            # Add configuration info
            summary['configuration'] = {
                'vector_db': self.config.vector_db_config['type'],
                'embedding_service': self.config.embedding_config['service'],
                'premium_features': self.config.get_config_summary()['premium_features']
            }
            
            return summary
            
    async def _get_danger_zones(self) -> Dict[str, Any]:
            """Internal method to get list of danger zones from security analysis"""
            if not self.current_analysis:
                return {
                    'error': 'No analysis performed yet. Run analyze_codebase first.'
                }
                
            security_results = self.current_analysis.get('agent_results', {}).get('security', {})
            danger_zones = security_results.get('danger_zones', [])
            
            return {
                'danger_zones': danger_zones,
                'total_count': len(danger_zones),
                'security_score': security_results.get('security_score', 'N/A')
            }
            
    async def _get_dependencies(self) -> Dict[str, Any]:
            """Internal method to get dependency analysis results"""
            if not self.current_analysis:
                return {
                    'error': 'No analysis performed yet. Run analyze_codebase first.'
                }
                
            dep_results = self.current_analysis.get('agent_results', {}).get('dependency', {})
            
            return {
                'external_dependencies': dep_results.get('external_dependencies', {}),
                'package_managers': dep_results.get('package_managers', []),
                'dependency_graph': dep_results.get('dependency_graph', {}),
                'internal_dependencies': dep_results.get('internal_dependencies', 0)
            }
            
    async def _discover_files(self, root: Path) -> Dict[str, Path]:
        """Enhanced file discovery with better filtering"""
        file_map = {}
        ignore_patterns = {
            '.git', 'node_modules', '__pycache__', 'dist', 'build', 
            '.next', 'target', 'out', '.cache', 'coverage', 'venv',
            '.pytest_cache', '.mypy_cache', '.tox', '.eggs'
        }
        
        # Walk directory tree efficiently
        for path in root.rglob('*'):
            # Skip if any parent directory is in ignore list
            if any(ignored in path.parts for ignored in ignore_patterns):
                continue
                
            if path.is_file():
                # Check file size
                try:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    if size_mb > self.config.performance_config['max_file_size_mb']:
                        continue
                except:
                    continue
                    
                # Check if it's a code file
                if path.suffix in {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', 
                                 '.go', '.rs', '.cpp', '.c', '.h', '.hpp',
                                 '.rb', '.php', '.swift', '.kt', '.scala', '.cs'}:
                    rel_path = str(path.relative_to(root))
                    file_map[rel_path] = path
                    
        logger.info(f"Discovered {len(file_map)} code files")
        return file_map
        
    def _determine_entity_type(self, path: str) -> str:
        """Determine entity type from file path"""
        path_lower = path.lower()
        
        if 'test' in path_lower:
            return 'test'
        elif 'component' in path_lower:
            return 'component'
        elif 'service' in path_lower:
            return 'service'
        elif 'model' in path_lower:
            return 'model'
        elif 'util' in path_lower or 'helper' in path_lower:
            return 'utility'
        elif 'config' in path_lower:
            return 'config'
        else:
            return 'module'
            
    async def _initialize_async_services(self):
        """Initialize async services like vector database"""
        if self.vector_db:
            try:
                await self.vector_db.initialize()
                logger.info(f"✅ Vector database initialized successfully")
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




# Entry point
if __name__ == "__main__":
    try:
        server = CodebaseIQProServer()
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n👋 CodebaseIQ Pro Server stopped")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        sys.exit(1)