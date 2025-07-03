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
import networkx as nx
import aiofiles

# Core imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# Import our modules
try:
    from codebaseiq.core import (
        get_config, SimpleOrchestrator, EnhancedCodeEntity, AgentRole,
        TokenManager, TokenBudget, CacheManager
    )
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
    # Import enhanced understanding agents
    from codebaseiq.agents.deep_understanding_agent import DeepUnderstandingAgent
    from codebaseiq.agents.cross_file_intelligence import CrossFileIntelligence
    from codebaseiq.agents.business_logic_extractor import BusinessLogicExtractor
    from codebaseiq.agents.ai_knowledge_packager import AIKnowledgePackager
    
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
        
        # Initialize token and cache managers
        self.token_manager = TokenManager()
        self.cache_manager = CacheManager()
        
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
                    name="get_codebase_context",
                    description="Get essential codebase context for safe modifications. Returns danger zones, impact analysis, and business understanding.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "refresh": {"type": "boolean", "default": False, "description": "Force refresh of cached analysis"}
                        }
                    }
                ),
                types.Tool(
                    name="check_understanding",
                    description="Verify your understanding of the codebase and get approval score before code implementation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "implementation_plan": {"type": "string", "description": "Describe what you plan to implement and why"},
                            "files_to_modify": {"type": "array", "items": {"type": "string"}, "description": "List of files you plan to modify"},
                            "understanding_points": {"type": "array", "items": {"type": "string"}, "description": "Key points showing your understanding"}
                        },
                        "required": ["implementation_plan"]
                    }
                ),
                types.Tool(
                    name="get_impact_analysis",
                    description="Get detailed impact analysis for a specific file before modification.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the file to analyze"}
                        },
                        "required": ["file_path"]
                    }
                ),
                types.Tool(
                    name="get_and_set_the_codebase_knowledge_foundation",
                    description="Run all 4 phases of analysis to establish complete codebase knowledge foundation. Runs 25K Gold tools, CIA tools, Crossing Guards, and Premium Embedders in optimal order.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the codebase directory", "default": "."},
                            "enable_embeddings": {"type": "boolean", "default": True, "description": "Enable semantic search capabilities"},
                            "force_refresh": {"type": "boolean", "default": False, "description": "Force fresh analysis ignoring cache"}
                        }
                    }
                ),
                types.Tool(
                    name="update_cached_knowledge_foundation",
                    description="Check if codebase has changed since last analysis and update if needed. Compares cache timestamp with latest git commit.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the codebase directory", "default": "."}
                        }
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
                ),
                types.Tool(
                    name="get_ai_knowledge_package",
                    description="Get comprehensive AI knowledge package with danger zones, business context, and modification guidance.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_business_context",
                    description="Get business logic understanding including domain model, user journeys, and business rules.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_modification_guidance",
                    description="Get specific guidance for safely modifying files, including impact analysis and risk assessment.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the file you want to modify"}
                        }
                    }
                ),
                # New individual analysis tools (25K tokens each)
                types.Tool(
                    name="get_dependency_analysis",
                    description="Get comprehensive dependency analysis with full dependency graphs, import chains, and package dependencies. Returns up to 25K tokens of detailed data.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force_refresh": {"type": "boolean", "default": False, "description": "Force fresh analysis ignoring cache"},
                            "include_transitive": {"type": "boolean", "default": True, "description": "Include transitive dependencies"}
                        }
                    }
                ),
                types.Tool(
                    name="get_security_analysis",
                    description="Get detailed security analysis including all vulnerabilities, auth mechanisms, and security patterns. Returns up to 25K tokens of security data.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force_refresh": {"type": "boolean", "default": False, "description": "Force fresh analysis ignoring cache"},
                            "severity_filter": {"type": "string", "enum": ["all", "critical", "high", "medium", "low"], "default": "all"}
                        }
                    }
                ),
                types.Tool(
                    name="get_architecture_analysis",
                    description="Get complete architecture analysis with layers, components, patterns, and structural details. Returns up to 25K tokens of architecture data.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force_refresh": {"type": "boolean", "default": False, "description": "Force fresh analysis ignoring cache"},
                            "include_diagrams": {"type": "boolean", "default": True, "description": "Include ASCII architecture diagrams"}
                        }
                    }
                ),
                types.Tool(
                    name="get_business_logic_analysis",
                    description="Get comprehensive business logic analysis with domain models, workflows, and business rules. Returns up to 25K tokens of business data.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force_refresh": {"type": "boolean", "default": False, "description": "Force fresh analysis ignoring cache"},
                            "include_workflows": {"type": "boolean", "default": True, "description": "Include detailed workflow analysis"}
                        }
                    }
                ),
                types.Tool(
                    name="get_technical_stack_analysis",
                    description="Get detailed technical stack analysis with frameworks, versions, build tools, and configurations. Returns up to 25K tokens of technical data.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force_refresh": {"type": "boolean", "default": False, "description": "Force fresh analysis ignoring cache"},
                            "include_configs": {"type": "boolean", "default": True, "description": "Include configuration details"}
                        }
                    }
                ),
                types.Tool(
                    name="get_code_intelligence_analysis",
                    description="Get comprehensive code intelligence with entry points, API surface, service registry, and patterns. Returns up to 25K tokens of code intelligence data.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force_refresh": {"type": "boolean", "default": False, "description": "Force fresh analysis ignoring cache"},
                            "include_patterns": {"type": "boolean", "default": True, "description": "Include design pattern analysis"}
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool calls"""
            
            if name == "get_codebase_context":
                result = await self._get_codebase_context(
                    refresh=arguments.get("refresh", False)
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "check_understanding":
                result = await self._check_understanding(
                    implementation_plan=arguments.get("implementation_plan"),
                    files_to_modify=arguments.get("files_to_modify", []),
                    understanding_points=arguments.get("understanding_points", [])
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_impact_analysis":
                result = await self._get_impact_analysis(
                    file_path=arguments.get("file_path")
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_and_set_the_codebase_knowledge_foundation":
                result = await self._get_and_set_knowledge_foundation(
                    path=arguments.get("path", "."),
                    enable_embeddings=arguments.get("enable_embeddings", True),
                    force_refresh=arguments.get("force_refresh", False)
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "update_cached_knowledge_foundation":
                result = await self._update_cached_knowledge_foundation(
                    path=arguments.get("path", ".")
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
                
            elif name == "get_ai_knowledge_package":
                result = await self._get_ai_knowledge_package()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_business_context":
                result = await self._get_business_context()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_modification_guidance":
                file_path = arguments.get("file_path")
                result = await self._get_modification_guidance(file_path)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            # New individual analysis tools
            elif name == "get_dependency_analysis":
                result = await self._get_dependency_analysis_full(
                    force_refresh=arguments.get("force_refresh", False),
                    include_transitive=arguments.get("include_transitive", True)
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_security_analysis":
                result = await self._get_security_analysis_full(
                    force_refresh=arguments.get("force_refresh", False),
                    severity_filter=arguments.get("severity_filter", "all")
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_architecture_analysis":
                result = await self._get_architecture_analysis_full(
                    force_refresh=arguments.get("force_refresh", False),
                    include_diagrams=arguments.get("include_diagrams", True)
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_business_logic_analysis":
                result = await self._get_business_logic_analysis_full(
                    force_refresh=arguments.get("force_refresh", False),
                    include_workflows=arguments.get("include_workflows", True)
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_technical_stack_analysis":
                result = await self._get_technical_stack_analysis_full(
                    force_refresh=arguments.get("force_refresh", False),
                    include_configs=arguments.get("include_configs", True)
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "get_code_intelligence_analysis":
                result = await self._get_code_intelligence_analysis_full(
                    force_refresh=arguments.get("force_refresh", False),
                    include_patterns=arguments.get("include_patterns", True)
                )
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            else:
                raise ValueError(f"Unknown tool: {name}")
        
    async def _analyze_codebase(
            self,
            path: str,
            analysis_type: str = "full",
            enable_embeddings: bool = True,
            focus_areas: Optional[List[str]] = None,
            force_refresh: bool = False
        ) -> Dict[str, Any]:
            """Internal method to analyze a codebase"""
            try:
                root_path = Path(path).resolve()
                if not root_path.exists():
                    return {'error': f'Path does not exist: {path}'}
                    
                if not root_path.is_dir():
                    return {'error': f'Path is not a directory: {path}'}
                    
                logger.info(f"Starting {analysis_type} analysis of {root_path}")
                
                # Check cache only if not forcing refresh
                cache_key = f"{root_path}:{analysis_type}:{enable_embeddings}"
                if self.cache and not force_refresh:
                    cached_result = await self.cache.get(cache_key)
                    if cached_result:
                        logger.info("Returning cached analysis result")
                        return cached_result
                elif force_refresh:
                    logger.info("Force refresh requested, bypassing cache")
                    # Clear any existing analysis
                    self.current_analysis = None
                        
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
                
                # DEPRECATED: This method should not run any agents
                return {
                    'error': 'analyze_codebase is deprecated',
                    'message': 'Please use get_and_set_the_codebase_knowledge_foundation instead',
                    'hint': 'The new tool runs all analysis phases in optimal order without token limit issues'
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
                    'suggestion': 'Run get_and_set_the_codebase_knowledge_foundation with enable_embeddings=true first'
                }
                
            if not self.current_analysis:
                return {
                    'error': 'No codebase analyzed yet. Run get_and_set_the_codebase_knowledge_foundation first.'
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
                    'error': 'No codebase analyzed yet. Run get_and_set_the_codebase_knowledge_foundation first.'
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
                    'error': 'No analysis performed yet. Run get_and_set_the_codebase_knowledge_foundation first.'
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
                    'error': 'No analysis performed yet. Run get_and_set_the_codebase_knowledge_foundation first.'
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
                    'error': 'No analysis performed yet. Run get_and_set_the_codebase_knowledge_foundation first.'
                }
                
            dep_results = self.current_analysis.get('agent_results', {}).get('dependency', {})
            
            return {
                'external_dependencies': dep_results.get('external_dependencies', {}),
                'package_managers': dep_results.get('package_managers', []),
                'dependency_graph': dep_results.get('dependency_graph', {}),
                'internal_dependencies': dep_results.get('internal_dependencies', 0)
            }
            
    async def _get_ai_knowledge_package(self) -> Dict[str, Any]:
        """Get comprehensive AI knowledge package for immediate understanding"""
        if not self.current_analysis:
            return {
                'error': 'No analysis performed yet. Run get_and_set_the_codebase_knowledge_foundation first.',
                'hint': 'Use get_codebase_context instead for optimized access'
            }
            
        enhanced = self.current_analysis.get('enhanced_understanding', {})
        if not enhanced:
            return {
                'error': 'Enhanced understanding not available. Run get_and_set_the_codebase_knowledge_foundation first.'
            }
            
        ai_package = enhanced.get('ai_knowledge_package', {})
        
        # Return optimized version - direct users to get_codebase_context
        return {
            'notice': '⚠️ This method returns large responses. Use get_codebase_context for optimized access.',
            'instant_context': ai_package.get('instant_context', ''),
            'danger_zones_summary': self._summarize_danger_zones(ai_package.get('danger_zones', {})),
            'golden_rules': ai_package.get('safe_modification_guide', {}).get('golden_rules', []),
            'quick_reference': ai_package.get('quick_reference', {}),
            'modification_checklist': ai_package.get('modification_checklist', [])[:10],
            'recommended_tool': 'get_codebase_context',
            'usage_hint': "Use get_codebase_context first, then check_understanding before ANY modification"
        }
        
    async def _get_business_context(self) -> Dict[str, Any]:
        """Get business logic understanding for the codebase"""
        if not self.current_analysis:
            return {
                'error': 'No analysis performed yet. Run get_and_set_the_codebase_knowledge_foundation first.'
            }
            
        enhanced = self.current_analysis.get('enhanced_understanding', {})
        if not enhanced:
            return {
                'error': 'Enhanced understanding not available. Run get_and_set_the_codebase_knowledge_foundation first.'
            }
            
        business_logic = enhanced.get('business_logic', {})
        
        # Optimize response size
        domain_model = business_logic.get('domain_model', {})
        entities = domain_model.get('entities', {})
        
        # Summarize entities instead of full details
        entity_summary = {}
        for name, data in list(entities.items())[:20]:  # Top 20 entities
            entity_summary[name] = {
                'purpose': data.get('business_purpose', '')[:100],
                'importance': data.get('importance', 0)
            }
        
        return {
            'executive_summary': business_logic.get('executive_summary', ''),
            'domain_entities_count': len(entities),
            'domain_entities_sample': entity_summary,
            'user_journeys': [
                {
                    'type': j.get('journey_type', ''),
                    'description': j.get('description', '')[:100],
                    'complexity': j.get('complexity', '')
                }
                for j in business_logic.get('user_journeys', [])[:10]
            ],
            'business_rules': [
                {
                    'rule': r.get('rule', '')[:100],
                    'impact': r.get('business_impact', '')
                }
                for r in business_logic.get('business_rules', [])
                if 'HIGH' in r.get('business_impact', '')
            ][:15],
            'key_features': business_logic.get('key_features', [])[:20],
            'compliance_requirements': business_logic.get('compliance_requirements', [])[:10],
            'immediate_context': business_logic.get('immediate_context', ''),
            'full_details_hint': 'Use get_codebase_context for optimized access to all details'
        }
        
    async def _get_modification_guidance(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Get specific guidance for safely modifying files"""
        if not self.current_analysis:
            return {
                'error': 'No analysis performed yet. Run get_and_set_the_codebase_knowledge_foundation first.'
            }
            
        enhanced = self.current_analysis.get('enhanced_understanding', {})
        if not enhanced:
            return {
                'error': 'Enhanced understanding not available. Run get_and_set_the_codebase_knowledge_foundation first.'
            }
            
        cross_file_intel = enhanced.get('cross_file_intelligence', {})
        ai_package = enhanced.get('ai_knowledge_package', {})
        
        if file_path:
            # Get specific guidance for a file
            impact_zones = cross_file_intel.get('impact_zones', {})
            file_impact = impact_zones.get(file_path, {})
            
            if not file_impact:
                return {
                    'file_path': file_path,
                    'status': 'not_analyzed',
                    'guidance': 'File not found in analysis. It may be safe to modify, but check manually.',
                    'general_rules': ai_package.get('safe_modification_guide', {}).get('golden_rules', [])
                }
                
            # Get deep understanding for this file
            deep_analysis = enhanced.get('deep_analysis', {})
            file_contexts = deep_analysis.get('file_contexts', {})
            file_context = file_contexts.get(file_path, {})
            
            return {
                'file_path': file_path,
                'risk_level': file_impact.get('risk_level', 'UNKNOWN'),
                'impact_summary': f"Modifying this file will affect {file_impact.get('total_impact', 0)} other files",
                'direct_dependents': file_impact.get('direct_impact', []),
                'indirect_dependents': file_impact.get('indirect_impact', []),
                'ai_warning': file_impact.get('ai_warning', ''),
                'modification_strategy': file_impact.get('modification_strategy', ''),
                'file_purpose': file_context.get('purpose', 'Unknown'),
                'business_logic': file_context.get('business_logic', ''),
                'critical_functions': file_context.get('critical_functions', []),
                'checklist': self._get_file_specific_checklist(file_path, file_impact, file_context),
                'safer_alternatives': self._suggest_safer_alternatives(file_path, file_impact)
            }
        else:
            # Return general modification guidance
            return {
                'general_guidance': ai_package.get('safe_modification_guide', {}),
                'danger_zones': ai_package.get('danger_zones', {}),
                'modification_workflow': ai_package.get('safe_modification_guide', {}).get('modification_workflow', []),
                'testing_requirements': ai_package.get('testing_requirements', {}),
                'hint': "Provide a file_path parameter to get specific guidance for that file"
            }
            
    def _get_file_specific_checklist(self, file_path: str, impact: Dict[str, Any], 
                                    context: Dict[str, Any]) -> List[str]:
        """Generate file-specific modification checklist"""
        checklist = []
        
        # Base checklist
        checklist.extend([
            f"□ Confirmed this file ({file_path}) is not in danger_zones",
            f"□ Reviewed {len(impact.get('direct_impact', []))} direct dependencies",
            "□ Read and understood current implementation"
        ])
        
        # Risk-specific items
        if impact.get('risk_level') == 'CRITICAL':
            checklist.extend([
                "□ ⚠️ CRITICAL FILE - Obtained explicit approval to modify",
                "□ Created comprehensive test suite BEFORE changes",
                "□ Documented every change with detailed reasoning"
            ])
        elif impact.get('risk_level') == 'HIGH':
            checklist.extend([
                "□ Followed all items in extreme_caution checklist",
                "□ Tested ALL files in impact zone"
            ])
            
        # Add specific items based on file type
        if 'auth' in file_path.lower() or 'security' in file_path.lower():
            checklist.append("□ Security review completed for authentication changes")
        if 'payment' in file_path.lower() or 'billing' in file_path.lower():
            checklist.append("□ Verified PCI compliance maintained")
        if 'api' in file_path.lower() or 'endpoint' in file_path.lower():
            checklist.append("□ API documentation updated")
            
        checklist.extend([
            "□ All tests passing",
            "□ No breaking changes introduced"
        ])
        
        return checklist
        
    def _suggest_safer_alternatives(self, file_path: str, impact: Dict[str, Any]) -> List[str]:
        """Suggest safer alternatives to direct modification"""
        alternatives = []
        
        if impact.get('risk_level') in ['CRITICAL', 'HIGH']:
            alternatives.extend([
                "Create a new function/class instead of modifying existing ones",
                "Use adapter pattern to wrap existing functionality",
                "Add optional parameters with defaults to maintain compatibility",
                "Create a new file with enhanced functionality"
            ])
            
        if impact.get('total_impact', 0) > 10:
            alternatives.append("Consider creating an abstraction layer to isolate changes")
            
        if not alternatives:
            alternatives.append("This file appears relatively safe to modify with standard precautions")
            
        return alternatives
    
    async def _run_with_timeout(self, coro, timeout_seconds: int, task_name: str) -> Any:
        """Run a coroutine with timeout and progress logging
        
        Helps prevent hanging on large codebases and provides visibility
        """
        try:
            logger.info(f"Starting {task_name}...")
            start_time = datetime.now()
            
            # Run with timeout
            result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ {task_name} completed in {elapsed:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"❌ {task_name} timed out after {timeout_seconds}s")
            return {
                'error': f'{task_name} timed out',
                'timeout': timeout_seconds,
                'suggestion': 'Try analyzing smaller portions of the codebase'
            }
        except Exception as e:
            logger.error(f"❌ {task_name} failed: {e}")
            return {
                'error': f'{task_name} failed: {str(e)}',
                'exception_type': type(e).__name__
            }
            
    async def _discover_files(self, root: Path) -> Dict[str, Path]:
        """Enhanced file discovery with better filtering and progress tracking
        
        Optimized for large codebases with proper logging and error handling
        """
        file_map = {}
        ignore_patterns = {
            '.git', 'node_modules', '__pycache__', 'dist', 'build', 
            '.next', 'target', 'out', '.cache', 'coverage', 'venv',
            '.pytest_cache', '.mypy_cache', '.tox', '.eggs', '.idea',
            '.vscode', 'env', '.env', 'bower_components', 'vendor'
        }
        
        # Track statistics
        total_files_seen = 0
        files_skipped_ignore = 0
        files_skipped_size = 0
        files_skipped_binary = 0
        start_time = datetime.now()
        
        logger.info(f"Starting file discovery in: {root}")
        logger.info(f"Ignoring patterns: {ignore_patterns}")
        
        # Walk directory tree efficiently
        try:
            # Use os.walk for better performance on large directories
            import os
            for dirpath, dirnames, filenames in os.walk(root):
                # Remove ignored directories from dirnames to prevent descending
                dirnames[:] = [d for d in dirnames if d not in ignore_patterns]
                
                # Convert to Path for consistency
                dir_path = Path(dirpath)
                
                # Skip if any parent directory is in ignore list
                if any(ignored in dir_path.parts for ignored in ignore_patterns):
                    files_skipped_ignore += len(filenames)
                    continue
                
                # Process files in this directory
                for filename in filenames:
                    total_files_seen += 1
                    
                    # Log progress every 1000 files
                    if total_files_seen % 1000 == 0:
                        logger.info(f"Progress: Scanned {total_files_seen} files, found {len(file_map)} code files")
                    
                    path = dir_path / filename
                    
                    # Check file size
                    try:
                        size_mb = path.stat().st_size / (1024 * 1024)
                        if size_mb > self.config.performance_config['max_file_size_mb']:
                            files_skipped_size += 1
                            logger.debug(f"Skipped large file ({size_mb:.1f}MB): {path}")
                            continue
                    except (OSError, IOError) as e:
                        logger.debug(f"Error accessing file {path}: {e}")
                        continue
                    
                    # Check if it's a code file
                    if path.suffix.lower() in {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', 
                                             '.go', '.rs', '.cpp', '.c', '.h', '.hpp', '.cc',
                                             '.rb', '.php', '.swift', '.kt', '.scala', '.cs',
                                             '.r', '.m', '.mm', '.vue', '.svelte'}:
                        rel_path = str(path.relative_to(root))
                        file_map[rel_path] = path
                    elif not path.suffix:
                        # Check for scripts without extensions
                        try:
                            with open(path, 'rb') as f:
                                first_line = f.readline(100)
                                if first_line.startswith(b'#!') and (b'python' in first_line or 
                                                                     b'node' in first_line or 
                                                                     b'ruby' in first_line):
                                    rel_path = str(path.relative_to(root))
                                    file_map[rel_path] = path
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"Error during file discovery: {e}")
            # Continue with what we found so far
            
        # Calculate discovery time
        discovery_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"File discovery completed in {discovery_time:.2f} seconds")
        logger.info(f"Total files seen: {total_files_seen}")
        logger.info(f"Code files found: {len(file_map)}")
        logger.info(f"Files skipped (ignored dirs): {files_skipped_ignore}")
        logger.info(f"Files skipped (too large): {files_skipped_size}")
        
        if len(file_map) == 0:
            logger.warning("No code files found! Check your path and ignore patterns.")
        elif len(file_map) > 1000:
            logger.warning(f"Large codebase detected ({len(file_map)} files). Analysis may take several minutes.")
            
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
            
    async def _get_codebase_context(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Get essential codebase context for AI assistants by aggregating from 6 individual analyses.
        This is the FIRST tool to use in any conversation.
        Returns optimized, chunked data that fits within token limits.
        
        This method now aggregates data from the 6 individual analysis tools rather than
        reading directly from codebase files, ensuring consistent and optimized output.
        """
        try:
            # Check if we have cached aggregated context
            codebase_path = Path(os.getcwd())
            cache_key = f"aggregated_context_{codebase_path.name}"
            
            if not refresh:
                cached_context = await self.cache_manager.load_analysis(codebase_path, "aggregated_context")
                if cached_context:
                    logger.info("Returning cached aggregated context")
                    return cached_context['analysis']
                    
            # Gather all 6 individual analyses
            logger.info("Aggregating context from 6 individual analyses...")
            
            # Call each analysis method to get their data
            analyses = await asyncio.gather(
                self._get_dependency_analysis_full(force_refresh=False),
                self._get_security_analysis_full(force_refresh=False),
                self._get_architecture_analysis_full(force_refresh=False),
                self._get_business_logic_analysis_full(force_refresh=False),
                self._get_technical_stack_analysis_full(force_refresh=False),
                self._get_code_intelligence_analysis_full(force_refresh=False),
                return_exceptions=True
            )
            
            # Extract results and handle errors
            dependency_analysis = analyses[0] if not isinstance(analyses[0], Exception) else {}
            security_analysis = analyses[1] if not isinstance(analyses[1], Exception) else {}
            architecture_analysis = analyses[2] if not isinstance(analyses[2], Exception) else {}
            business_analysis = analyses[3] if not isinstance(analyses[3], Exception) else {}
            technical_analysis = analyses[4] if not isinstance(analyses[4], Exception) else {}
            intelligence_analysis = analyses[5] if not isinstance(analyses[5], Exception) else {}
            
            # Check if we have at least some analysis data
            if all(isinstance(a, Exception) or a.get('error') for a in analyses):
                return {
                    'error': 'No analysis available. Please run get_and_set_the_codebase_knowledge_foundation first.',
                    'hint': 'This is a one-time setup that takes 4-5 minutes.',
                    'details': [str(a) if isinstance(a, Exception) else a.get('error') for a in analyses]
                }
            
            # Build instant context from aggregated data
            instant_context = self._build_instant_context(
                dependency_analysis,
                security_analysis,
                architecture_analysis,
                business_analysis,
                technical_analysis,
                intelligence_analysis
            )
            
            # Extract danger zones from security analysis
            danger_zones = self._extract_aggregated_danger_zones(security_analysis, architecture_analysis)
            
            # Build the aggregated context with token limits (5K per section)
            context = {
                'instant_context': instant_context,
                'danger_zones': danger_zones,
                'critical_files': self._extract_critical_files(danger_zones),
                'safe_modification_guide': self._build_safe_modification_guide(security_analysis, architecture_analysis),
                
                # Business understanding (5K tokens)
                'business_understanding': self.token_manager.truncate_to_tokens({
                    'executive_summary': business_analysis.get('executive_summary', ''),
                    'key_features': business_analysis.get('summary', {}).get('key_features', []),
                    'domain_entities': business_analysis.get('summary', {}).get('total_entities', 0),
                    'business_rules': business_analysis.get('summary', {}).get('total_rules', 0),
                    'compliance': business_analysis.get('compliance_requirements', [])[:5]
                }, 5000),
                
                # Architecture overview (5K tokens)
                'architecture_overview': self.token_manager.truncate_to_tokens({
                    'style': architecture_analysis.get('architecture_style', ''),
                    'layers': architecture_analysis.get('layers', {}),
                    'components': architecture_analysis.get('summary', {}).get('total_components', 0),
                    'services': architecture_analysis.get('summary', {}).get('total_services', 0),
                    'communication_patterns': architecture_analysis.get('communication_patterns', [])[:5]
                }, 5000),
                
                # Technical stack (5K tokens)
                'technical_stack': self.token_manager.truncate_to_tokens({
                    'languages': technical_analysis.get('languages', {}),
                    'frameworks': list(technical_analysis.get('frameworks', {}).keys())[:10],
                    'package_managers': technical_analysis.get('package_managers', []),
                    'build_tools': technical_analysis.get('build_tools', [])[:5]
                }, 5000),
                
                # Security summary (3K tokens)
                'security_summary': self.token_manager.truncate_to_tokens({
                    'score': security_analysis.get('security_score', 0),
                    'critical_issues': security_analysis.get('summary', {}).get('critical_count', 0),
                    'high_issues': security_analysis.get('summary', {}).get('high_count', 0),
                    'auth_mechanisms': security_analysis.get('auth_mechanisms', [])
                }, 3000),
                
                # Code intelligence (5K tokens)
                'code_intelligence': self.token_manager.truncate_to_tokens({
                    'entry_points': intelligence_analysis.get('entry_points', [])[:5],
                    'api_endpoints': intelligence_analysis.get('summary', {}).get('api_endpoints', 0),
                    'critical_interfaces': intelligence_analysis.get('summary', {}).get('critical_interfaces', 0),
                    'main_components': intelligence_analysis.get('main_components', [])[:5]
                }, 5000),
                
                # Dependencies overview (2K tokens)
                'dependencies_summary': self.token_manager.truncate_to_tokens({
                    'total_packages': dependency_analysis.get('summary', {}).get('total_packages', 0),
                    'outdated': dependency_analysis.get('summary', {}).get('outdated_count', 0),
                    'security_vulnerabilities': dependency_analysis.get('summary', {}).get('vulnerable_count', 0)
                }, 2000),
                
                # Metadata and latest changes
                'metadata': {
                    'aggregation_timestamp': datetime.now().isoformat(),
                    'analyses_status': {
                        'dependency': 'available' if not dependency_analysis.get('error') else 'error',
                        'security': 'available' if not security_analysis.get('error') else 'error',
                        'architecture': 'available' if not architecture_analysis.get('error') else 'error',
                        'business': 'available' if not business_analysis.get('error') else 'error',
                        'technical': 'available' if not technical_analysis.get('error') else 'error',
                        'intelligence': 'available' if not intelligence_analysis.get('error') else 'error'
                    },
                    'latest_changes': dependency_analysis.get('latest_changes', {})
                }
            }
            
            # Ensure total output is within 25K tokens
            is_valid, tokens = self.token_manager.validate_output_size(context)
            if not is_valid:
                logger.warning(f"Aggregated context exceeds limit ({tokens} tokens), truncating...")
                context = self.token_manager.truncate_to_tokens(context, self.token_manager.MCP_TOKEN_LIMIT)
            
            # Cache the aggregated context
            file_hashes = {}  # Aggregated context doesn't need file hashes
            await self.cache_manager.save_analysis(codebase_path, "aggregated_context", context, file_hashes)
            
            logger.info(f"Successfully aggregated context from 6 analyses ({tokens} tokens)")
            return context
            
        except Exception as e:
            logger.error(f"Failed to aggregate codebase context: {e}")
            return {'error': str(e)}
            
    async def _check_understanding(self, implementation_plan: str, 
                                  files_to_modify: List[str] = None,
                                  understanding_points: List[str] = None) -> Dict[str, Any]:
        """
        Check AI's understanding before allowing code implementation.
        This is the "red flag" system that prevents overconfident changes.
        """
        # Get the codebase context to check understanding
        context = await self._get_codebase_context(refresh=False)
        if context.get('error'):
            return {
                'error': 'No analysis available. Run get_and_set_the_codebase_knowledge_foundation first.',
                'approval': False,
                'score': 0
            }
            
        try:
            # Initialize scoring
            score = 0
            max_score = 10
            feedback = []
            warnings = []
            
            # Check 1: Plan clarity (2 points)
            if implementation_plan and len(implementation_plan) > 50:
                score += 2
                feedback.append("✓ Clear implementation plan provided")
            else:
                feedback.append("✗ Implementation plan needs more detail")
                
            # Check 2: Files identified (2 points)
            if files_to_modify:
                score += 1
                feedback.append(f"✓ Identified {len(files_to_modify)} files to modify")
                
                # Check if any are dangerous
                danger_zones = context.get('danger_zones', {})
                critical_files = set()
                for f in danger_zones.get('do_not_modify', []):
                    critical_files.add(f.get('file', ''))
                for f in danger_zones.get('extreme_caution', []):
                    critical_files.add(f.get('file', ''))
                    
                dangerous_modifications = [f for f in files_to_modify if f in critical_files]
                if dangerous_modifications:
                    warnings.append(f"⚠️ CRITICAL: Planning to modify high-risk files: {dangerous_modifications}")
                    score -= 2  # Penalty for dangerous modifications
                else:
                    score += 1
                    feedback.append("✓ No critical files in modification list")
            else:
                feedback.append("✗ No files specified for modification")
                
            # Check 3: Understanding demonstrated (3 points)
            if understanding_points and len(understanding_points) >= 3:
                score += 2
                feedback.append(f"✓ Demonstrated understanding with {len(understanding_points)} points")
                
                # Bonus for mentioning specific risks
                risk_awareness = any('risk' in p.lower() or 'impact' in p.lower() 
                                   or 'dependency' in p.lower() for p in understanding_points)
                if risk_awareness:
                    score += 1
                    feedback.append("✓ Shows awareness of risks and dependencies")
            else:
                feedback.append("✗ Need to demonstrate deeper understanding")
                
            # Check 4: Business impact awareness (2 points)
            business_keywords = ['business', 'feature', 'user', 'functionality', 'behavior']
            if any(keyword in implementation_plan.lower() for keyword in business_keywords):
                score += 2
                feedback.append("✓ Considers business impact")
            else:
                feedback.append("✗ Should consider business/user impact")
                
            # Check 5: Testing plan (1 point)
            if 'test' in implementation_plan.lower():
                score += 1
                feedback.append("✓ Includes testing considerations")
            else:
                feedback.append("✗ No testing plan mentioned")
                
            # Determine approval
            approval = score >= 8 and len(warnings) == 0
            
            # Generate detailed guidance
            if not approval:
                guidance = self._generate_understanding_guidance(
                    score, feedback, warnings, implementation_plan
                )
            else:
                guidance = "Approved! Your understanding is sufficient. Proceed with caution."
                
            return {
                'approval': approval,
                'score': f"{score}/{max_score}",
                'feedback': feedback,
                'warnings': warnings,
                'guidance': guidance,
                'next_steps': self._suggest_next_steps(approval, files_to_modify)
            }
            
        except Exception as e:
            logger.error(f"Failed to check understanding: {e}")
            return {
                'error': str(e),
                'approval': False,
                'score': 0
            }
            
    async def _get_impact_analysis(self, file_path: str) -> Dict[str, Any]:
        """Get detailed impact analysis for a specific file."""
        if not self.current_analysis:
            return {
                'error': 'No analysis available. Run get_codebase_context first.'
            }
            
        try:
            # Get cross-file intelligence
            cross_file_intel = self.current_analysis.get('enhanced_understanding', {}).get(
                'cross_file_intelligence', {})
            impact_zones = cross_file_intel.get('impact_zones', {})
            
            # Get specific file impact
            file_impact = impact_zones.get(file_path, {})
            
            if not file_impact:
                return {
                    'file_path': file_path,
                    'status': 'not_analyzed',
                    'message': 'File not found in analysis. It may be safe to modify.',
                    'recommendation': 'Proceed with standard precautions.'
                }
                
            # Get additional context
            deep_analysis = self.current_analysis.get('enhanced_understanding', {}).get(
                'deep_analysis', {})
            file_context = deep_analysis.get('file_contexts', {}).get(file_path, {})
            
            # Build comprehensive impact report
            return {
                'file_path': file_path,
                'risk_level': file_impact.get('risk_level', 'UNKNOWN'),
                'risk_score': file_impact.get('risk_score', 0),
                'impact_summary': {
                    'direct_dependencies': len(file_impact.get('direct_impact', [])),
                    'indirect_dependencies': len(file_impact.get('indirect_impact', [])),
                    'total_impact': file_impact.get('total_impact', 0)
                },
                'risk_factors': file_impact.get('risk_factors', []),
                'ai_warning': file_impact.get('ai_warning', ''),
                'modification_strategy': file_impact.get('modification_strategy', ''),
                'file_details': {
                    'purpose': file_context.get('purpose', 'Unknown'),
                    'language': file_context.get('language', 'Unknown'),
                    'complexity': file_context.get('complexity_score', 0),
                    'critical_functions': len(file_context.get('critical_functions', []))
                },
                'dependencies': {
                    'imports': file_impact.get('direct_impact', [])[:10],
                    'imported_by': file_impact.get('indirect_impact', [])[:10]
                },
                'safe_modification_checklist': self._generate_file_checklist(
                    file_path, file_impact, file_context
                ),
                'alternatives': self._suggest_safer_alternatives(file_path, file_impact)
            }
            
        except Exception as e:
            logger.error(f"Failed to get impact analysis: {e}")
            return {'error': str(e)}
            
    def _summarize_danger_zones(self, danger_zones: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize danger zones to fit within token limits."""
        return {
            'summary': danger_zones.get('summary', ''),
            'do_not_modify_count': len(danger_zones.get('do_not_modify', [])),
            'do_not_modify_sample': [
                {
                    'file': item['file'],
                    'reason': item['reason'][:100] + '...' if len(str(item['reason'])) > 100 else item['reason']
                }
                for item in danger_zones.get('do_not_modify', [])[:5]
            ],
            'extreme_caution_count': len(danger_zones.get('extreme_caution', [])),
            'extreme_caution_sample': [
                {
                    'file': item['file'],
                    'impact': item.get('impact', 0)
                }
                for item in danger_zones.get('extreme_caution', [])[:5]
            ]
        }
        
    def _extract_critical_files(self, danger_zones: Dict[str, Any]) -> List[str]:
        """Extract list of critical files."""
        critical = []
        for item in danger_zones.get('do_not_modify', []):
            critical.append(item.get('file', ''))
        for item in danger_zones.get('extreme_caution', []):
            critical.append(item.get('file', ''))
        return critical
        
    def _build_instant_context(self, dependency: Dict[str, Any], security: Dict[str, Any],
                             architecture: Dict[str, Any], business: Dict[str, Any],
                             technical: Dict[str, Any], intelligence: Dict[str, Any]) -> str:
        """Build instant context string from aggregated analyses."""
        # Extract key information from each analysis
        total_files = sum([
            dependency.get('summary', {}).get('files_analyzed', 0),
            security.get('summary', {}).get('files_analyzed', 0),
            architecture.get('summary', {}).get('files_analyzed', 0),
            business.get('summary', {}).get('files_analyzed', 0),
            technical.get('summary', {}).get('files_analyzed', 0),
            intelligence.get('summary', {}).get('files_analyzed', 0)
        ]) // 6  # Average to avoid duplication
        
        languages = technical.get('languages', {}).get('all', [])
        key_features = business.get('summary', {}).get('key_features', [])
        critical_count = security.get('summary', {}).get('critical_count', 0)
        high_count = security.get('summary', {}).get('high_count', 0)
        
        instant_context = f"""🚀 INSTANT CODEBASE CONTEXT (Read this first!)
=============================================

📊 **Quick Stats:**
- Files: {total_files} | Languages: {', '.join(languages[:3])} | Critical files: {critical_count + high_count}

💼 **What This Does:**
{business.get('executive_summary', 'Codebase analysis not yet complete. Run get_and_set_the_codebase_knowledge_foundation first.')[:200]}

🌟 **Key Features:** {', '.join(key_features[:5])}

🔒 **Security Score:** {security.get('security_score', 'N/A')}/10

⚡ **CRITICAL RULE:** Always check danger_zones before ANY modification!

🎯 **Your Goal:** Make changes safely without breaking existing functionality.

📦 **Tech Stack:** {', '.join(list(technical.get('frameworks', {}).keys())[:3])}
🏗️ **Architecture:** {architecture.get('architecture_style', 'Unknown')}
🔍 **Entry Points:** {len(intelligence.get('entry_points', []))}

⚠️ **Dependencies:** {dependency.get('summary', {}).get('total_packages', 0)} packages ({dependency.get('summary', {}).get('vulnerable_count', 0)} with vulnerabilities)

Remember: This is an AI-optimized summary. Use individual analysis tools for detailed information."""
        
        return instant_context
        
    def _extract_aggregated_danger_zones(self, security: Dict[str, Any], 
                                       architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and aggregate danger zones from security and architecture analyses."""
        danger_zones = {
            'summary': f"⛔ {security.get('summary', {}).get('critical_count', 0) + security.get('summary', {}).get('high_count', 0)} files require extreme caution",
            'do_not_modify': [],
            'extreme_caution': []
        }
        
        # Extract critical security files
        for vuln in security.get('vulnerabilities', []):
            if vuln.get('severity') in ['CRITICAL', 'HIGH']:
                for file_path in vuln.get('affected_files', []):
                    danger_zones['do_not_modify'].append({
                        'file': file_path,
                        'reason': f"Security vulnerability: {vuln.get('type', 'Unknown')}"
                    })
                    
        # Extract critical architecture components
        critical_components = architecture.get('critical_components', [])
        for comp in critical_components[:10]:  # Limit to top 10
            if comp.get('criticality', 0) > 8:
                danger_zones['extreme_caution'].append({
                    'file': comp.get('file_path', ''),
                    'impact': comp.get('impact_score', 0),
                    'reason': comp.get('reason', 'Critical system component')
                })
                
        # Remove duplicates
        seen_files = set()
        unique_do_not_modify = []
        for item in danger_zones['do_not_modify']:
            if item['file'] not in seen_files:
                seen_files.add(item['file'])
                unique_do_not_modify.append(item)
        danger_zones['do_not_modify'] = unique_do_not_modify[:15]  # Limit
        
        unique_extreme_caution = []
        for item in danger_zones['extreme_caution']:
            if item['file'] not in seen_files:
                seen_files.add(item['file'])
                unique_extreme_caution.append(item)
        danger_zones['extreme_caution'] = unique_extreme_caution[:15]  # Limit
        
        return danger_zones
        
    def _build_safe_modification_guide(self, security: Dict[str, Any], 
                                     architecture: Dict[str, Any]) -> List[str]:
        """Build safe modification guidelines from security and architecture analyses."""
        guidelines = [
            "1. 🔍 ALWAYS check danger_zones BEFORE opening any file",
            "2. 📊 Review impact analysis to understand dependencies",
            "3. 🧪 Write tests BEFORE making changes",
            "4. 🔒 Never modify authentication or payment logic without approval",
            "5. 📝 Document all changes with clear reasoning",
            "6. 🎯 Make minimal, incremental changes",
            "7. ✅ Run existing tests after each change",
            "8. 🚨 If you see a security warning, STOP and ask for guidance"
        ]
        
        # Add security-specific guidelines if available
        if security.get('security_guidelines'):
            guidelines.extend(security.get('security_guidelines', [])[:2])
            
        # Add architecture-specific guidelines if available
        if architecture.get('modification_guidelines'):
            guidelines.extend(architecture.get('modification_guidelines', [])[:2])
            
        return guidelines[:10]  # Limit to 10 guidelines
        
    def _generate_understanding_guidance(self, score: int, feedback: List[str], 
                                       warnings: List[str], plan: str) -> str:
        """Generate guidance to improve understanding."""
        guidance = "To improve your understanding score:\n\n"
        
        if score < 8:
            guidance += "1. Provide more detailed implementation plan\n"
            guidance += "2. List ALL files that will be affected\n"
            guidance += "3. Explain the business impact of your changes\n"
            guidance += "4. Describe your testing strategy\n"
            guidance += "5. Show awareness of dependencies and risks\n"
            
        if warnings:
            guidance += "\n⚠️ CRITICAL ISSUES TO ADDRESS:\n"
            for warning in warnings:
                guidance += f"- {warning}\n"
                
        guidance += "\nResubmit with check_understanding when ready."
        return guidance
        
    def _suggest_next_steps(self, approved: bool, files: List[str] = None) -> List[str]:
        """Suggest next steps based on approval status."""
        if approved:
            steps = [
                "1. Run get_modification_guidance for each file before editing",
                "2. Make minimal, incremental changes",
                "3. Run tests after each change",
                "4. Document your changes clearly"
            ]
        else:
            steps = [
                "1. Review the feedback and improve your plan",
                "2. Use get_impact_analysis to understand file dependencies",
                "3. Read get_business_context for domain understanding",
                "4. Resubmit with check_understanding"
            ]
            
        return steps
        
    def _generate_file_checklist(self, file_path: str, impact: Dict[str, Any], 
                                context: Dict[str, Any]) -> List[str]:
        """Generate specific checklist for file modification."""
        checklist = [
            f"□ Confirmed {file_path} risk level: {impact.get('risk_level', 'UNKNOWN')}",
            f"□ Reviewed {impact.get('total_impact', 0)} dependent files",
            "□ Read current implementation completely",
            "□ Created tests for your changes"
        ]
        
        if impact.get('risk_level') in ['CRITICAL', 'HIGH']:
            checklist.extend([
                "□ Got explicit approval for high-risk modification",
                "□ Created comprehensive test coverage",
                "□ Documented every change with reasoning"
            ])
            
        return checklist
        
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

    # Helper method for running individual agents
    async def _run_individual_agent(self, agent_type: str, codebase_path: Path) -> Dict[str, Any]:
        """Run an individual analysis agent without requiring full analysis"""
        logger.info(f"Running {agent_type} agent independently...")
        
        # Discover files
        file_map = await self._discover_files(codebase_path)
        
        # Create entities
        entities = {}
        for rel_path in file_map:
            entities[rel_path] = EnhancedCodeEntity(
                path=rel_path,
                type=self._determine_entity_type(rel_path),
                name=Path(rel_path).stem
            )
        
        # Prepare context for agent
        agent_context = {
            'root_path': codebase_path,
            'file_map': file_map,
            'entities': entities
        }
        
        # Run the appropriate agent
        if agent_type == 'dependency':
            agent = DependencyAnalysisAgent()
        elif agent_type == 'security':
            agent = SecurityAuditAgent()
        elif agent_type == 'architecture':
            agent = ArchitectureAnalysisAgent()
        elif agent_type == 'version':
            agent = VersionCompatibilityAgent()
        elif agent_type == 'pattern':
            agent = PatternDetectionAgent()
        elif agent_type == 'performance':
            agent = PerformanceAnalysisAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        return await agent.analyze(agent_context)
    
    async def _run_enhanced_analysis(self, codebase_path: Path) -> Dict[str, Any]:
        """Run enhanced understanding analysis (deep understanding, cross-file, business logic)"""
        logger.info("Running enhanced understanding analysis...")
        
        # Discover files and read contents
        file_map = await self._discover_files(codebase_path)
        file_contents = {}
        for rel_path, full_path in file_map.items():
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_contents[rel_path] = f.read()
            except Exception as e:
                logger.warning(f"Could not read {rel_path}: {e}")
        
        # Phase 1: Deep Understanding
        deep_agent = DeepUnderstandingAgent()
        deep_contexts = {}
        for file_path, content in file_contents.items():
            try:
                context = deep_agent.analyze_file(file_path, content)
                deep_contexts[file_path] = context
            except Exception as e:
                logger.warning(f"Deep analysis failed for {file_path}: {e}")
        
        deep_understanding = deep_agent.generate_understanding_summary()
        
        # Phase 2: Cross-File Intelligence
        cross_intel = CrossFileIntelligence()
        cross_file_results = cross_intel.analyze_relationships(deep_contexts, file_contents)
        
        # Phase 3: Business Logic Extraction
        business_extractor = BusinessLogicExtractor()
        business_results = business_extractor.extract_business_logic(
            deep_contexts, cross_file_results, file_contents
        )
        
        # Phase 4: AI Knowledge Packaging
        packager = AIKnowledgePackager()
        ai_knowledge_package = packager.create_knowledge_package(
            deep_understanding,
            cross_file_results,
            business_results,
            len(file_map)
        )
        
        return {
            'deep_analysis': deep_understanding,
            'cross_file_intelligence': cross_file_results,
            'business_logic': business_results,
            'ai_knowledge_package': ai_knowledge_package,
            'file_contexts': deep_contexts
        }
    
    # New individual analysis methods (25K tokens each)
    
    async def _get_dependency_analysis_full(self, 
                                           force_refresh: bool = False,
                                           include_transitive: bool = True) -> Dict[str, Any]:
        """Get comprehensive dependency analysis (up to 25K tokens)"""
        try:
            # Check cache first
            codebase_path = Path(os.getcwd())  # Use current working directory
            cached_data = None
            
            if not force_refresh:
                cached_data = await self.cache_manager.load_analysis(codebase_path, "dependency")
                if cached_data:
                    # Check for file changes
                    file_map = await self._discover_files(codebase_path)
                    changes, needs_full = await self.cache_manager.detect_changes(
                        codebase_path, file_map, cached_data
                    )
                    
                    if not changes:
                        logger.info("Returning cached dependency analysis")
                        return cached_data['analysis']
                    elif not needs_full:
                        # Incremental update
                        logger.info(f"Updating dependency analysis for {len(changes)} changed files")
                        # TODO: Implement incremental dependency updates
                        
            # Need full analysis - run the dependency agent directly
            dependency_data = await self._run_individual_agent('dependency', codebase_path)
            
            # Build comprehensive response
            result = {
                'summary': {
                    'total_files': dependency_data.get('total_files', 0),
                    'files_with_dependencies': dependency_data.get('files_with_deps', 0),
                    'package_managers': dependency_data.get('package_managers', []),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'dependency_graph': dependency_data.get('dependency_graph', {}),
                'external_dependencies': dependency_data.get('external_dependencies', {}),
                'internal_dependencies': dependency_data.get('internal_dependencies', 0),
                'file_dependencies': self._extract_file_dependencies(dependency_data),
                'circular_dependencies': dependency_data.get('dependency_graph', {}).get('cycles', []),
                'most_depended_on': dependency_data.get('dependency_graph', {}).get('most_depended_on', []),
                'most_dependencies': dependency_data.get('dependency_graph', {}).get('most_dependencies', []),
                'latest_changes': self.cache_manager.get_latest_changes_summary(codebase_path, cached_data)
            }
            
            # Add transitive dependencies if requested
            if include_transitive:
                result['transitive_dependencies'] = self._calculate_transitive_dependencies(dependency_data)
                
            # Ensure within token limit
            is_valid, tokens = self.token_manager.validate_output_size(result)
            if not is_valid:
                result = self.token_manager.truncate_to_tokens(result, self.token_manager.MCP_TOKEN_LIMIT)
                
            # Cache the result
            file_hashes = await self.cache_manager.hash_files(await self._discover_files(codebase_path))
            await self.cache_manager.save_analysis(codebase_path, "dependency", result, file_hashes)
            
            return result
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return {'error': str(e)}
    
    def _extract_file_dependencies(self, dependency_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract per-file dependency details from dependency analysis data
        
        Note: Since the dependency agent doesn't return per-file data by default,
        we return a summary instead. In a future update, we should modify the
        dependency agent to return this granular data.
        """
        try:
            # Log what data we actually received for debugging
            logger.debug(f"Dependency data keys: {list(dependency_data.keys())}")
            
            # For now, return a summary since per-file data isn't available
            # TODO: Modify DependencyAnalysisAgent to return per-file dependencies
            file_deps = {
                '_summary': {
                    'total_files_analyzed': dependency_data.get('total_files', 0),
                    'files_with_dependencies': dependency_data.get('files_with_deps', 0),
                    'external_packages': dependency_data.get('external_dependencies', {}),
                    'internal_dependency_count': dependency_data.get('internal_dependencies', 0),
                    'note': 'Per-file dependency data not yet available. Use dependency graph for relationships.'
                }
            }
            
            # If we have dependency graph info, add some useful data
            dep_graph = dependency_data.get('dependency_graph', {})
            if dep_graph:
                file_deps['_graph_summary'] = {
                    'total_nodes': dep_graph.get('nodes', 0),
                    'total_edges': dep_graph.get('edges', 0),
                    'has_cycles': len(dep_graph.get('cycles', [])) > 0,
                    'cycle_count': len(dep_graph.get('cycles', [])),
                    'most_depended_on': dep_graph.get('most_depended_on', [])[:5],
                    'most_dependencies': dep_graph.get('most_dependencies', [])[:5]
                }
            
            return file_deps
            
        except Exception as e:
            logger.error(f"Error extracting file dependencies: {e}")
            return {
                '_error': f'Failed to extract file dependencies: {str(e)}',
                '_summary': {'note': 'Dependency extraction failed, see logs for details'}
            }
    
    def _calculate_transitive_dependencies(self, dependency_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate transitive dependencies from dependency graph
        
        Note: The dependency agent returns cycles and dependency counts, but not
        the full graph structure needed for transitive analysis. This method
        provides what information is available.
        """
        try:
            # Log what data we received for debugging
            logger.debug(f"Calculating transitive deps from: {list(dependency_data.keys())}")
            
            dep_graph_info = dependency_data.get('dependency_graph', {})
            
            # Since we don't have the actual graph, provide available info
            transitive_deps = {
                '_summary': {
                    'note': 'Full transitive dependency calculation requires graph structure',
                    'available_data': {
                        'total_nodes': dep_graph_info.get('nodes', 0),
                        'total_edges': dep_graph_info.get('edges', 0),
                        'has_circular_dependencies': len(dep_graph_info.get('cycles', [])) > 0,
                        'circular_dependency_count': len(dep_graph_info.get('cycles', []))
                    }
                }
            }
            
            # Include cycle information if available
            cycles = dep_graph_info.get('cycles', [])
            if cycles:
                transitive_deps['circular_dependencies'] = {
                    'count': len(cycles),
                    'cycles': cycles[:10],  # Limit to first 10 cycles
                    'warning': 'Circular dependencies detected - these create transitive loops'
                }
            
            # Include most connected files
            most_depended = dep_graph_info.get('most_depended_on', [])
            most_deps = dep_graph_info.get('most_dependencies', [])
            
            if most_depended or most_deps:
                transitive_deps['key_files'] = {
                    'most_depended_on': most_depended[:10],
                    'most_dependencies': most_deps[:10],
                    'note': 'These files likely have the most transitive impact'
                }
            
            return transitive_deps
            
        except Exception as e:
            logger.error(f"Error calculating transitive dependencies: {e}")
            return {
                '_error': f'Failed to calculate transitive dependencies: {str(e)}',
                '_summary': {'note': 'Transitive dependency calculation failed'}
            }
    
    async def _calculate_language_breakdown(self, file_map: Dict[str, Path]) -> Dict[str, int]:
        """Calculate file count breakdown by language"""
        language_counts = {}
        
        ext_to_lang = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'JavaScript',
            '.ts': 'TypeScript', 
            '.tsx': 'TypeScript',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.m': 'Objective-C',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.json': 'JSON',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.md': 'Markdown',
            '.sh': 'Shell',
            '.bat': 'Batch',
            '.sql': 'SQL'
        }
        
        for _, full_path in file_map.items():
            ext = full_path.suffix.lower()
            lang = ext_to_lang.get(ext, 'Other')
            language_counts[lang] = language_counts.get(lang, 0) + 1
            
        return language_counts
    
    async def _extract_env_variables(self, file_map: Dict[str, Path]) -> Dict[str, List[str]]:
        """Extract environment variable usage from files
        
        Scans common file types for environment variable usage patterns
        across multiple programming languages.
        """
        env_vars = {}
        errors = []
        
        # Common patterns for env var usage across languages
        patterns = [
            r'os\.environ\.get\([\'"](\w+)[\'"]',  # Python os.environ.get
            r'os\.environ\[[\'"](\w+)[\'"]',        # Python os.environ[]
            r'process\.env\.(\w+)',                 # JavaScript/TypeScript
            r'ENV\[[\'"](\w+)[\'"]',                # Ruby
            r'\$ENV\{(\w+)\}',                      # Perl
            r'getenv\([\'"](\w+)[\'"]',             # C/C++/PHP
            r'\$\{(\w+)\}',                         # Shell variables
            r'\$(\w+)',                             # Shell variables simple
        ]
        
        import re
        
        # Track statistics
        files_scanned = 0
        total_vars_found = 0
        
        for rel_path, full_path in file_map.items():
            try:
                # Only check text files likely to contain env vars
                if full_path.suffix.lower() in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rb', '.sh', '.env', '.yml', '.yaml']:
                    files_scanned += 1
                    
                    async with aiofiles.open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = await f.read()
                        
                    found_vars = set()
                    for pattern in patterns:
                        try:
                            matches = re.findall(pattern, content, re.MULTILINE)
                            # Filter out common false positives
                            filtered_matches = [m for m in matches if m.upper() == m or '_' in m]
                            found_vars.update(filtered_matches)
                        except re.error as e:
                            logger.warning(f"Regex error in pattern {pattern}: {e}")
                    
                    if found_vars:
                        env_vars[rel_path] = sorted(list(found_vars))
                        total_vars_found += len(found_vars)
                        
            except PermissionError:
                errors.append(f"Permission denied: {rel_path}")
            except UnicodeDecodeError:
                # Binary file, skip silently
                pass
            except Exception as e:
                logger.debug(f"Error reading {rel_path} for env vars: {e}")
                errors.append(f"{rel_path}: {str(e)}")
        
        # Add summary
        env_vars['_summary'] = {
            'files_scanned': files_scanned,
            'files_with_env_vars': len(env_vars) - 1,  # Exclude _summary
            'total_unique_vars': total_vars_found,
            'scan_errors': len(errors),
            'error_sample': errors[:5] if errors else []
        }
        
        if errors:
            logger.info(f"Environment variable scan completed with {len(errors)} errors")
        
        return env_vars
    
    async def _detect_config_files(self, file_map: Dict[str, Path]) -> Dict[str, str]:
        """Detect and categorize configuration files"""
        config_files = {}
        
        config_patterns = {
            'package.json': 'Node.js package configuration',
            'requirements.txt': 'Python package requirements',
            'setup.py': 'Python package setup',
            'setup.cfg': 'Python setup configuration',
            'pyproject.toml': 'Python project configuration',
            'Cargo.toml': 'Rust package configuration',
            'go.mod': 'Go module configuration',
            'pom.xml': 'Maven configuration',
            'build.gradle': 'Gradle configuration',
            'Gemfile': 'Ruby gem configuration',
            'composer.json': 'PHP composer configuration',
            '.env': 'Environment variables',
            '.env.example': 'Environment variables template',
            'config.json': 'JSON configuration',
            'config.yaml': 'YAML configuration',
            'config.yml': 'YAML configuration',
            'settings.json': 'Settings configuration',
            'appsettings.json': '.NET application settings',
            'web.config': '.NET web configuration',
            '.gitignore': 'Git ignore rules',
            '.dockerignore': 'Docker ignore rules',
            'Dockerfile': 'Docker container configuration',
            'docker-compose.yml': 'Docker compose configuration',
            'Makefile': 'Make build configuration',
            'webpack.config.js': 'Webpack bundler configuration',
            'vite.config.js': 'Vite bundler configuration',
            'rollup.config.js': 'Rollup bundler configuration',
            'tsconfig.json': 'TypeScript configuration',
            'jest.config.js': 'Jest test configuration',
            '.eslintrc': 'ESLint configuration',
            '.prettierrc': 'Prettier configuration',
            'babel.config.js': 'Babel configuration',
            '.github/workflows': 'GitHub Actions workflows',
            '.gitlab-ci.yml': 'GitLab CI configuration',
            '.travis.yml': 'Travis CI configuration',
            'jenkins': 'Jenkins configuration',
            '.circleci/config.yml': 'CircleCI configuration'
        }
        
        for rel_path, full_path in file_map.items():
            file_name = full_path.name
            
            # Check exact matches
            if file_name in config_patterns:
                config_files[rel_path] = config_patterns[file_name]
            # Check patterns
            elif file_name.endswith('config.js') or file_name.endswith('config.json'):
                config_files[rel_path] = f'{file_name.replace(".config", "")} configuration'
            elif file_name.endswith('.yml') or file_name.endswith('.yaml'):
                if 'config' in file_name.lower():
                    config_files[rel_path] = 'YAML configuration file'
            elif '.github/workflows' in rel_path:
                config_files[rel_path] = 'GitHub Actions workflow'
                
        return config_files
            
    async def _get_security_analysis_full(self,
                                        force_refresh: bool = False,
                                        severity_filter: str = "all") -> Dict[str, Any]:
        """Get comprehensive security analysis (up to 25K tokens)"""
        try:
            # Check cache first
            codebase_path = Path(os.getcwd())
            cached_data = None
            
            if not force_refresh:
                cached_data = await self.cache_manager.load_analysis(codebase_path, "security")
                if cached_data:
                    # Check for file changes
                    file_map = await self._discover_files(codebase_path)
                    changes, needs_full = await self.cache_manager.detect_changes(
                        codebase_path, file_map, cached_data
                    )
                    
                    if not changes:
                        logger.info("Returning cached security analysis")
                        return cached_data['analysis']
                        
            # Need full analysis - run the security agent directly
            security_data = await self._run_individual_agent('security', codebase_path)
            
            # Filter vulnerabilities by severity
            all_vulns = security_data.get('vulnerabilities', [])
            if severity_filter != "all":
                filtered_vulns = [v for v in all_vulns if v.get('severity', '').lower() == severity_filter.lower()]
            else:
                filtered_vulns = all_vulns
                
            # Build comprehensive response
            result = {
                'summary': {
                    'security_score': security_data.get('security_score', 0),
                    'total_vulnerabilities': len(all_vulns),
                    'critical_count': len([v for v in all_vulns if v.get('severity') == 'critical']),
                    'high_count': len([v for v in all_vulns if v.get('severity') == 'high']),
                    'auth_mechanisms': security_data.get('auth_mechanisms', []),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'vulnerabilities': filtered_vulns,
                'security_patterns': security_data.get('security_patterns', {}),
                'auth_implementations': security_data.get('auth_implementations', {}),
                'encryption_usage': security_data.get('encryption_usage', {}),
                'input_validation': security_data.get('input_validation', {}),
                'security_headers': security_data.get('security_headers', {}),
                'sensitive_data_handling': security_data.get('sensitive_data_handling', {}),
                'recommendations': security_data.get('recommendations', []),
                'compliance_status': security_data.get('compliance_status', {}),
                'latest_changes': self.cache_manager.get_latest_changes_summary(codebase_path, cached_data)
            }
            
            # Ensure within token limit
            is_valid, tokens = self.token_manager.validate_output_size(result)
            if not is_valid:
                result = self.token_manager.truncate_to_tokens(result, self.token_manager.MCP_TOKEN_LIMIT)
                
            # Cache the result
            file_hashes = await self.cache_manager.hash_files(await self._discover_files(codebase_path))
            await self.cache_manager.save_analysis(codebase_path, "security", result, file_hashes)
            
            return result
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return {'error': str(e)}
            
    async def _get_architecture_analysis_full(self,
                                            force_refresh: bool = False,
                                            include_diagrams: bool = True) -> Dict[str, Any]:
        """Get comprehensive architecture analysis (up to 25K tokens)"""
        try:
            # Check cache first
            codebase_path = Path(os.getcwd())
            cached_data = None
            
            if not force_refresh:
                cached_data = await self.cache_manager.load_analysis(codebase_path, "architecture")
                if cached_data:
                    # Check for file changes
                    file_map = await self._discover_files(codebase_path)
                    changes, needs_full = await self.cache_manager.detect_changes(
                        codebase_path, file_map, cached_data
                    )
                    
                    if not changes:
                        logger.info("Returning cached architecture analysis")
                        return cached_data['analysis']
                        
            # Need full analysis - run the architecture agent directly
            arch_data = await self._run_individual_agent('architecture', codebase_path)
            
            # Build comprehensive response
            result = {
                'summary': {
                    'architecture_style': arch_data.get('architecture_style', 'unknown'),
                    'patterns_detected': arch_data.get('patterns_detected', []),
                    'total_components': len(arch_data.get('components', {})),
                    'modularity_score': arch_data.get('modularity_score', 0),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'directory_structure': arch_data.get('directory_structure', {}),
                'layers': arch_data.get('layers', {}),
                'components': arch_data.get('components', {}),
                'coupling_analysis': arch_data.get('coupling_analysis', {}),
                'design_patterns': arch_data.get('design_patterns', []),
                'architectural_decisions': arch_data.get('architectural_decisions', []),
                'tech_stack_integration': arch_data.get('tech_stack_integration', {}),
                'scalability_assessment': arch_data.get('scalability_assessment', {}),
                'latest_changes': self.cache_manager.get_latest_changes_summary(codebase_path, cached_data)
            }
            
            # Add ASCII diagrams if requested
            if include_diagrams:
                result['diagrams'] = {
                    'layer_diagram': self._generate_layer_diagram(arch_data.get('layers', {})),
                    'component_diagram': self._generate_component_diagram(arch_data.get('components', {}))
                }
                
            # Ensure within token limit
            is_valid, tokens = self.token_manager.validate_output_size(result)
            if not is_valid:
                result = self.token_manager.truncate_to_tokens(result, self.token_manager.MCP_TOKEN_LIMIT)
                
            # Cache the result
            file_hashes = await self.cache_manager.hash_files(await self._discover_files(codebase_path))
            await self.cache_manager.save_analysis(codebase_path, "architecture", result, file_hashes)
            
            return result
            
        except Exception as e:
            logger.error(f"Architecture analysis failed: {e}")
            return {'error': str(e)}
            
    async def _get_business_logic_analysis_full(self,
                                              force_refresh: bool = False,
                                              include_workflows: bool = True) -> Dict[str, Any]:
        """Get comprehensive business logic analysis (up to 25K tokens)"""
        try:
            # Check cache first
            codebase_path = Path(os.getcwd())
            cached_data = None
            
            if not force_refresh:
                cached_data = await self.cache_manager.load_analysis(codebase_path, "business_logic")
                if cached_data:
                    # Check for file changes
                    file_map = await self._discover_files(codebase_path)
                    changes, needs_full = await self.cache_manager.detect_changes(
                        codebase_path, file_map, cached_data
                    )
                    
                    if not changes:
                        logger.info("Returning cached business logic analysis")
                        return cached_data['analysis']
                        
            # Need full analysis - run enhanced analysis
            enhanced_results = await self._run_enhanced_analysis(codebase_path)
            business_data = enhanced_results.get('business_logic', {})
            
            # Build comprehensive response
            result = {
                'summary': {
                    'total_entities': len(business_data.get('domain_model', {}).get('entities', {})),
                    'total_workflows': len(business_data.get('user_journeys', [])),
                    'total_rules': len(business_data.get('business_rules', [])),
                    'key_features': business_data.get('key_features', []),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'executive_summary': business_data.get('executive_summary', ''),
                'domain_model': business_data.get('domain_model', {}),
                'business_entities': business_data.get('domain_model', {}).get('entities', {}),
                'entity_relationships': business_data.get('domain_model', {}).get('relationships', []),
                'business_rules': business_data.get('business_rules', []),
                'compliance_requirements': business_data.get('compliance_requirements', []),
                'business_glossary': business_data.get('business_glossary', {}),
                'latest_changes': self.cache_manager.get_latest_changes_summary(codebase_path, cached_data)
            }
            
            # Add workflows if requested
            if include_workflows:
                result['user_journeys'] = business_data.get('user_journeys', [])
                result['business_flows'] = business_data.get('business_flows', {})
                result['process_maps'] = business_data.get('process_maps', {})
                
            # Ensure within token limit
            is_valid, tokens = self.token_manager.validate_output_size(result)
            if not is_valid:
                result = self.token_manager.truncate_to_tokens(result, self.token_manager.MCP_TOKEN_LIMIT)
                
            # Cache the result
            file_hashes = await self.cache_manager.hash_files(await self._discover_files(codebase_path))
            await self.cache_manager.save_analysis(codebase_path, "business_logic", result, file_hashes)
            
            return result
            
        except Exception as e:
            logger.error(f"Business logic analysis failed: {e}")
            return {'error': str(e)}
            
    async def _get_technical_stack_analysis_full(self,
                                               force_refresh: bool = False,
                                               include_configs: bool = True) -> Dict[str, Any]:
        """Get comprehensive technical stack analysis (up to 25K tokens)"""
        try:
            # Check cache first
            codebase_path = Path(os.getcwd())
            cached_data = None
            
            if not force_refresh:
                cached_data = await self.cache_manager.load_analysis(codebase_path, "technical_stack")
                if cached_data:
                    # Check for file changes
                    file_map = await self._discover_files(codebase_path)
                    changes, needs_full = await self.cache_manager.detect_changes(
                        codebase_path, file_map, cached_data
                    )
                    
                    if not changes:
                        logger.info("Returning cached technical stack analysis")
                        return cached_data['analysis']
                        
            # Need full analysis - run required agents
            logger.info("Running technical stack analysis...")
            
            # Run dependency and version agents for technical stack
            dependency_data = await self._run_individual_agent('dependency', codebase_path)
            version_data = await self._run_individual_agent('version', codebase_path)
            pattern_data = await self._run_individual_agent('pattern', codebase_path)
            
            # Run enhanced analysis for language detection
            enhanced_results = await self._run_enhanced_analysis(codebase_path)
            deep_analysis = enhanced_results.get('deep_analysis', {})
            
            # Create agent_results structure
            agent_results = {
                'dependency': dependency_data,
                'version': version_data,
                'pattern': pattern_data
            }
            
            # Build comprehensive response
            result = {
                'summary': {
                    'languages': deep_analysis.get('languages_found', []),
                    'package_managers': agent_results.get('dependency', {}).get('package_managers', []),
                    'framework_count': len(agent_results.get('version', {}).get('requirements', {})),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'languages': {
                    'primary': deep_analysis.get('languages_found', [])[0] if deep_analysis.get('languages_found') else 'unknown',
                    'all': deep_analysis.get('languages_found', []),
                    'file_breakdown': await self._calculate_language_breakdown(file_map)
                },
                'frameworks': agent_results.get('version', {}).get('requirements', {}),
                'package_managers': agent_results.get('dependency', {}).get('package_managers', []),
                'external_dependencies': agent_results.get('dependency', {}).get('external_dependencies', {}),
                'build_tools': agent_results.get('pattern', {}).get('build_patterns', []),
                'version_requirements': agent_results.get('version', {}).get('requirements', {}),
                'compatibility_issues': agent_results.get('version', {}).get('compatibility_issues', []),
                'outdated_packages': agent_results.get('version', {}).get('outdated_packages', []),
                'latest_changes': self.cache_manager.get_latest_changes_summary(codebase_path, cached_data)
            }
            
            # Add configuration details if requested
            if include_configs:
                result['configurations'] = {
                    'environment_variables': await self._extract_env_variables(file_map),
                    'config_files': await self._detect_config_files(file_map),
                    'build_scripts': agent_results.get('pattern', {}).get('build_patterns', [])
                }
                
            # Ensure within token limit
            is_valid, tokens = self.token_manager.validate_output_size(result)
            if not is_valid:
                result = self.token_manager.truncate_to_tokens(result, self.token_manager.MCP_TOKEN_LIMIT)
                
            # Cache the result
            file_hashes = await self.cache_manager.hash_files(await self._discover_files(codebase_path))
            await self.cache_manager.save_analysis(codebase_path, "technical_stack", result, file_hashes)
            
            return result
            
        except Exception as e:
            logger.error(f"Technical stack analysis failed: {e}")
            return {'error': str(e)}
            
    async def _get_code_intelligence_analysis_full(self,
                                                 force_refresh: bool = False,
                                                 include_patterns: bool = True) -> Dict[str, Any]:
        """Get comprehensive code intelligence analysis (up to 25K tokens)"""
        try:
            # Check cache first
            codebase_path = Path(os.getcwd())
            cached_data = None
            
            if not force_refresh:
                cached_data = await self.cache_manager.load_analysis(codebase_path, "code_intelligence")
                if cached_data:
                    # Check for file changes
                    file_map = await self._discover_files(codebase_path)
                    changes, needs_full = await self.cache_manager.detect_changes(
                        codebase_path, file_map, cached_data
                    )
                    
                    if not changes:
                        logger.info("Returning cached code intelligence analysis")
                        return cached_data['analysis']
                        
            # Need full analysis - run required analyses
            logger.info("Running code intelligence analysis...")
            
            # Run enhanced analysis for code intelligence
            enhanced_results = await self._run_enhanced_analysis(codebase_path)
            deep_analysis = enhanced_results.get('deep_analysis', {})
            cross_file_intel = enhanced_results.get('cross_file_intelligence', {})
            
            # Run architecture and pattern agents
            arch_data = await self._run_individual_agent('architecture', codebase_path)
            pattern_data = await self._run_individual_agent('pattern', codebase_path)
            
            # Create agent_results structure
            agent_results = {
                'architecture': arch_data,
                'pattern': pattern_data
            }
            
            # Build comprehensive response
            result = {
                'summary': {
                    'entry_points': len(deep_analysis.get('entry_points', [])),
                    'api_endpoints': len(cross_file_intel.get('api_boundaries', {})),
                    'services': len(agent_results.get('architecture', {}).get('components', {})),
                    'critical_interfaces': len(cross_file_intel.get('critical_interfaces', [])),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'entry_points': deep_analysis.get('entry_points', []),
                'main_components': deep_analysis.get('main_components', []),
                'api_boundaries': cross_file_intel.get('api_boundaries', {}),
                'service_registry': agent_results.get('architecture', {}).get('components', {}),
                'critical_interfaces': cross_file_intel.get('critical_interfaces', []),
                'function_signatures': deep_analysis.get('key_functions', {}),
                'class_hierarchy': deep_analysis.get('class_hierarchy', {}),
                'error_handling': {
                    'patterns': agent_results.get('pattern', {}).get('error_patterns', []),
                    'strategies': deep_analysis.get('error_handling_strategies', {})
                },
                'data_flow': cross_file_intel.get('data_flow_analysis', {}),
                'latest_changes': self.cache_manager.get_latest_changes_summary(codebase_path, cached_data)
            }
            
            # Add design patterns if requested
            if include_patterns:
                result['design_patterns'] = agent_results.get('pattern', {}).get('patterns', {})
                result['code_smells'] = agent_results.get('pattern', {}).get('code_smells', [])
                result['best_practices'] = agent_results.get('pattern', {}).get('best_practices', {})
                
            # Ensure within token limit
            is_valid, tokens = self.token_manager.validate_output_size(result)
            if not is_valid:
                result = self.token_manager.truncate_to_tokens(result, self.token_manager.MCP_TOKEN_LIMIT)
                
            # Cache the result
            file_hashes = await self.cache_manager.hash_files(await self._discover_files(codebase_path))
            await self.cache_manager.save_analysis(codebase_path, "code_intelligence", result, file_hashes)
            
            return result
            
        except Exception as e:
            logger.error(f"Code intelligence analysis failed: {e}")
            return {'error': str(e)}
            
    def _generate_layer_diagram(self, layers: Dict[str, List[str]]) -> str:
        """Generate ASCII diagram for architectural layers"""
        diagram = """
┌─────────────────────────────────────────┐
│           Presentation Layer            │
├─────────────────────────────────────────┤
│            Business Layer               │
├─────────────────────────────────────────┤
│              Data Layer                 │
├─────────────────────────────────────────┤
│          Infrastructure Layer           │
└─────────────────────────────────────────┘
"""
        return diagram
        
    def _generate_component_diagram(self, components: Dict[str, Any]) -> str:
        """Generate ASCII diagram for components"""
        # Simple component diagram
        return "Component diagram generation not yet implemented"
    
    async def _get_and_set_knowledge_foundation(
            self,
            path: str = ".",
            enable_embeddings: bool = True,
            force_refresh: bool = False
        ) -> Dict[str, Any]:
        """Run all 4 phases of analysis to establish complete codebase knowledge foundation"""
        try:
            root_path = Path(path).resolve()
            if not root_path.exists():
                return {'error': f'Path does not exist: {path}'}
                
            if not root_path.is_dir():
                return {'error': f'Path is not a directory: {path}'}
                
            logger.info("🚀 Starting complete knowledge foundation setup...")
            start_time = datetime.now()
            
            # Phase 1: 25K Gold Tools (Foundation Data) - Run in parallel with timeouts
            logger.info("Phase 1: Running 25K Gold tools in parallel...")
            
            # Define timeout based on expected codebase size
            file_map = await self._discover_files(root_path)
            file_count = len(file_map)
            
            # Scale timeout with codebase size: 60s base + 1s per 10 files
            analysis_timeout = 60 + (file_count // 10)
            max_timeout = 600  # 10 minutes max
            timeout_seconds = min(analysis_timeout, max_timeout)
            
            logger.info(f"Setting analysis timeout to {timeout_seconds}s for {file_count} files")
            
            # Create tasks with individual logging
            phase1_tasks = [
                asyncio.create_task(self._run_with_timeout(
                    self._get_dependency_analysis_full(force_refresh=force_refresh),
                    timeout_seconds,
                    "Dependency Analysis"
                )),
                asyncio.create_task(self._run_with_timeout(
                    self._get_security_analysis_full(force_refresh=force_refresh),
                    timeout_seconds,
                    "Security Analysis"
                )),
                asyncio.create_task(self._run_with_timeout(
                    self._get_architecture_analysis_full(force_refresh=force_refresh),
                    timeout_seconds,
                    "Architecture Analysis"
                )),
                asyncio.create_task(self._run_with_timeout(
                    self._get_technical_stack_analysis_full(force_refresh=force_refresh),
                    timeout_seconds,
                    "Technical Stack Analysis"
                )),
                asyncio.create_task(self._run_with_timeout(
                    self._get_code_intelligence_analysis_full(force_refresh=force_refresh),
                    timeout_seconds,
                    "Code Intelligence Analysis"
                )),
                asyncio.create_task(self._run_with_timeout(
                    self._get_business_logic_analysis_full(force_refresh=force_refresh),
                    timeout_seconds,
                    "Business Logic Analysis"
                ))
            ]
            
            # Run with progress tracking
            phase1_results = await asyncio.gather(*phase1_tasks, return_exceptions=True)
            
            # Check for errors in Phase 1
            phase1_errors = [str(r) for r in phase1_results if isinstance(r, Exception)]
            if phase1_errors:
                logger.warning(f"Phase 1 had {len(phase1_errors)} errors: {phase1_errors}")
            
            # Phase 2: CIA Tools (Enhanced Understanding) - Already run within business logic analysis
            logger.info("Phase 2: CIA tools completed (ran within business logic analysis)")
            
            # Phase 3: Crossing Guards (Safety Features)
            logger.info("Phase 3: Running Crossing Guards tools...")
            
            # These tools depend on the previous phases
            context_result = await self._get_codebase_context(refresh=False)
            
            # Phase 4: Premium Embedders (if enabled)
            if enable_embeddings and self.vector_db:
                logger.info("Phase 4: Running Premium Embedders...")
                # Run embedding agent on all analyzed content
                file_map = await self._discover_files(root_path)
                entities = {}
                for rel_path in file_map:
                    entities[rel_path] = EnhancedCodeEntity(
                        path=rel_path,
                        type=self._determine_entity_type(rel_path),
                        name=Path(rel_path).stem
                    )
                
                embedding_context = {
                    'root_path': root_path,
                    'file_map': file_map,
                    'entities': entities,
                    'enable_embeddings': True,
                    'previous_results': {
                        'dependency': phase1_results[0] if not isinstance(phase1_results[0], Exception) else {},
                        'security': phase1_results[1] if not isinstance(phase1_results[1], Exception) else {},
                        'architecture': phase1_results[2] if not isinstance(phase1_results[2], Exception) else {},
                        'technical': phase1_results[3] if not isinstance(phase1_results[3], Exception) else {},
                        'intelligence': phase1_results[4] if not isinstance(phase1_results[4], Exception) else {},
                        'business': phase1_results[5] if not isinstance(phase1_results[5], Exception) else {}
                    }
                }
                
                embedding_agent = EmbeddingAgent()
                await embedding_agent.analyze(embedding_context)
                logger.info("✅ Embeddings created successfully")
            else:
                logger.info("Phase 4: Skipped (embeddings not enabled)")
            
            # CRITICAL: Store the analysis results so subsequent tools can access them
            # This was missing and causing phases to not share data!
            logger.info("Storing analysis results for cross-phase access...")
            
            # Build comprehensive analysis results
            self.current_analysis = {
                'timestamp': datetime.now().isoformat(),
                'root_path': str(root_path),
                'phase1_results': {
                    'dependency': phase1_results[0] if not isinstance(phase1_results[0], Exception) else {'error': str(phase1_results[0])},
                    'security': phase1_results[1] if not isinstance(phase1_results[1], Exception) else {'error': str(phase1_results[1])},
                    'architecture': phase1_results[2] if not isinstance(phase1_results[2], Exception) else {'error': str(phase1_results[2])},
                    'technical': phase1_results[3] if not isinstance(phase1_results[3], Exception) else {'error': str(phase1_results[3])},
                    'intelligence': phase1_results[4] if not isinstance(phase1_results[4], Exception) else {'error': str(phase1_results[4])},
                    'business': phase1_results[5] if not isinstance(phase1_results[5], Exception) else {'error': str(phase1_results[5])}
                },
                'context_result': context_result if 'context_result' in locals() else {},
                'embeddings_enabled': enable_embeddings and self.vector_db,
                'files_analyzed': len(await self._discover_files(root_path))
            }
            
            # Also store enhanced understanding if it exists
            if 'phase1_results' in locals() and len(phase1_results) > 5:
                business_result = phase1_results[5]
                if not isinstance(business_result, Exception) and 'enhanced_understanding' in business_result:
                    self.current_analysis['enhanced_understanding'] = business_result['enhanced_understanding']
            
            logger.info(f"Analysis results stored. Keys: {list(self.current_analysis.keys())}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Return summary
            return {
                'status': 'success',
                'message': 'Knowledge foundation established successfully',
                'execution_time': f"{execution_time:.2f} seconds",
                'phases_completed': {
                    'phase1_25k_gold': 'completed' if not all(isinstance(r, Exception) for r in phase1_results) else 'partial',
                    'phase2_cia': 'completed',
                    'phase3_crossing_guards': 'completed' if not context_result.get('error') else 'failed',
                    'phase4_embedders': 'completed' if enable_embeddings and self.vector_db else 'skipped'
                },
                'files_analyzed': self.current_analysis['files_analyzed'],
                'analysis_stored': True,  # Indicate that results are available for other tools
                'next_steps': [
                    'Use individual analysis tools to access specific data',
                    'Use get_codebase_context for aggregated safety context',
                    'Use semantic_code_search to search your codebase',
                    'Use check_understanding before making changes'
                ],
                'errors': phase1_errors if phase1_errors else None
            }
            
        except Exception as e:
            logger.error(f"Knowledge foundation setup failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'path': path
            }
    
    async def _update_cached_knowledge_foundation(
            self,
            path: str = "."
        ) -> Dict[str, Any]:
        """Check if codebase has changed and update if needed"""
        try:
            root_path = Path(path).resolve()
            if not root_path.exists():
                return {'error': f'Path does not exist: {path}'}
                
            logger.info("Checking for codebase changes...")
            
            # Get latest git commit
            try:
                import subprocess
                result = subprocess.run(
                    ['git', 'log', '-1', '--format=%H %ct'],
                    cwd=root_path,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    commit_hash, commit_timestamp = result.stdout.strip().split()
                    commit_time = datetime.fromtimestamp(int(commit_timestamp))
                else:
                    # Not a git repo, use file modification times
                    commit_time = datetime.now()
                    commit_hash = "no-git"
            except Exception as e:
                logger.warning(f"Could not get git info: {e}")
                commit_time = datetime.now()
                commit_hash = "error"
            
            # Check cache timestamps
            cache_dir = Path.home() / ".codebaseiq" / "cache"
            needs_update = False
            oldest_cache = None
            
            for analysis_type in ['dependency', 'security', 'architecture', 'business_logic', 'technical_stack', 'code_intelligence']:
                cache_file = cache_dir / f"{root_path.name}_{analysis_type}.json"
                if cache_file.exists():
                    cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if oldest_cache is None or cache_mtime < oldest_cache:
                        oldest_cache = cache_mtime
                else:
                    needs_update = True
                    break
            
            if not needs_update and oldest_cache and commit_time > oldest_cache:
                needs_update = True
                
            if needs_update:
                logger.info("Changes detected, updating knowledge foundation...")
                result = await self._get_and_set_knowledge_foundation(
                    path=str(root_path),
                    enable_embeddings=True,
                    force_refresh=True
                )
                result['update_reason'] = 'Codebase changed since last analysis'
                result['git_commit'] = commit_hash
                return result
            else:
                return {
                    'status': 'up_to_date',
                    'message': 'Knowledge foundation is up to date',
                    'last_analysis': oldest_cache.isoformat() if oldest_cache else 'never',
                    'latest_commit': commit_hash,
                    'commit_time': commit_time.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'path': path
            }




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