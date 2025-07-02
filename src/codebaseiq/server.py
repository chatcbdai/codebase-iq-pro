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
                    name="analyze_codebase",
                    description="Full codebase analysis (4-5 minutes). Use get_codebase_context instead for quick access.",
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
                
            elif name == "analyze_codebase":
                result = await self._analyze_codebase(
                    path=arguments.get("path"),
                    analysis_type=arguments.get("analysis_type", "full"),
                    enable_embeddings=arguments.get("enable_embeddings", True),
                    focus_areas=arguments.get("focus_areas"),
                    force_refresh=arguments.get("force_refresh", True)  # Default to True for fresh analysis
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
                
                # Execute orchestrated analysis
                results = await self.orchestrator.execute(context, analysis_type)
                
                # Enhanced Understanding Phase - Provides immediate AI context
                logger.info("🧠 Starting enhanced understanding for AI assistants...")
                
                # Read file contents for enhanced analysis
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
                
                # Add enhanced understanding to results
                results['enhanced_understanding'] = {
                    'deep_analysis': deep_understanding,
                    'cross_file_intelligence': cross_file_results,
                    'business_logic': business_results,
                    'ai_knowledge_package': ai_knowledge_package
                }
                
                # Store in cache
                if self.cache:
                    await self.cache.set(cache_key, results, ttl=3600)  # 1 hour
                    
                # Store current analysis
                self.current_analysis = results
                
                # Save to persistent storage
                try:
                    storage_dir = Path.home() / ".codebaseiq"
                    storage_dir.mkdir(exist_ok=True)
                    
                    # Use codebase-specific cache file name
                    codebase_name = root_path.name
                    cache_filename = f"analysis_cache_{codebase_name}.json"
                    storage_path = storage_dir / cache_filename
                    
                    # Also save a pointer to the latest analysis
                    latest_path = storage_dir / "latest_analysis.json"
                    
                    # Save with metadata
                    to_save = {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'codebase_path': str(root_path),
                        'codebase_name': codebase_name,
                        'files_analyzed': len(file_map),
                        **results
                    }
                    
                    with open(storage_path, 'w') as f:
                        json.dump(to_save, f, indent=2, default=str)
                    
                    # Save pointer to latest
                    with open(latest_path, 'w') as f:
                        json.dump({
                            'latest_codebase': str(root_path),
                            'cache_file': cache_filename,
                            'timestamp': datetime.now().isoformat()
                        }, f)
                        
                    logger.info(f"Saved analysis to {storage_path}")
                except Exception as e:
                    logger.warning(f"Failed to save analysis to persistent storage: {e}")
                
                # Return optimized response (under 25K tokens)
                return {
                    'status': 'success',
                    'path': str(root_path),
                    'files_analyzed': len(file_map),
                    'analysis_type': analysis_type,
                    'summary': {
                        'total_files': len(file_map),
                        'languages': list(set(context.get('language', 'unknown') 
                                           for context in deep_contexts.values())),
                        'high_risk_files': len([f for f, z in cross_file_results.get('impact_zones', {}).items()
                                              if z.get('risk_level') in ['CRITICAL', 'HIGH']]),
                        'key_features': business_results.get('key_features', [])[:5]
                    },
                    'instant_context': ai_knowledge_package.get('instant_context', ''),
                    'danger_zones_preview': {
                        'summary': ai_knowledge_package.get('danger_zones', {}).get('summary', ''),
                        'critical_count': len(ai_knowledge_package.get('danger_zones', {}).get('do_not_modify', [])),
                        'high_risk_count': len(ai_knowledge_package.get('danger_zones', {}).get('extreme_caution', []))
                    },
                    'next_steps': [
                        'Use get_codebase_context for full analysis details',
                        'Use check_understanding before any implementation',
                        'Use get_modification_guidance for specific files'
                    ],
                    'storage_location': str(Path.home() / ".codebaseiq" / "analysis_cache.json")
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
            
    async def _get_ai_knowledge_package(self) -> Dict[str, Any]:
        """Get comprehensive AI knowledge package for immediate understanding"""
        if not self.current_analysis:
            return {
                'error': 'No analysis performed yet. Run analyze_codebase first.',
                'hint': 'Use get_codebase_context instead for optimized access'
            }
            
        enhanced = self.current_analysis.get('enhanced_understanding', {})
        if not enhanced:
            return {
                'error': 'Enhanced understanding not available. Run analyze_codebase with latest version.'
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
                'error': 'No analysis performed yet. Run analyze_codebase first.'
            }
            
        enhanced = self.current_analysis.get('enhanced_understanding', {})
        if not enhanced:
            return {
                'error': 'Enhanced understanding not available. Run analyze_codebase with latest version.'
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
                'error': 'No analysis performed yet. Run analyze_codebase first.'
            }
            
        enhanced = self.current_analysis.get('enhanced_understanding', {})
        if not enhanced:
            return {
                'error': 'Enhanced understanding not available. Run analyze_codebase with latest version.'
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
                    'error': 'No analysis available. Please run analyze_codebase first.',
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
        if not self.current_analysis:
            return {
                'error': 'No analysis available. Run get_codebase_context first.',
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
                danger_zones = self.current_analysis.get('enhanced_understanding', {}).get(
                    'ai_knowledge_package', {}).get('danger_zones', {})
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
{business.get('executive_summary', 'Codebase analysis not yet complete. Run analyze_codebase first.')[:200]}

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
                        
            # Need full analysis
            if not self.current_analysis:
                return {
                    'error': 'No analysis available. Run analyze_codebase first.',
                    'hint': 'This is required for the initial setup.'
                }
                
            # Extract dependency data
            agent_results = self.current_analysis.get('agent_results', {})
            dependency_data = agent_results.get('dependency', {})
            
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
                'file_dependencies': {},  # TODO: Add per-file dependency details
                'circular_dependencies': dependency_data.get('dependency_graph', {}).get('cycles', []),
                'most_depended_on': dependency_data.get('dependency_graph', {}).get('most_depended_on', []),
                'most_dependencies': dependency_data.get('dependency_graph', {}).get('most_dependencies', []),
                'latest_changes': self.cache_manager.get_latest_changes_summary(codebase_path, cached_data)
            }
            
            # Add transitive dependencies if requested
            if include_transitive:
                # TODO: Calculate transitive dependencies
                result['transitive_dependencies'] = {}
                
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
                        
            # Need full analysis
            if not self.current_analysis:
                return {
                    'error': 'No analysis available. Run analyze_codebase first.',
                    'hint': 'This is required for the initial setup.'
                }
                
            # Extract security data
            agent_results = self.current_analysis.get('agent_results', {})
            security_data = agent_results.get('security', {})
            
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
                        
            # Need full analysis
            if not self.current_analysis:
                return {
                    'error': 'No analysis available. Run analyze_codebase first.',
                    'hint': 'This is required for the initial setup.'
                }
                
            # Extract architecture data
            agent_results = self.current_analysis.get('agent_results', {})
            arch_data = agent_results.get('architecture', {})
            
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
                        
            # Need full analysis
            if not self.current_analysis:
                return {
                    'error': 'No analysis available. Run analyze_codebase first.',
                    'hint': 'This is required for the initial setup.'
                }
                
            # Extract business logic data
            enhanced = self.current_analysis.get('enhanced_understanding', {})
            business_data = enhanced.get('business_logic', {})
            
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
                        
            # Need full analysis
            if not self.current_analysis:
                return {
                    'error': 'No analysis available. Run analyze_codebase first.',
                    'hint': 'This is required for the initial setup.'
                }
                
            # Extract technical stack data
            agent_results = self.current_analysis.get('agent_results', {})
            enhanced = self.current_analysis.get('enhanced_understanding', {})
            deep_analysis = enhanced.get('deep_analysis', {})
            
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
                    'file_breakdown': {}  # TODO: Add language breakdown by file count
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
                    'environment_variables': {},  # TODO: Extract env var usage
                    'config_files': {},  # TODO: List config files and their purposes
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
                        
            # Need full analysis
            if not self.current_analysis:
                return {
                    'error': 'No analysis available. Run analyze_codebase first.',
                    'hint': 'This is required for the initial setup.'
                }
                
            # Extract code intelligence data
            enhanced = self.current_analysis.get('enhanced_understanding', {})
            deep_analysis = enhanced.get('deep_analysis', {})
            cross_file_intel = enhanced.get('cross_file_intelligence', {})
            agent_results = self.current_analysis.get('agent_results', {})
            
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
                'function_signatures': {},  # TODO: Extract key function signatures
                'class_hierarchy': {},  # TODO: Build class hierarchy
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