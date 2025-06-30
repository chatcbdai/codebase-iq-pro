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
                
                return {
                    'status': 'success',
                    'path': str(root_path),
                    'files_analyzed': len(file_map),
                    'analysis_type': analysis_type,
                    'results': results,
                    'config': self.config.get_config_summary(),
                    'ai_ready': True,
                    'instant_context': ai_knowledge_package.get('instant_context', ''),
                    'danger_zones_summary': ai_knowledge_package.get('danger_zones', {}).get('summary', '')
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
                'error': 'No analysis performed yet. Run analyze_codebase first.'
            }
            
        enhanced = self.current_analysis.get('enhanced_understanding', {})
        if not enhanced:
            return {
                'error': 'Enhanced understanding not available. Run analyze_codebase with latest version.'
            }
            
        ai_package = enhanced.get('ai_knowledge_package', {})
        
        # Add quick access information
        return {
            'metadata': ai_package.get('metadata', {}),
            'instant_context': ai_package.get('instant_context', ''),
            'danger_zones': ai_package.get('danger_zones', {}),
            'safe_modification_guide': ai_package.get('safe_modification_guide', {}),
            'ai_instructions': ai_package.get('ai_instructions', ''),
            'quick_reference': ai_package.get('quick_reference', {}),
            'modification_checklist': ai_package.get('modification_checklist', []),
            'testing_requirements': ai_package.get('testing_requirements', {}),
            'emergency_contacts': ai_package.get('emergency_contacts', {}),
            'usage_hint': "Read instant_context first, then check danger_zones before ANY modification"
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
        
        return {
            'executive_summary': business_logic.get('executive_summary', ''),
            'domain_model': business_logic.get('domain_model', {}),
            'user_journeys': business_logic.get('user_journeys', []),
            'business_rules': business_logic.get('business_rules', []),
            'key_features': business_logic.get('key_features', []),
            'compliance_requirements': business_logic.get('compliance_requirements', []),
            'business_glossary': business_logic.get('business_glossary', {}),
            'immediate_context': business_logic.get('immediate_context', '')
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