#!/usr/bin/env python3
"""
CodebaseIQ Pro - Advanced MCP Server with Adaptive Service Selection
Refactored version using modular components
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import base functionality and mixins
from .server_base import CodebaseIQProServerBase
from .analysis_methods import AnalysisMethods
from .helper_methods import HelperMethods

# Import specific agents needed here
from .core import EnhancedCodeEntity
from .agents import EmbeddingAgent

# MCP imports
import mcp.types as types


class CodebaseIQProServer(CodebaseIQProServerBase, AnalysisMethods, HelperMethods):
    """Enhanced MCP server with adaptive service selection"""
    
    def __init__(self):
        super().__init__()
        # Setup MCP tools
        self._setup_tools()
        
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
            name: str,
            arguments: Optional[Dict[str, Any]] = None
        ) -> Union[str, List[types.TextContent], List[types.ImageContent], List[types.EmbeddedResource]]:
            """Handle tool calls"""
            try:
                logger.info(f"Tool called: {name} with args: {arguments}")
                
                if name == "get_codebase_context":
                    result = await self._get_codebase_context(
                        refresh=arguments.get('refresh', False) if arguments else False
                    )
                elif name == "check_understanding":
                    result = await self._check_understanding(
                        implementation_plan=arguments.get('implementation_plan', ''),
                        files_to_modify=arguments.get('files_to_modify', []),
                        understanding_points=arguments.get('understanding_points', [])
                    )
                elif name == "get_impact_analysis":
                    result = await self._get_impact_analysis(
                        file_path=arguments.get('file_path', '')
                    )
                elif name == "get_and_set_the_codebase_knowledge_foundation":
                    result = await self._get_and_set_knowledge_foundation(
                        path=arguments.get('path', '.'),
                        enable_embeddings=arguments.get('enable_embeddings', True),
                        force_refresh=arguments.get('force_refresh', False)
                    )
                elif name == "update_cached_knowledge_foundation":
                    result = await self._update_cached_knowledge_foundation(
                        path=arguments.get('path', '.')
                    )
                elif name == "semantic_code_search":
                    result = await self._semantic_code_search(
                        query=arguments.get('query', ''),
                        top_k=arguments.get('top_k', 10),
                        filters=arguments.get('filters'),
                        search_type=arguments.get('search_type', 'semantic')
                    )
                elif name == "find_similar_code":
                    result = await self._find_similar_code(
                        entity_path=arguments.get('entity_path', ''),
                        top_k=arguments.get('top_k', 5),
                        similarity_threshold=arguments.get('similarity_threshold', 0.7)
                    )
                elif name == "get_analysis_summary":
                    result = await self._get_analysis_summary()
                elif name == "get_danger_zones":
                    result = await self._get_danger_zones()
                elif name == "get_dependencies":
                    result = await self._get_dependencies()
                elif name == "get_ai_knowledge_package":
                    result = await self._get_ai_knowledge_package()
                elif name == "get_business_context":
                    result = await self._get_business_context()
                elif name == "get_modification_guidance":
                    result = await self._get_modification_guidance(
                        file_path=arguments.get('file_path', '')
                    )
                elif name == "get_dependency_analysis":
                    result = await self._get_dependency_analysis_full(
                        force_refresh=arguments.get('force_refresh', False),
                        include_transitive=arguments.get('include_transitive', True)
                    )
                elif name == "get_security_analysis":
                    result = await self._get_security_analysis_full(
                        force_refresh=arguments.get('force_refresh', False),
                        severity_filter=arguments.get('severity_filter', 'all')
                    )
                elif name == "get_architecture_analysis":
                    result = await self._get_architecture_analysis_full(
                        force_refresh=arguments.get('force_refresh', False),
                        include_diagrams=arguments.get('include_diagrams', True)
                    )
                elif name == "get_business_logic_analysis":
                    result = await self._get_business_logic_analysis_full(
                        force_refresh=arguments.get('force_refresh', False),
                        include_workflows=arguments.get('include_workflows', True)
                    )
                elif name == "get_technical_stack_analysis":
                    result = await self._get_technical_stack_analysis_full(
                        force_refresh=arguments.get('force_refresh', False),
                        include_configs=arguments.get('include_configs', True)
                    )
                elif name == "get_code_intelligence_analysis":
                    result = await self._get_code_intelligence_analysis_full(
                        force_refresh=arguments.get('force_refresh', False),
                        include_patterns=arguments.get('include_patterns', True)
                    )
                else:
                    result = {'error': f'Unknown tool: {name}'}
                    
                # Return as JSON string
                return json.dumps(result, indent=2)
                
            except Exception as e:
                logger.error(f"Tool {name} failed: {e}", exc_info=True)
                return json.dumps({'error': str(e)})
                
    # Core analysis coordination methods
    async def _get_codebase_context(self, refresh: bool = False) -> Dict[str, Any]:
        """Get essential codebase context for safe modifications"""
        if not refresh and self.current_analysis:
            # Use cached analysis if available
            agent_results = self.current_analysis.get('agent_results', {})
            
            # Extract essential context
            dependency = agent_results.get('dependency', {})
            security = agent_results.get('security', {})
            architecture = agent_results.get('architecture', {})
            business_logic = self.current_analysis.get('business_logic', {})
            
            danger_zones = self._summarize_danger_zones(
                self._extract_aggregated_danger_zones(security, architecture, business_logic)
            )
            
            return {
                'danger_zones': danger_zones,
                'critical_files': self._extract_critical_files(danger_zones),
                'business_context': self._extract_business_summary(business_logic),
                'dependencies': {
                    'external': list(dependency.get('external_dependencies', {}).keys()),
                    'most_depended_on': dependency.get('dependency_graph', {}).get('most_depended_on', [])
                },
                'architecture': {
                    'layers': list(architecture.get('layers', {}).keys()),
                    'entry_points': architecture.get('entry_points', [])
                },
                'ready_for_modifications': True,
                'guidance': 'Use check_understanding before making changes'
            }
            
        # Need to run analysis first
        return {
            'error': 'No analysis available. Run get_and_set_the_codebase_knowledge_foundation first.',
            'ready_for_modifications': False
        }
        
    async def _check_understanding(
            self,
            implementation_plan: str,
            files_to_modify: List[str] = None,
            understanding_points: List[str] = None
        ) -> Dict[str, Any]:
        """Verify understanding before implementation"""
        if not self.current_analysis:
            return {
                'approved': False,
                'score': 0,
                'feedback': ['No codebase analysis available. Run analysis first.']
            }
            
        score = 0
        feedback = []
        
        # Check if plan mentions key aspects
        plan_lower = implementation_plan.lower()
        
        # Check for danger zone awareness
        danger_zones = self._extract_aggregated_danger_zones(
            self.current_analysis.get('agent_results', {}).get('security', {}),
            self.current_analysis.get('agent_results', {}).get('architecture', {}),
            self.current_analysis.get('business_logic', {})
        )
        
        if danger_zones and any(word in plan_lower for word in ['security', 'danger', 'risk', 'careful']):
            score += 20
            feedback.append('✓ Shows awareness of security considerations')
        elif danger_zones:
            feedback.append('⚠️ Plan should acknowledge security/danger zones')
            
        # Check for business logic understanding
        if any(word in plan_lower for word in ['business', 'domain', 'user', 'workflow']):
            score += 20
            feedback.append('✓ Demonstrates business logic understanding')
        else:
            feedback.append('⚠️ Consider business impact and user workflows')
            
        # Check for dependency awareness
        if any(word in plan_lower for word in ['dependency', 'import', 'affect', 'impact']):
            score += 20
            feedback.append('✓ Shows dependency awareness')
        else:
            feedback.append('⚠️ Consider dependency impacts')
            
        # Check if files are mentioned
        if files_to_modify:
            critical_files = self._extract_critical_files(danger_zones)
            risky_files = [f for f in files_to_modify if f in critical_files]
            if risky_files:
                if 'careful' in plan_lower or 'test' in plan_lower:
                    score += 20
                    feedback.append(f'✓ Appropriate caution for critical files: {risky_files}')
                else:
                    feedback.append(f'⚠️ Extra caution needed for critical files: {risky_files}')
            else:
                score += 10
                
        # Check understanding points
        if understanding_points:
            score += min(20, len(understanding_points) * 5)
            feedback.append(f'✓ Provided {len(understanding_points)} understanding points')
            
        # Final verdict
        approved = score >= 60
        
        return {
            'approved': approved,
            'score': score,
            'feedback': feedback,
            'guidance': self._generate_understanding_guidance(score, feedback, approved),
            'next_steps': self._suggest_next_steps(approved, files_to_modify)
        }
        
    async def _get_impact_analysis(self, file_path: str) -> Dict[str, Any]:
        """Get detailed impact analysis for a specific file"""
        if not self.current_analysis:
            return {
                'error': 'No analysis available. Run get_and_set_the_codebase_knowledge_foundation first.'
            }
            
        agent_results = self.current_analysis.get('agent_results', {})
        
        # Find file in various analyses
        dependencies = agent_results.get('dependency', {})
        security = agent_results.get('security', {})
        architecture = agent_results.get('architecture', {})
        
        impact = {
            'file': file_path,
            'exists': file_path in self.current_analysis.get('file_map', {}),
            'dependencies': {
                'imports': [],
                'imported_by': [],
                'external_deps': dependencies.get('external_dependencies', {}).get(file_path, [])
            },
            'security_concerns': [],
            'architectural_role': 'unknown',
            'modification_risk': 'low',
            'testing_requirements': [],
            'suggested_approach': []
        }
        
        # Check dependencies
        dep_graph = dependencies.get('dependency_graph', {})
        if file_path in dep_graph.get('most_depended_on', []):
            impact['modification_risk'] = 'high'
            impact['dependencies']['imported_by'] = ['Multiple files depend on this']
            
        # Check security
        for vuln in security.get('vulnerabilities', []):
            if vuln.get('file') == file_path:
                impact['security_concerns'].append(vuln)
                impact['modification_risk'] = 'high'
                
        # Check architecture
        for layer, files in architecture.get('layers', {}).items():
            if file_path in files:
                impact['architectural_role'] = layer
                
        # Generate guidance
        impact['checklist'] = self._get_file_specific_checklist(file_path, impact, agent_results)
        impact['suggested_approach'] = self._suggest_safer_alternatives(file_path, impact)
        
        return impact
        
    async def _get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of current analysis"""
        if not self.current_analysis:
            return {'error': 'No analysis available yet'}
            
        return {
            'analyzed_at': self.current_analysis.get('timestamp', 'unknown'),
            'files_analyzed': len(self.current_analysis.get('file_map', {})),
            'agents_run': list(self.current_analysis.get('agent_results', {}).keys()),
            'has_embeddings': self.current_analysis.get('embeddings_created', 0) > 0,
            'danger_zones_found': len(self._extract_aggregated_danger_zones(
                self.current_analysis.get('agent_results', {}).get('security', {}),
                self.current_analysis.get('agent_results', {}).get('architecture', {}),
                self.current_analysis.get('business_logic', {})
            )),
            'ready_for_queries': True
        }
        
    async def _get_danger_zones(self) -> Dict[str, Any]:
        """Get danger zones from analysis"""
        if not self.current_analysis:
            return {'error': 'No analysis available yet'}
            
        security = self.current_analysis.get('agent_results', {}).get('security', {})
        architecture = self.current_analysis.get('agent_results', {}).get('architecture', {})
        business_logic = self.current_analysis.get('business_logic', {})
        
        danger_zones = self._extract_aggregated_danger_zones(security, architecture, business_logic)
        
        return {
            'summary': self._summarize_danger_zones(danger_zones),
            'zones': danger_zones,
            'critical_files': self._extract_critical_files(danger_zones),
            'recommendations': self._build_safe_modification_guide(security, architecture, business_logic)
        }
        
    async def _get_dependencies(self) -> Dict[str, Any]:
        """Get dependency information"""
        if not self.current_analysis:
            return {'error': 'No analysis available yet'}
            
        dep_data = self.current_analysis.get('agent_results', {}).get('dependency', {})
        
        return {
            'external': dep_data.get('external_dependencies', {}),
            'internal': dep_data.get('internal_dependencies', 0),
            'graph_stats': dep_data.get('dependency_graph', {}),
            'package_managers': dep_data.get('package_managers', [])
        }
        
    async def _get_ai_knowledge_package(self) -> Dict[str, Any]:
        """Get comprehensive AI knowledge package"""
        if not self.current_analysis:
            return {'error': 'No analysis available. Run get_and_set_the_codebase_knowledge_foundation first.'}
            
        # Package everything an AI needs to know
        return {
            'instant_context': self._build_instant_context(
                self.current_analysis.get('agent_results', {}).get('dependency', {}),
                self.current_analysis.get('agent_results', {}).get('security', {}),
                self.current_analysis.get('agent_results', {}).get('architecture', {}),
                self.current_analysis.get('business_logic', {})
            ),
            'danger_zones': self._extract_aggregated_danger_zones(
                self.current_analysis.get('agent_results', {}).get('security', {}),
                self.current_analysis.get('agent_results', {}).get('architecture', {}),
                self.current_analysis.get('business_logic', {})
            ),
            'modification_guide': self._build_safe_modification_guide(
                self.current_analysis.get('agent_results', {}).get('security', {}),
                self.current_analysis.get('agent_results', {}).get('architecture', {}),
                self.current_analysis.get('business_logic', {})
            ),
            'key_patterns': self.current_analysis.get('agent_results', {}).get('pattern', {}).get('patterns', {}),
            'ready': True
        }
        
    async def _get_business_context(self) -> Dict[str, Any]:
        """Get business logic context"""
        if not self.current_analysis:
            return {'error': 'No analysis available yet'}
            
        business_logic = self.current_analysis.get('business_logic', {})
        
        return {
            'domain_entities': business_logic.get('domain_entities', []),
            'business_rules': business_logic.get('business_rules', []),
            'user_journeys': business_logic.get('user_journeys', []),
            'critical_operations': business_logic.get('critical_operations', [])
        }
        
    async def _get_modification_guidance(self, file_path: str) -> Dict[str, Any]:
        """Get specific modification guidance for a file"""
        # First get impact analysis
        impact = await self._get_impact_analysis(file_path)
        
        if 'error' in impact:
            return impact
            
        # Add specific guidance
        guidance = {
            'file': file_path,
            'risk_level': impact['modification_risk'],
            'pre_modification_checklist': self._generate_file_checklist(
                file_path, impact, 
                self.current_analysis.get('agent_results', {}) if self.current_analysis else {}
            ),
            'dependencies_to_check': impact['dependencies']['imported_by'],
            'security_considerations': impact['security_concerns'],
            'testing_strategy': impact['testing_requirements'],
            'safer_alternatives': impact['suggested_approach']
        }
        
        return guidance
        
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
                
                embedding_agent = EmbeddingAgent(self.vector_db, self.embedding_service)
                embedding_result = await embedding_agent.analyze(embedding_context)
                logger.info("✅ Embeddings created successfully")
                
                # Update embeddings_ready flag
                if embedding_result.get('embeddings_exists') or embedding_result.get('embeddings_created', 0) > 0:
                    self.embeddings_ready = True
            else:
                logger.info("Phase 4: Skipped (embeddings not enabled)")
                embedding_result = {'embeddings_created': 0}
            
            # Store complete analysis
            self.current_analysis = {
                'timestamp': datetime.now().isoformat(),
                'path': str(root_path),
                'file_map': file_map,
                'agent_results': {
                    'dependency': phase1_results[0] if not isinstance(phase1_results[0], Exception) else {},
                    'security': phase1_results[1] if not isinstance(phase1_results[1], Exception) else {},
                    'architecture': phase1_results[2] if not isinstance(phase1_results[2], Exception) else {},
                    'pattern': {},  # Would need to be added
                    'version': {},  # Would need to be added
                    'performance': {}  # Would need to be added
                },
                'business_logic': phase1_results[5] if not isinstance(phase1_results[5], Exception) else {},
                'embeddings_created': embedding_result.get('embeddings_created', 0)
            }
            
            # Save comprehensive analysis to cache
            try:
                file_hashes = await self.cache_manager.hash_files(file_map)
                await self.cache_manager.save_analysis(root_path, "comprehensive", self.current_analysis, file_hashes)
                logger.info("✅ Saved comprehensive analysis to cache")
            except Exception as e:
                logger.warning(f"Failed to save comprehensive analysis: {e}")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return {
                'status': 'success',
                'phases_completed': 4,
                'files_analyzed': len(file_map),
                'embeddings_created': embedding_result.get('embeddings_created', 0),
                'analysis_time': f'{elapsed:.2f}s',
                'ready_for_queries': True,
                'next_steps': [
                    'Use get_ai_knowledge_package for comprehensive AI context',
                    'Use semantic_code_search to find code',
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
                cache_file = self.cache_manager._get_cache_path(root_path, analysis_type)
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
                
            # Check if vector DB has embeddings
            try:
                stats = await self.vector_db.get_stats()
                if not stats or stats.get('total_vectors', 0) == 0:
                    return {
                        'error': 'No embeddings found. Run get_and_set_the_codebase_knowledge_foundation with enable_embeddings=true first.'
                    }
            except Exception as e:
                logger.warning(f"Could not check vector DB stats: {e}")
                
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
                logger.error(f"Semantic search failed: {e}")
                return {'error': str(e)}
                
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
                
            # Check if vector DB has embeddings
            try:
                stats = await self.vector_db.get_stats()
                if not stats or stats.get('total_vectors', 0) == 0:
                    return {
                        'error': 'No embeddings found. Run get_and_set_the_codebase_knowledge_foundation with enable_embeddings=true first.'
                    }
            except Exception as e:
                logger.warning(f"Could not check vector DB stats: {e}")
                
            try:
                # Try to find the entity by searching for it directly
                # Generate embedding for the entity path as a query
                path_embedding = await self.embedding_service.embed_text(f"Code file: {entity_path}")
                
                # Search for the entity itself first
                initial_results = await self.vector_db.search(
                    query_vector=path_embedding,
                    top_k=10,
                    filter={'path': entity_path} if hasattr(self.vector_db, 'filter') else None
                )
                
                # If we found the entity, use its embedding
                entity_embedding = None
                for result in initial_results:
                    if result.get('metadata', {}).get('path') == entity_path:
                        entity_embedding = result.get('values')
                        break
                        
                if not entity_embedding:
                    # Fallback: use the path embedding
                    entity_embedding = path_embedding
                    
                # Search for similar
                results = await self.vector_db.search(
                    query_vector=entity_embedding,
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
                    'total_found': len(similar),
                    'threshold_used': similarity_threshold
                }
                
            except Exception as e:
                logger.error(f"Similar code search failed: {e}")
                return {'error': str(e)}
                
    # Helper methods for analysis coordination
    def _extract_business_summary(self, business_logic: Dict[str, Any]) -> Dict[str, Any]:
        """Extract business summary from analysis"""
        return {
            'domain_entities': len(business_logic.get('domain_entities', [])),
            'business_rules': len(business_logic.get('business_rules', [])),
            'critical_operations': business_logic.get('critical_operations', [])[:5]  # Top 5
        }
        
    def _extract_aggregated_danger_zones(self, security: Dict[str, Any], 
                                       architecture: Dict[str, Any],
                                       business_logic: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and aggregate danger zones from multiple analyses"""
        danger_zones = []
        
        # From security analysis
        for vuln in security.get('vulnerabilities', []):
            if vuln.get('severity') in ['critical', 'high']:
                danger_zones.append({
                    'type': 'security_vulnerability',
                    'severity': vuln.get('severity'),
                    'file': vuln.get('file'),
                    'description': vuln.get('description'),
                    'line': vuln.get('line')
                })
                
        # From architecture analysis
        for entry_point in architecture.get('entry_points', []):
            danger_zones.append({
                'type': 'entry_point',
                'severity': 'medium',
                'file': entry_point,
                'description': 'Application entry point - changes affect entire application'
            })
            
        # From business logic
        for critical_op in business_logic.get('critical_operations', []):
            danger_zones.append({
                'type': 'critical_business_operation',
                'severity': 'high',
                'file': critical_op.get('file'),
                'description': critical_op.get('description', 'Critical business operation')
            })
            
        return danger_zones
        
    def _build_instant_context(self, dependency: Dict[str, Any], security: Dict[str, Any],
                             architecture: Dict[str, Any], business_logic: Dict[str, Any]) -> Dict[str, Any]:
        """Build instant context for AI consumption"""
        return {
            'summary': 'Essential codebase knowledge for safe modifications',
            'critical_facts': [
                f"Total files: {dependency.get('total_files', 0)}",
                f"Security issues: {len(security.get('vulnerabilities', []))}",
                f"Architecture layers: {len(architecture.get('layers', {}))}",
                f"Business rules: {len(business_logic.get('business_rules', []))}"
            ],
            'must_know': {
                'entry_points': architecture.get('entry_points', [])[:3],
                'critical_files': self._extract_critical_files(
                    self._extract_aggregated_danger_zones(security, architecture, business_logic)
                )[:5],
                'main_dependencies': list(dependency.get('external_dependencies', {}).keys())[:10]
            }
        }
        
    def _build_safe_modification_guide(self, security: Dict[str, Any], 
                                     architecture: Dict[str, Any],
                                     business_logic: Dict[str, Any]) -> List[str]:
        """Build safe modification guidelines"""
        guidelines = [
            "Always check impact analysis before modifying files",
            "Run tests after any changes",
            "Be extra careful with entry points and critical files"
        ]
        
        if security.get('vulnerabilities'):
            guidelines.append("Review security vulnerabilities in affected files")
            
        if business_logic.get('critical_operations'):
            guidelines.append("Ensure business logic integrity is maintained")
            
        return guidelines
        
    def _generate_understanding_guidance(self, score: int, feedback: List[str], 
                                       approved: bool) -> str:
        """Generate guidance based on understanding score"""
        if approved:
            return "Good understanding demonstrated. Proceed with caution and follow the checklist."
        elif score >= 40:
            return "Partial understanding shown. Review the feedback and enhance your plan."
        else:
            return "More analysis needed. Study the codebase context and danger zones first."
            
    def _suggest_next_steps(self, approved: bool, files: List[str] = None) -> List[str]:
        """Suggest next steps based on approval status"""
        if approved:
            steps = ["Implement changes following the modification guide"]
            if files:
                steps.append(f"Start with least critical files: {files[-1:]}") 
            steps.append("Run tests after each change")
        else:
            steps = [
                "Review get_ai_knowledge_package for comprehensive context",
                "Use get_impact_analysis on specific files",
                "Revise your implementation plan"
            ]
        return steps
        
    def _generate_file_checklist(self, file_path: str, impact: Dict[str, Any], 
                                agent_results: Dict[str, Any]) -> List[str]:
        """Generate file-specific checklist"""
        checklist = []
        
        # Basic checks
        checklist.append("□ Review all imports and dependencies")
        checklist.append("□ Check for existing tests")
        
        # Risk-based checks
        if impact.get('modification_risk') == 'high':
            checklist.append("□ Get code review before merging")
            checklist.append("□ Run integration tests")
            
        # Security checks
        if impact.get('security_concerns'):
            checklist.append("□ Security review required")
            checklist.append("□ Validate all inputs")
            
        return checklist
        
    def _get_file_specific_checklist(self, file_path: str, impact: Dict[str, Any], 
                                   agent_results: Dict[str, Any]) -> List[str]:
        """Get file-specific modification checklist"""
        return self._generate_file_checklist(file_path, impact, agent_results)
        
    def _suggest_safer_alternatives(self, file_path: str, impact: Dict[str, Any]) -> List[str]:
        """Suggest safer modification alternatives"""
        alternatives = []
        
        if impact.get('modification_risk') == 'high':
            alternatives.append("Consider creating a new function instead of modifying existing")
            alternatives.append("Add feature flags for gradual rollout")
            alternatives.append("Create comprehensive tests before making changes")
            
        return alternatives


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