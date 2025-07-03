#!/usr/bin/env python3
"""
Analysis methods for CodebaseIQ Pro
Contains all the analysis-related methods (dependency, security, architecture, etc.)
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import os
import networkx as nx

from .core import EnhancedCodeEntity
from .agents import (
    DependencyAnalysisAgent,
    SecurityAuditAgent,
    PatternDetectionAgent,
    VersionCompatibilityAgent,
    ArchitectureAnalysisAgent,
    PerformanceAnalysisAgent,
    EmbeddingAgent
)
from .agents.deep_understanding_agent import DeepUnderstandingAgent
from .agents.cross_file_intelligence import CrossFileIntelligence
from .agents.business_logic_extractor import BusinessLogicExtractor
from .agents.ai_knowledge_packager import AIKnowledgePackager

logger = logging.getLogger(__name__)


class AnalysisMethods:
    """Mixin class containing all analysis methods"""
    
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
            
    async def _get_security_analysis_full(self,
                                        force_refresh: bool = False,
                                        severity_filter: str = "all") -> Dict[str, Any]:
        """Get detailed security analysis (up to 25K tokens)"""
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
                        
            # Need full analysis - run security agent
            logger.info("Running security analysis...")
            security_data = await self._run_individual_agent('security', codebase_path)
            
            # Filter by severity if requested
            filtered_vulnerabilities = security_data.get('vulnerabilities', [])
            if severity_filter != "all":
                filtered_vulnerabilities = [
                    v for v in filtered_vulnerabilities
                    if v.get('severity', '').lower() == severity_filter.lower()
                ]
                
            # Build comprehensive response
            result = {
                'summary': {
                    'total_vulnerabilities': len(security_data.get('vulnerabilities', [])),
                    'critical': sum(1 for v in security_data.get('vulnerabilities', []) if v.get('severity') == 'critical'),
                    'high': sum(1 for v in security_data.get('vulnerabilities', []) if v.get('severity') == 'high'),
                    'medium': sum(1 for v in security_data.get('vulnerabilities', []) if v.get('severity') == 'medium'),
                    'low': sum(1 for v in security_data.get('vulnerabilities', []) if v.get('severity') == 'low'),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'vulnerabilities': filtered_vulnerabilities,
                'auth_mechanisms': security_data.get('auth_mechanisms', []),
                'input_validation': security_data.get('input_validation', {}),
                'dangerous_patterns': security_data.get('dangerous_patterns', []),
                'security_headers': security_data.get('security_headers', {}),
                'cryptography_usage': security_data.get('cryptography_usage', {}),
                'third_party_risks': security_data.get('third_party_risks', []),
                'danger_zones': self._extract_danger_zones(security_data),
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
        """Get complete architecture analysis (up to 25K tokens)"""
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
                        
            # Need full analysis - run architecture agent
            logger.info("Running architecture analysis...")
            arch_data = await self._run_individual_agent('architecture', codebase_path)
            
            # Build comprehensive response
            result = {
                'summary': {
                    'total_components': len(arch_data.get('components', {})),
                    'layers': list(arch_data.get('layers', {}).keys()),
                    'patterns_used': list(arch_data.get('patterns', {}).keys()),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'layers': arch_data.get('layers', {}),
                'components': arch_data.get('components', {}),
                'patterns': arch_data.get('patterns', {}),
                'entry_points': arch_data.get('entry_points', []),
                'data_flow': arch_data.get('data_flow', {}),
                'external_interfaces': arch_data.get('external_interfaces', []),
                'configuration_management': arch_data.get('configuration', {}),
                'deployment_structure': arch_data.get('deployment', {}),
                'latest_changes': self.cache_manager.get_latest_changes_summary(codebase_path, cached_data)
            }
            
            # Add diagrams if requested
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
            logger.info("Running business logic analysis...")
            
            # Run enhanced analysis which includes business logic
            enhanced_results = await self._run_enhanced_analysis(codebase_path)
            business_logic = enhanced_results.get('business_logic', {})
            cross_file_intel = enhanced_results.get('cross_file_intelligence', {})
            
            # Build comprehensive response
            result = {
                'summary': {
                    'domain_entities': len(business_logic.get('domain_entities', [])),
                    'business_rules': len(business_logic.get('business_rules', [])),
                    'workflows': len(business_logic.get('user_journeys', [])),
                    'critical_operations': len(business_logic.get('critical_operations', [])),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'domain_model': {
                    'entities': business_logic.get('domain_entities', []),
                    'relationships': business_logic.get('entity_relationships', []),
                    'value_objects': business_logic.get('value_objects', [])
                },
                'business_rules': business_logic.get('business_rules', []),
                'user_journeys': business_logic.get('user_journeys', []) if include_workflows else [],
                'critical_operations': business_logic.get('critical_operations', []),
                'data_validation_rules': business_logic.get('validation_rules', []),
                'business_constraints': business_logic.get('constraints', []),
                'api_contracts': cross_file_intel.get('api_boundaries', {}),
                'event_flows': cross_file_intel.get('event_flow', {}),
                'state_machines': business_logic.get('state_machines', []),
                'latest_changes': self.cache_manager.get_latest_changes_summary(codebase_path, cached_data)
            }
            
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