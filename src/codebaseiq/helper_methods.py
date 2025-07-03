#!/usr/bin/env python3
"""
Helper methods for CodebaseIQ Pro
Contains utility methods for file discovery, entity management, and analysis support
"""

import asyncio
import logging
import aiofiles
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

from .core import EnhancedCodeEntity
from .agents import (
    DependencyAnalysisAgent,
    SecurityAuditAgent,
    PatternDetectionAgent,
    VersionCompatibilityAgent,
    ArchitectureAnalysisAgent,
    PerformanceAnalysisAgent
)
from .agents.deep_understanding_agent import DeepUnderstandingAgent
from .agents.cross_file_intelligence import CrossFileIntelligence
from .agents.business_logic_extractor import BusinessLogicExtractor

logger = logging.getLogger(__name__)


class HelperMethods:
    """Mixin class containing helper methods"""
    
    async def _discover_files(self, root_path: Path) -> Dict[str, Path]:
        """Discover all relevant code files in the directory"""
        file_map = {}
        exclude_patterns = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv',
            'dist', 'build', '.pytest_cache', '.mypy_cache', 'coverage',
            '.idea', '.vscode', '*.pyc', '*.pyo', '.DS_Store'
        }
        
        for path in root_path.rglob('*'):
            # Skip if any exclude pattern matches
            if any(pattern in str(path) for pattern in exclude_patterns):
                continue
                
            if path.is_file() and path.suffix in [
                '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go',
                '.rs', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.php',
                '.swift', '.kt', '.scala', '.r', '.m', '.mm', '.vue',
                '.json', '.yaml', '.yml', '.toml', '.xml', '.sql'
            ]:
                rel_path = str(path.relative_to(root_path))
                file_map[rel_path] = path
                
        logger.info(f"Discovered {len(file_map)} code files")
        return file_map
        
    def _determine_entity_type(self, path: str) -> str:
        """Determine the type of code entity based on file path"""
        path_lower = path.lower()
        
        if 'test' in path_lower:
            return 'test'
        elif 'config' in path_lower or path.endswith(('.json', '.yaml', '.yml', '.toml')):
            return 'config'
        elif 'model' in path_lower or 'entity' in path_lower:
            return 'model'
        elif 'service' in path_lower or 'controller' in path_lower:
            return 'service'
        elif 'util' in path_lower or 'helper' in path_lower:
            return 'utility'
        elif 'component' in path_lower or path.endswith(('.jsx', '.tsx', '.vue')):
            return 'component'
        elif path.endswith(('.sql',)):
            return 'database'
        else:
            return 'module'
            
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
                async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                    file_contents[rel_path] = await f.read()
            except Exception as e:
                logger.warning(f"Failed to read {rel_path}: {e}")
                
        # Run enhanced agents
        context = {
            'root_path': codebase_path,
            'file_map': file_map,
            'file_contents': file_contents
        }
        
        # Run the three enhanced agents
        deep_agent = DeepUnderstandingAgent()
        cross_file_agent = CrossFileIntelligence()
        business_agent = BusinessLogicExtractor()
        
        deep_results = await deep_agent.analyze(context)
        cross_file_results = await cross_file_agent.analyze(context)
        business_results = await business_agent.analyze(context)
        
        return {
            'deep_analysis': deep_results,
            'cross_file_intelligence': cross_file_results,
            'business_logic': business_results
        }
        
    async def _run_with_timeout(self, coro, timeout_seconds: int, task_name: str):
        """Run a coroutine with timeout and logging"""
        try:
            logger.info(f"Starting {task_name}...")
            result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            logger.info(f"✅ {task_name} completed successfully")
            return result
        except asyncio.TimeoutError:
            logger.error(f"❌ {task_name} timed out after {timeout_seconds}s")
            return {'error': f'{task_name} timed out', 'timeout': timeout_seconds}
        except Exception as e:
            logger.error(f"❌ {task_name} failed: {e}")
            return {'error': str(e)}
            
    def _extract_file_dependencies(self, dependency_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract per-file dependency information"""
        # This would need the actual implementation from dependency data
        return {}
        
    def _calculate_transitive_dependencies(self, dependency_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate transitive dependencies from dependency graph"""
        # This would need the actual implementation
        return {}
        
    def _extract_danger_zones(self, security_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract danger zones from security analysis"""
        danger_zones = []
        
        # Extract from vulnerabilities
        for vuln in security_data.get('vulnerabilities', []):
            if vuln.get('severity') in ['critical', 'high']:
                danger_zones.append({
                    'file': vuln.get('file'),
                    'type': vuln.get('type'),
                    'severity': vuln.get('severity'),
                    'description': vuln.get('description'),
                    'line': vuln.get('line')
                })
                
        # Extract from dangerous patterns
        for pattern in security_data.get('dangerous_patterns', []):
            danger_zones.append({
                'file': pattern.get('file'),
                'type': 'dangerous_pattern',
                'severity': 'high',
                'description': pattern.get('pattern'),
                'line': pattern.get('line')
            })
            
        return danger_zones
        
    async def _calculate_language_breakdown(self, file_map: Dict[str, Path]) -> Dict[str, int]:
        """Calculate breakdown of files by language"""
        language_counts = defaultdict(int)
        
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
            '.vue': 'Vue',
            '.sql': 'SQL'
        }
        
        for rel_path in file_map:
            ext = Path(rel_path).suffix.lower()
            lang = ext_to_lang.get(ext, 'Other')
            language_counts[lang] += 1
            
        return dict(language_counts)
        
    async def _extract_env_variables(self, file_map: Dict[str, Path]) -> List[str]:
        """Extract environment variables from the codebase"""
        env_vars = set()
        
        # Look for .env files and common config files
        for rel_path, full_path in file_map.items():
            if '.env' in rel_path or rel_path.endswith(('.env', '.env.example')):
                try:
                    async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        for line in content.splitlines():
                            if '=' in line and not line.strip().startswith('#'):
                                var_name = line.split('=')[0].strip()
                                if var_name:
                                    env_vars.add(var_name)
                except Exception as e:
                    logger.debug(f"Failed to read env file {rel_path}: {e}")
                    
        return sorted(list(env_vars))
        
    async def _detect_config_files(self, file_map: Dict[str, Path]) -> List[str]:
        """Detect configuration files in the codebase"""
        config_files = []
        
        config_patterns = [
            'config', 'settings', '.env', 'environment',
            'application.properties', 'appsettings.json'
        ]
        
        for rel_path in file_map:
            path_lower = rel_path.lower()
            if any(pattern in path_lower for pattern in config_patterns):
                config_files.append(rel_path)
            elif rel_path.endswith(('.json', '.yaml', '.yml', '.toml', '.ini', '.conf')):
                config_files.append(rel_path)
                
        return sorted(config_files)
        
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
        
    def _summarize_danger_zones(self, danger_zones: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize danger zones for quick understanding"""
        if not danger_zones:
            return {'total': 0, 'critical': 0, 'high': 0, 'files': []}
            
        # Count by severity
        severity_counts = defaultdict(int)
        affected_files = set()
        
        for zone in danger_zones.get('zones', []):
            severity = zone.get('severity', 'unknown')
            severity_counts[severity] += 1
            if 'file' in zone:
                affected_files.add(zone['file'])
                
        return {
            'total': len(danger_zones.get('zones', [])),
            'critical': severity_counts.get('critical', 0),
            'high': severity_counts.get('high', 0),
            'medium': severity_counts.get('medium', 0),
            'files': sorted(list(affected_files))
        }
        
    def _extract_critical_files(self, danger_zones: Dict[str, Any]) -> List[str]:
        """Extract list of critical files from danger zones"""
        critical_files = set()
        
        for zone in danger_zones.get('zones', []):
            if zone.get('severity') in ['critical', 'high'] and 'file' in zone:
                critical_files.add(zone['file'])
                
        return sorted(list(critical_files))