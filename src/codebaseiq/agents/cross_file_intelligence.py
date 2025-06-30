#!/usr/bin/env python3
"""
Cross-File Intelligence System
Maps relationships between files to prevent AI from making breaking changes.
Provides immediate awareness of impact zones and danger areas.
"""

import re
import os
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
import logging
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

class CrossFileIntelligence:
    """
    Maps relationships between files to understand the complete system.
    This is CRITICAL for preventing AI from making changes that break other parts.
    """
    
    def __init__(self):
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # Who imports this file
        self.function_calls: Dict[str, List[Dict]] = defaultdict(list)
        self.class_inheritance: Dict[str, Dict[str, Any]] = {}
        self.shared_globals: Dict[str, Set[str]] = defaultdict(set)
        self.execution_paths: List[List[str]] = []
        self.critical_paths: Set[str] = set()
        self.change_risk_map: Dict[str, Dict[str, Any]] = {}
        
    def analyze_relationships(self, file_contexts: Dict[str, Any], 
                            file_contents: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze relationships between all files in the codebase.
        This provides AI with immediate understanding of change impacts.
        """
        logger.info("🔗 Analyzing cross-file relationships for AI safety...")
        
        # Phase 1: Build import graph
        self._build_import_graph(file_contexts, file_contents)
        
        # Phase 2: Identify execution paths and entry points
        self._trace_execution_paths(file_contexts, file_contents)
        
        # Phase 3: Find shared resources and state
        self._find_shared_resources(file_contexts, file_contents)
        
        # Phase 4: Map API boundaries and interfaces
        api_boundaries = self._map_api_boundaries(file_contexts)
        
        # Phase 5: Calculate impact zones with AI-specific risk assessment
        impact_zones = self._calculate_impact_zones()
        
        # Phase 6: Identify critical interfaces that many files depend on
        critical_interfaces = self._identify_critical_interfaces(file_contexts)
        
        # Phase 7: Generate change risk map for AI guidance
        self._generate_change_risk_map(file_contexts, impact_zones)
        
        # Phase 8: Find circular dependencies (high risk for changes)
        circular_deps = self._find_circular_dependencies()
        
        return {
            "import_graph": dict(self.import_graph),
            "reverse_dependencies": dict(self.reverse_graph),
            "execution_paths": self.execution_paths,
            "api_boundaries": api_boundaries,
            "impact_zones": impact_zones,
            "critical_interfaces": critical_interfaces,
            "integration_points": self._find_integration_points(file_contexts),
            "shared_state": dict(self.shared_globals),
            "circular_dependencies": circular_deps,
            "change_risk_map": self.change_risk_map,
            "ai_modification_guidance": self._generate_ai_modification_guidance(
                impact_zones, critical_interfaces, circular_deps
            )
        }
        
    def _build_import_graph(self, file_contexts: Dict[str, Any], 
                           file_contents: Optional[Dict[str, str]] = None):
        """Build comprehensive import dependency graph"""
        
        for file_path, context in file_contexts.items():
            # Get the directory of the current file for relative imports
            file_dir = os.path.dirname(file_path)
            
            for dep in context.dependencies:
                if dep.startswith('import:') or dep.startswith('from:'):
                    # Parse the import
                    import_type, import_path = dep.split(':', 1)
                    
                    # Clean up the import path
                    if import_type == 'from' and '.' in import_path:
                        parts = import_path.split('.')
                        # Remove the specific import (last part) to get module
                        module_path = '.'.join(parts[:-1]) if len(parts) > 1 else parts[0]
                    else:
                        module_path = import_path
                        
                    # Try to resolve to actual files
                    possible_files = self._resolve_import_to_files(
                        module_path, file_path, file_dir, file_contexts
                    )
                    
                    for resolved_file in possible_files:
                        if resolved_file in file_contexts:
                            self.import_graph[file_path].add(resolved_file)
                            self.reverse_graph[resolved_file].add(file_path)
                            
    def _resolve_import_to_files(self, import_path: str, current_file: str, 
                                current_dir: str, file_contexts: Dict[str, Any]) -> List[str]:
        """Intelligently resolve import statements to actual file paths"""
        possible_files = []
        
        # Handle relative imports
        if import_path.startswith('.'):
            # Count leading dots for level
            level = len(import_path) - len(import_path.lstrip('.'))
            import_path = import_path.lstrip('.')
            
            # Go up directories based on level
            base_dir = current_dir
            for _ in range(level - 1):
                base_dir = os.path.dirname(base_dir)
                
            if import_path:
                base_path = os.path.join(base_dir, import_path.replace('.', '/'))
            else:
                base_path = base_dir
        else:
            # Absolute import - try multiple common locations
            base_paths = [
                import_path.replace('.', '/'),
                f"src/{import_path.replace('.', '/')}",
                f"lib/{import_path.replace('.', '/')}",
                f"app/{import_path.replace('.', '/')}",
            ]
            
            # Check each possible base path
            for base in base_paths:
                candidates = [
                    f"{base}.py",
                    f"{base}/__init__.py",
                    f"{base}/index.py",
                    f"{base}.js",
                    f"{base}/index.js",
                    f"{base}.ts",
                    f"{base}/index.ts",
                ]
                
                # Check if any candidate exists in file_contexts
                for candidate in candidates:
                    # Normalize path
                    normalized = os.path.normpath(candidate)
                    # Check both with and without leading slash
                    if normalized in file_contexts or f"/{normalized}" in file_contexts:
                        possible_files.append(normalized)
                    # Check if any file in contexts matches this pattern
                    for ctx_file in file_contexts:
                        if ctx_file.endswith(normalized) or normalized in ctx_file:
                            possible_files.append(ctx_file)
                            
            return list(set(possible_files))
            
        # For relative imports, check standard patterns
        candidates = [
            f"{base_path}.py",
            f"{base_path}/__init__.py",
            f"{base_path}/index.py",
            f"{base_path}.js",
            f"{base_path}/index.js",
        ]
        
        for candidate in candidates:
            normalized = os.path.normpath(candidate)
            if normalized in file_contexts:
                possible_files.append(normalized)
                
        return possible_files
        
    def _trace_execution_paths(self, file_contexts: Dict[str, Any], 
                              file_contents: Optional[Dict[str, str]] = None):
        """Trace execution flow to understand runtime dependencies"""
        
        # Find entry points
        entry_points = []
        
        for file_path, context in file_contexts.items():
            # Check for main functions or entry markers
            is_entry = False
            
            # Check critical functions for entry points
            for func in context.critical_functions:
                if func['name'] in ['main', '__main__', 'run', 'start', 'serve', 'index']:
                    entry_points.append((file_path, func['name']))
                    is_entry = True
                    break
                    
            # Check for script patterns
            if not is_entry and file_contents and file_path in file_contents:
                content = file_contents[file_path]
                if 'if __name__ == "__main__"' in content or 'if __name__ == \'__main__\':' in content:
                    entry_points.append((file_path, '__main__'))
                    is_entry = True
                    
            # Check for server/app initialization
            if not is_entry and any(indicator in context.purpose.lower() 
                                  for indicator in ['server', 'app', 'main', 'entry', 'start']):
                entry_points.append((file_path, 'inferred_entry'))
                
        # Trace execution from each entry point
        for entry_file, entry_func in entry_points:
            path = self._trace_single_execution_path(entry_file, entry_func, file_contexts)
            if path:
                self.execution_paths.append(path)
                # Mark all files in execution paths as critical
                for step in path:
                    if ':' in step:
                        file_part = step.split(':')[0]
                        self.critical_paths.add(file_part)
                        
    def _trace_single_execution_path(self, start_file: str, start_func: str, 
                                   file_contexts: Dict[str, Any], 
                                   visited: Optional[Set[str]] = None,
                                   depth: int = 0) -> List[str]:
        """Trace a single execution path with cycle detection"""
        if visited is None:
            visited = set()
            
        if depth > 20:  # Prevent infinite recursion
            return []
            
        if start_file in visited:
            return []  # Cycle detected
            
        visited.add(start_file)
        path = [f"{start_file}:{start_func}"]
        
        # Follow imports and add to path
        if start_file in self.import_graph:
            # Sort imports by criticality
            imports = sorted(self.import_graph[start_file], 
                           key=lambda f: len(self.reverse_graph.get(f, [])), 
                           reverse=True)
            
            for imported_file in imports[:5]:  # Limit depth
                if imported_file in file_contexts:
                    sub_path = self._trace_single_execution_path(
                        imported_file, "imported", file_contexts, visited.copy(), depth + 1
                    )
                    if sub_path:
                        path.extend(sub_path)
                        
        return path
        
    def _find_shared_resources(self, file_contexts: Dict[str, Any], 
                              file_contents: Optional[Dict[str, str]] = None):
        """Find resources shared between files (config, globals, singletons)"""
        
        if not file_contents:
            return
            
        # Patterns indicating shared resources
        shared_patterns = [
            (r'\b(config|CONFIG|Config)\b\.\w+', 'configuration'),
            (r'\b(settings|SETTINGS|Settings)\b\.\w+', 'settings'),
            (r'os\.environ\[[\'"](.*?)[\'"]\]', 'environment'),
            (r'process\.env\.([\w_]+)', 'environment'),
            (r'@singleton|Singleton', 'singleton'),
            (r'getInstance\(\)', 'singleton'),
            (r'\bglobal\s+(\w+)', 'global_variable'),
            (r'(cache|CACHE|Cache)\.\w+', 'cache'),
            (r'(logger|LOGGER|Logger)\.\w+', 'logging'),
            (r'(db|DB|database|DATABASE)\.\w+', 'database'),
        ]
        
        for file_path, content in file_contents.items():
            for pattern, resource_type in shared_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    self.shared_globals[resource_type].add(file_path)
                    # If many files share this resource, it's critical
                    if len(self.shared_globals[resource_type]) > 3:
                        self.critical_paths.add(file_path)
                        
    def _map_api_boundaries(self, file_contexts: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """Map API boundaries with detailed endpoint information"""
        
        api_boundaries = {
            "rest_endpoints": [],
            "graphql_resolvers": [],
            "rpc_methods": [],
            "cli_commands": [],
            "event_handlers": [],
            "websocket_handlers": [],
            "grpc_services": []
        }
        
        for file_path, context in file_contexts.items():
            # REST endpoints
            if any(indicator in str(context.side_effects).lower() or 
                   indicator in context.business_logic.lower()
                   for indicator in ['api', 'endpoint', 'route', 'rest']):
                api_boundaries["rest_endpoints"].append({
                    "file": file_path,
                    "purpose": context.purpose,
                    "risk": "HIGH" if context.security_concerns else "MEDIUM"
                })
                
            # GraphQL
            if any('graphql' in dep.lower() or 'resolver' in dep.lower() 
                   for dep in context.dependencies):
                api_boundaries["graphql_resolvers"].append({
                    "file": file_path,
                    "purpose": context.purpose,
                    "risk": "HIGH"  # GraphQL always high risk due to query complexity
                })
                
            # CLI commands
            if any(cli_lib in str(context.dependencies).lower() 
                   for cli_lib in ['click', 'argparse', 'fire', 'commander']):
                api_boundaries["cli_commands"].append({
                    "file": file_path,
                    "purpose": context.purpose,
                    "risk": "MEDIUM"
                })
                
            # Event handlers
            if any(event_indicator in func['name'].lower()
                   for func in context.critical_functions
                   for event_indicator in ['event', 'handler', 'listener', 'on_', 'handle_']):
                api_boundaries["event_handlers"].append({
                    "file": file_path,
                    "purpose": context.purpose,
                    "risk": "MEDIUM"
                })
                
            # WebSocket handlers
            if any(ws in str(context.dependencies).lower() 
                   for ws in ['websocket', 'socket.io', 'ws']):
                api_boundaries["websocket_handlers"].append({
                    "file": file_path,
                    "purpose": context.purpose,
                    "risk": "HIGH"  # Real-time = high risk
                })
                
        return api_boundaries
        
    def _calculate_impact_zones(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate comprehensive impact zones for each file.
        This is CRITICAL for AI to understand change consequences.
        """
        impact_zones = {}
        
        for file_path in set(list(self.import_graph.keys()) + list(self.reverse_graph.keys())):
            # Direct impact: Files that import this one
            direct_impact = list(self.reverse_graph.get(file_path, set()))
            
            # Calculate indirect impact using BFS
            indirect_impact = set()
            visited = {file_path}
            queue = deque(direct_impact)
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                
                # Add files that import the current file
                next_level = self.reverse_graph.get(current, set())
                for next_file in next_level:
                    if next_file not in direct_impact and next_file != file_path:
                        indirect_impact.add(next_file)
                        queue.append(next_file)
                        
            indirect_impact = list(indirect_impact)
            
            # Calculate risk metrics
            direct_count = len(direct_impact)
            indirect_count = len(indirect_impact)
            is_in_critical_path = file_path in self.critical_paths
            is_shared_resource = any(file_path in files for files in self.shared_globals.values())
            
            # Risk scoring with AI-specific considerations
            risk_score = 0
            risk_factors = []
            
            if direct_count > 0:
                risk_score += direct_count * 3
                risk_factors.append(f"{direct_count} direct dependencies")
                
            if indirect_count > 0:
                risk_score += indirect_count
                risk_factors.append(f"{indirect_count} indirect dependencies")
                
            if is_in_critical_path:
                risk_score += 10
                risk_factors.append("In critical execution path")
                
            if is_shared_resource:
                risk_score += 8
                risk_factors.append("Manages shared state")
                
            # Special cases that increase risk
            if any(critical in file_path.lower() for critical in ['auth', 'security', 'payment', 'config']):
                risk_score += 5
                risk_factors.append("Security/Critical domain")
                
            # Determine risk level
            if risk_score >= 20:
                risk_level = "CRITICAL"
                ai_warning = "⛔ DO NOT MODIFY without extensive analysis and testing"
            elif risk_score >= 10:
                risk_level = "HIGH"
                ai_warning = "⚠️ Modifications require careful review of all dependencies"
            elif risk_score >= 5:
                risk_level = "MEDIUM"
                ai_warning = "⚡ Check impact on dependent files before modifying"
            else:
                risk_level = "LOW"
                ai_warning = "✓ Safe to modify with standard precautions"
                
            impact_zones[file_path] = {
                "direct_impact": direct_impact,
                "indirect_impact": indirect_impact,
                "total_impact": direct_count + indirect_count,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "ai_warning": ai_warning,
                "is_critical_path": is_in_critical_path,
                "is_shared_resource": is_shared_resource,
                "modification_strategy": self._suggest_modification_strategy(risk_level, risk_factors)
            }
            
        return impact_zones
        
    def _suggest_modification_strategy(self, risk_level: str, risk_factors: List[str]) -> str:
        """Suggest safe modification strategies for AI"""
        if risk_level == "CRITICAL":
            return "Create new functions instead of modifying existing ones. Use feature flags for gradual rollout."
        elif risk_level == "HIGH":
            return "Consider creating wrapper functions. Ensure backward compatibility. Add comprehensive tests first."
        elif risk_level == "MEDIUM":
            return "Test all direct dependencies after changes. Consider refactoring if interface needs to change."
        else:
            return "Standard modification approach. Ensure tests pass."
            
    def _identify_critical_interfaces(self, file_contexts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify interfaces that many files depend on.
        These require extreme caution when modifying.
        """
        critical_interfaces = []
        
        # Analyze each file's role as an interface
        for file_path, dependents in self.reverse_graph.items():
            dependent_count = len(dependents)
            
            if dependent_count >= 2:  # Lower threshold for better coverage
                context = file_contexts.get(file_path, {})
                
                # Analyze what makes this interface critical
                interface_info = {
                    "file": file_path,
                    "dependent_count": dependent_count,
                    "dependents": sorted(list(dependents))[:10],  # Top 10
                    "interface_type": self._determine_interface_type(file_path, context),
                    "exported_items": self._extract_exported_items(context),
                    "stability_score": self._calculate_stability_score(file_path, context),
                    "ai_guidance": self._generate_interface_guidance(dependent_count, file_path)
                }
                
                critical_interfaces.append(interface_info)
                
        # Sort by criticality (dependent count × stability importance)
        critical_interfaces.sort(
            key=lambda x: x['dependent_count'] * (3 if 'config' in x['file'].lower() else 1), 
            reverse=True
        )
        
        return critical_interfaces[:20]  # Top 20 most critical
        
    def _determine_interface_type(self, file_path: str, context: Any) -> str:
        """Determine what type of interface this file provides"""
        file_lower = file_path.lower()
        # Handle both dict and object contexts
        if hasattr(context, 'purpose'):
            purpose_lower = str(context.purpose).lower()
        else:
            purpose_lower = str(context.get('purpose', '')).lower()
        
        if 'config' in file_lower or 'settings' in file_lower:
            return "Configuration Interface"
        elif 'model' in file_lower or 'schema' in file_lower:
            return "Data Model Interface"
        elif 'service' in file_lower or 'manager' in file_lower:
            return "Service Interface"
        elif 'util' in file_lower or 'helper' in file_lower:
            return "Utility Interface"
        elif 'api' in purpose_lower or 'endpoint' in purpose_lower:
            return "API Interface"
        elif 'component' in file_lower or 'widget' in file_lower:
            return "UI Component Interface"
        else:
            return "General Interface"
            
    def _extract_exported_items(self, context: Any) -> List[str]:
        """Extract what this file exports (functions, classes, etc.)"""
        exported = []
        
        # From critical functions
        critical_funcs = context.critical_functions if hasattr(context, 'critical_functions') else context.get('critical_functions', [])
        for func in critical_funcs[:5]:
            if func.get('is_exported') or not func['name'].startswith('_'):
                exported.append(f"function: {func['name']}")
                
        # Could be enhanced to detect classes, constants, etc.
        return exported
        
    def _calculate_stability_score(self, file_path: str, context: Any) -> int:
        """Calculate how stable this interface should be (higher = more stable)"""
        score = 0
        
        # Files with many dependents should be stable
        score += min(len(self.reverse_graph.get(file_path, [])) * 2, 20)
        
        # Critical domains should be stable
        if any(critical in file_path.lower() for critical in ['config', 'auth', 'model', 'schema']):
            score += 10
            
        # Files with complex functions should be stable
        critical_funcs = context.critical_functions if hasattr(context, 'critical_functions') else context.get('critical_functions', [])
        if critical_funcs:
            avg_complexity = sum(f.get('complexity', 1) for f in critical_funcs) / len(critical_funcs)
            score += int(avg_complexity * 2)
            
        return min(score, 100)  # Cap at 100
        
    def _generate_interface_guidance(self, dependent_count: int, file_path: str) -> str:
        """Generate specific guidance for modifying interfaces"""
        if dependent_count > 10:
            return f"⛔ EXTREME CAUTION: {dependent_count} files depend on this. Any change requires updating ALL dependents. Consider deprecation strategy instead of direct modification."
        elif dependent_count > 5:
            return f"⚠️ HIGH IMPACT: {dependent_count} files depend on this. Create new methods instead of changing existing signatures. Use adapters if interface must change."
        elif dependent_count > 2:
            return f"⚡ MODERATE IMPACT: {dependent_count} files depend on this. Ensure backward compatibility or update all dependents atomically."
        else:
            return f"✓ LOW IMPACT: {dependent_count} files depend on this. Standard refactoring practices apply."
            
    def _find_integration_points(self, file_contexts: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """Find where the codebase integrates with external systems"""
        integrations = {
            "database": [],
            "http_apis": [],
            "file_system": [],
            "message_queues": [],
            "caches": [],
            "external_services": [],
            "authentication_providers": []
        }
        
        for file_path, context in file_contexts.items():
            # Check side effects and dependencies for integration patterns
            side_effects_str = ' '.join(context.side_effects)
            deps_str = ' '.join(context.dependencies)
            combined = f"{side_effects_str} {deps_str}".lower()
            
            # Database integrations
            if any(db in combined for db in ['database', 'sql', 'orm', 'mongoose', 'sequelize', 'prisma']):
                integrations["database"].append({
                    "file": file_path,
                    "type": "Database Operations",
                    "risk": "HIGH" if 'migration' in file_path else "MEDIUM"
                })
                
            # HTTP/API integrations  
            if any(http in combined for http in ['http', 'request', 'axios', 'fetch', 'api']):
                integrations["http_apis"].append({
                    "file": file_path,
                    "type": "HTTP/API Calls",
                    "risk": "MEDIUM"
                })
                
            # File system
            if 'file' in side_effects_str.lower():
                integrations["file_system"].append({
                    "file": file_path,
                    "type": "File I/O",
                    "risk": "LOW"
                })
                
            # Message queues
            if any(mq in combined for mq in ['rabbitmq', 'kafka', 'sqs', 'pubsub', 'redis', 'bull']):
                integrations["message_queues"].append({
                    "file": file_path,
                    "type": "Message Queue",
                    "risk": "HIGH"  # Async = complex
                })
                
            # External services
            if any(svc in combined for svc in ['stripe', 'twilio', 'sendgrid', 'aws', 'gcp', 'azure']):
                integrations["external_services"].append({
                    "file": file_path,
                    "type": "Third-party Service",
                    "risk": "HIGH"  # External deps = high risk
                })
                
        return integrations
        
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies which are high-risk for modifications"""
        circular_deps = []
        visited = set()
        
        def find_cycle(node: str, path: List[str], visiting: Set[str]) -> Optional[List[str]]:
            if node in visiting:
                # Found a cycle
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
                
            if node in visited:
                return None
                
            visiting.add(node)
            path.append(node)
            
            for neighbor in self.import_graph.get(node, []):
                cycle = find_cycle(neighbor, path.copy(), visiting.copy())
                if cycle:
                    return cycle
                    
            visiting.remove(node)
            visited.add(node)
            return None
            
        # Check each node for cycles
        for node in self.import_graph:
            if node not in visited:
                cycle = find_cycle(node, [], set())
                if cycle and len(cycle) > 2:  # Ignore self-references
                    # Normalize cycle (start with smallest element)
                    min_idx = cycle.index(min(cycle))
                    normalized = cycle[min_idx:] + cycle[:min_idx]
                    if normalized not in circular_deps:
                        circular_deps.append(normalized)
                        
        return circular_deps
        
    def _generate_change_risk_map(self, file_contexts: Dict[str, Any], 
                                 impact_zones: Dict[str, Dict[str, Any]]):
        """Generate a comprehensive risk map for each file"""
        
        for file_path, context in file_contexts.items():
            impact = impact_zones.get(file_path, {})
            
            # Combine all risk factors
            risk_factors = []
            
            # From impact analysis
            if impact.get('risk_level') in ['CRITICAL', 'HIGH']:
                risk_factors.append(f"Impact Risk: {impact['risk_level']}")
                
            # From file analysis - handle both dict and object
            mod_risk = context.modification_risk if hasattr(context, 'modification_risk') else context.get('modification_risk', '')
            if 'CRITICAL' in mod_risk or 'HIGH' in mod_risk:
                risk_factors.append(f"File Risk: {mod_risk}")
                
            # From security analysis
            sec_concerns = context.security_concerns if hasattr(context, 'security_concerns') else context.get('security_concerns', [])
            if len(sec_concerns) > 2:
                risk_factors.append(f"Security Concerns: {len(sec_concerns)}")
                
            # From circular dependencies
            in_cycle = any(file_path in cycle for cycle in self._find_circular_dependencies())
            if in_cycle:
                risk_factors.append("Circular Dependency")
                
            # Calculate overall risk
            if len(risk_factors) >= 3 or 'CRITICAL' in str(risk_factors):
                overall_risk = "CRITICAL"
            elif len(risk_factors) >= 2 or 'HIGH' in str(risk_factors):
                overall_risk = "HIGH"
            elif len(risk_factors) >= 1:
                overall_risk = "MEDIUM"
            else:
                overall_risk = "LOW"
                
            self.change_risk_map[file_path] = {
                "overall_risk": overall_risk,
                "risk_factors": risk_factors,
                "safe_change_checklist": self._generate_safe_change_checklist(overall_risk, context, impact),
                "alternative_approaches": self._suggest_alternatives(overall_risk, context)
            }
            
    def _generate_safe_change_checklist(self, risk_level: str, context: Any, 
                                       impact: Dict[str, Any]) -> List[str]:
        """Generate a checklist for safe modifications"""
        checklist = ["✓ Read and understand the current implementation"]
        
        if risk_level in ["CRITICAL", "HIGH"]:
            checklist.extend([
                "✓ Identify ALL files in the impact zone",
                "✓ Create comprehensive tests BEFORE making changes",
                "✓ Consider using feature flags for gradual rollout",
                "✓ Document the reason for changes",
                "✓ Plan rollback strategy"
            ])
            
        if impact.get('total_impact', 0) > 5:
            checklist.append(f"✓ Review and update {impact['total_impact']} dependent files")
            
        sec_concerns = context.security_concerns if hasattr(context, 'security_concerns') else context.get('security_concerns', [])
        if sec_concerns:
            checklist.append("✓ Security review required - check for vulnerabilities")
            
        err_handling = context.error_handling if hasattr(context, 'error_handling') else context.get('error_handling', [])
        if err_handling:
            checklist.append("✓ Preserve error handling logic and test error cases")
            
        checklist.append("✓ Run full test suite including integration tests")
        
        return checklist
        
    def _suggest_alternatives(self, risk_level: str, context: Any) -> List[str]:
        """Suggest safer alternatives to direct modification"""
        alternatives = []
        
        if risk_level == "CRITICAL":
            alternatives.extend([
                "Create a new version of this module instead of modifying",
                "Use adapter pattern to wrap existing functionality",
                "Implement changes behind a feature flag",
                "Consider deprecation strategy over direct changes"
            ])
        elif risk_level == "HIGH":
            alternatives.extend([
                "Add new methods instead of changing existing ones",
                "Use dependency injection to swap implementations",
                "Create abstraction layer for safer changes"
            ])
            
        language = context.language if hasattr(context, 'language') else context.get('language', '')
        if language == "python":
            alternatives.append("Use `warnings.deprecated` for gradual migration")
        elif language in ["javascript", "typescript"]:
            alternatives.append("Use JSDoc @deprecated for gradual migration")
            
        return alternatives
        
    def _generate_ai_modification_guidance(self, impact_zones: Dict[str, Dict[str, Any]], 
                                         critical_interfaces: List[Dict[str, Any]],
                                         circular_deps: List[List[str]]) -> Dict[str, Any]:
        """
        Generate comprehensive guidance for AI assistants.
        This is the key output that prevents breaking changes.
        """
        
        # Categorize files by modification risk
        do_not_modify = []
        extreme_caution = []
        careful_review = []
        safe_with_tests = []
        
        for file_path, impact in impact_zones.items():
            risk_map = self.change_risk_map.get(file_path, {})
            
            if impact['risk_level'] == 'CRITICAL' or risk_map.get('overall_risk') == 'CRITICAL':
                if impact['total_impact'] > 20:
                    do_not_modify.append({
                        'file': file_path,
                        'reason': f"{impact['total_impact']} files depend on this",
                        'alternatives': risk_map.get('alternative_approaches', [])
                    })
                else:
                    extreme_caution.append({
                        'file': file_path,
                        'impact_count': impact['total_impact'],
                        'checklist': risk_map.get('safe_change_checklist', [])
                    })
            elif impact['risk_level'] == 'HIGH':
                careful_review.append({
                    'file': file_path,
                    'dependencies': impact['direct_impact'][:5],
                    'strategy': impact.get('modification_strategy', '')
                })
            else:
                safe_with_tests.append(file_path)
                
        # Generate summary statistics
        total_files = len(impact_zones)
        high_risk_count = len(do_not_modify) + len(extreme_caution)
        circular_count = len(circular_deps)
        
        return {
            "summary": {
                "total_files": total_files,
                "high_risk_files": high_risk_count,
                "circular_dependencies": circular_count,
                "critical_interfaces": len(critical_interfaces)
            },
            "risk_categories": {
                "do_not_modify": do_not_modify[:10],  # Top 10
                "extreme_caution": extreme_caution[:10],
                "careful_review": careful_review[:20],
                "safe_with_tests": len(safe_with_tests)  # Just count
            },
            "circular_dependencies": [
                {
                    "cycle": cycle,
                    "risk": "HIGH - Breaking this cycle requires coordinated changes",
                    "suggestion": "Consider dependency injection or interfaces to break the cycle"
                }
                for cycle in circular_deps[:5]  # Top 5 cycles
            ],
            "critical_interfaces": [
                {
                    "file": intf['file'],
                    "dependents": intf['dependent_count'],
                    "guidance": intf['ai_guidance']
                }
                for intf in critical_interfaces[:10]  # Top 10
            ],
            "general_rules": [
                "1. ALWAYS check impact zones before modifying any file",
                "2. NEVER modify files in 'do_not_modify' category without explicit permission",
                "3. For HIGH risk files, create new functions instead of modifying existing ones",
                "4. Run ALL tests for files in the impact zone, not just the modified file",
                "5. When in doubt, ask for human review before making changes",
                "6. Prefer additive changes (new functions/classes) over modifications",
                "7. Document WHY you're making changes, not just what changed"
            ],
            "quick_reference": self._generate_quick_reference(impact_zones)
        }
        
    def _generate_quick_reference(self, impact_zones: Dict[str, Dict[str, Any]]) -> str:
        """Generate a quick reference guide for AI"""
        high_risk_files = [f for f, zone in impact_zones.items() 
                          if zone['risk_level'] in ['CRITICAL', 'HIGH']]
        
        return f"""
🚨 QUICK RISK REFERENCE:
- Total files analyzed: {len(impact_zones)}
- HIGH/CRITICAL risk files: {len(high_risk_files)}
- Files you should AVOID modifying: {', '.join(high_risk_files[:5])}...

Before ANY modification:
1. Check if file is in high risk list above
2. Run: get_impact_zone(filename) to see dependencies  
3. Review the safe_change_checklist for the file
4. Consider alternative approaches suggested

Remember: Breaking changes cascade through dependencies!
"""