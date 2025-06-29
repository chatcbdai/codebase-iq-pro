#!/usr/bin/env python3
"""
Specialized Analysis Agents for CodebaseIQ Pro
Implements all the missing agents for comprehensive codebase analysis
"""

import asyncio
import logging
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict
import networkx as nx
from datetime import datetime
import aiofiles
import json

from ..core import BaseAgent, AgentRole, EnhancedCodeEntity

logger = logging.getLogger(__name__)

class DependencyAnalysisAgent(BaseAgent):
    """Agent for analyzing code dependencies and imports"""
    
    def __init__(self):
        super().__init__(AgentRole.DEPENDENCY)
        self.import_patterns = {
            'python': [
                r'^\s*import\s+(\S+)',
                r'^\s*from\s+(\S+)\s+import',
            ],
            'javascript': [
                r'import\s+.*\s+from\s+[\'"](.+?)[\'"]',
                r'require\s*\([\'"](.+?)[\'"]\)',
                r'import\s*\([\'"](.+?)[\'"]\)',
            ],
            'typescript': [
                r'import\s+.*\s+from\s+[\'"](.+?)[\'"]',
                r'import\s+type\s+.*\s+from\s+[\'"](.+?)[\'"]',
                r'require\s*\([\'"](.+?)[\'"]\)',
            ],
            'java': [
                r'^\s*import\s+(\S+);',
                r'^\s*import\s+static\s+(\S+);',
            ],
            'go': [
                r'^\s*import\s+"(.+?)"',
                r'^\s*import\s+\(\s*"(.+?)"',
            ],
        }
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dependencies across the codebase"""
        file_map = context.get('file_map', {})
        entities = context.get('entities', {})
        
        # Initialize dependency graph
        dep_graph = nx.DiGraph()
        external_deps = defaultdict(set)
        internal_deps = defaultdict(set)
        
        # Analyze each file
        for rel_path, full_path in file_map.items():
            try:
                # Detect language
                language = self._detect_language(full_path)
                if language not in self.import_patterns:
                    continue
                    
                # Read file content
                async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    
                # Extract imports
                imports = self._extract_imports(content, language)
                
                # Classify dependencies
                for imp in imports:
                    if self._is_internal_import(imp, rel_path):
                        internal_deps[rel_path].add(imp)
                        # Add edge to dependency graph
                        dep_graph.add_edge(rel_path, imp)
                    else:
                        external_deps[rel_path].add(imp)
                        
                # Update entity
                if rel_path in entities:
                    entities[rel_path].dependencies = imports
                    
            except Exception as e:
                logger.warning(f"Failed to analyze dependencies for {rel_path}: {e}")
                
        # Analyze dependency graph
        analysis_results = {
            'total_files': len(file_map),
            'files_with_deps': len(external_deps) + len(internal_deps),
            'external_dependencies': self._aggregate_external_deps(external_deps),
            'internal_dependencies': len(internal_deps),
            'dependency_graph': {
                'nodes': dep_graph.number_of_nodes(),
                'edges': dep_graph.number_of_edges(),
                'cycles': list(nx.simple_cycles(dep_graph)) if dep_graph.number_of_nodes() < 100 else [],
                'most_depended_on': self._find_most_depended_on(dep_graph),
                'most_dependencies': self._find_most_dependencies(dep_graph),
            },
            'package_managers': self._detect_package_managers(file_map),
        }
        
        return analysis_results
        
    def _detect_language(self, path: Path) -> Optional[str]:
        """Detect programming language from file extension"""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.go': 'go',
        }
        return ext_to_lang.get(path.suffix.lower())
        
    def _extract_imports(self, content: str, language: str) -> Set[str]:
        """Extract imports from file content"""
        imports = set()
        patterns = self.import_patterns.get(language, [])
        
        for line in content.split('\n'):
            for pattern in patterns:
                matches = re.findall(pattern, line, re.MULTILINE)
                imports.update(matches)
                
        return imports
        
    def _is_internal_import(self, import_path: str, current_file: str) -> bool:
        """Check if import is internal to the project"""
        # Simple heuristic: relative imports or no package prefix
        return (import_path.startswith('.') or 
                import_path.startswith('/') or
                not ('/' in import_path or '.' in import_path))
                
    def _aggregate_external_deps(self, external_deps: Dict[str, Set[str]]) -> Dict[str, int]:
        """Aggregate and count external dependencies"""
        dep_counts = defaultdict(int)
        for deps in external_deps.values():
            for dep in deps:
                # Extract package name (first part before /)
                package = dep.split('/')[0].split('.')[0]
                dep_counts[package] += 1
                
        # Sort by frequency
        return dict(sorted(dep_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        
    def _find_most_depended_on(self, graph: nx.DiGraph) -> List[Tuple[str, int]]:
        """Find files that are most depended on"""
        in_degrees = [(node, graph.in_degree(node)) for node in graph.nodes()]
        return sorted(in_degrees, key=lambda x: x[1], reverse=True)[:10]
        
    def _find_most_dependencies(self, graph: nx.DiGraph) -> List[Tuple[str, int]]:
        """Find files with most dependencies"""
        out_degrees = [(node, graph.out_degree(node)) for node in graph.nodes()]
        return sorted(out_degrees, key=lambda x: x[1], reverse=True)[:10]
        
    def _detect_package_managers(self, file_map: Dict[str, Path]) -> List[str]:
        """Detect package managers used in the project"""
        managers = []
        
        manager_files = {
            'npm': ['package.json', 'package-lock.json'],
            'yarn': ['yarn.lock'],
            'pnpm': ['pnpm-lock.yaml'],
            'pip': ['requirements.txt', 'Pipfile', 'pyproject.toml'],
            'poetry': ['poetry.lock'],
            'maven': ['pom.xml'],
            'gradle': ['build.gradle', 'build.gradle.kts'],
            'cargo': ['Cargo.toml', 'Cargo.lock'],
            'go mod': ['go.mod', 'go.sum'],
        }
        
        file_names = {Path(p).name for p in file_map.keys()}
        
        for manager, files in manager_files.items():
            if any(f in file_names for f in files):
                managers.append(manager)
                
        return managers

class SecurityAuditAgent(BaseAgent):
    """Agent for security analysis and vulnerability detection"""
    
    def __init__(self):
        super().__init__(AgentRole.SECURITY)
        self.security_patterns = {
            'hardcoded_secrets': [
                r'(?i)(api_key|apikey|api-key)\s*=\s*["\'][\w\-]+["\']',
                r'(?i)(secret|password|passwd|pwd)\s*=\s*["\'][\w\-]+["\']',
                r'(?i)(token|auth|bearer)\s*=\s*["\'][\w\-]+["\']',
                r'(?i)aws_access_key_id\s*=\s*["\'][\w\-]+["\']',
                r'(?i)private_key\s*=\s*["\'][\w\-]+["\']',
            ],
            'sql_injection': [
                r'(?i)(query|execute)\s*\([^)]*\+[^)]*\)',
                r'(?i)(query|execute)\s*\([^)]*%[^)]*\)',
                r'(?i)(query|execute)\s*\([^)]*\.format\([^)]*\)',
                r'f["\'].*SELECT.*{.*}.*["\']',
            ],
            'command_injection': [
                r'(?i)os\.system\s*\([^)]*\+[^)]*\)',
                r'(?i)subprocess\.\w+\s*\([^)]*shell\s*=\s*True',
                r'(?i)eval\s*\([^)]*\)',
                r'(?i)exec\s*\([^)]*\)',
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\\\',
                r'(?i)os\.path\.join\s*\([^)]*request\.',
            ],
            'weak_crypto': [
                r'(?i)md5\s*\(',
                r'(?i)sha1\s*\(',
                r'(?i)DES\s*\(',
                r'(?i)random\.\w+\s*\(',  # Using random for security
            ],
        }
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform security audit on the codebase"""
        file_map = context.get('file_map', {})
        entities = context.get('entities', {})
        
        vulnerabilities = []
        security_score = 100  # Start with perfect score
        danger_zones = []
        
        # Scan each file
        for rel_path, full_path in file_map.items():
            try:
                async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    
                # Check for various vulnerability patterns
                file_vulns = await self._scan_file_vulnerabilities(rel_path, content)
                
                if file_vulns:
                    vulnerabilities.extend(file_vulns)
                    
                    # Calculate danger level
                    danger_level = self._calculate_danger_level(file_vulns)
                    
                    # Update entity
                    if rel_path in entities:
                        entities[rel_path].danger_level = danger_level
                        
                    if danger_level >= 7:
                        danger_zones.append({
                            'file': rel_path,
                            'danger_level': danger_level,
                            'vulnerabilities': len(file_vulns),
                            'types': list(set(v['type'] for v in file_vulns))
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to scan {rel_path}: {e}")
                
        # Calculate overall security score
        if vulnerabilities:
            # Deduct points based on severity
            for vuln in vulnerabilities:
                if vuln['severity'] == 'critical':
                    security_score -= 10
                elif vuln['severity'] == 'high':
                    security_score -= 5
                elif vuln['severity'] == 'medium':
                    security_score -= 2
                else:
                    security_score -= 1
                    
        security_score = max(0, security_score)
        
        return {
            'security_score': security_score / 10,  # Convert to 0-10 scale
            'total_vulnerabilities': len(vulnerabilities),
            'vulnerability_breakdown': self._aggregate_vulnerabilities(vulnerabilities),
            'danger_zones': sorted(danger_zones, key=lambda x: x['danger_level'], reverse=True),
            'secure_coding_practices': self._check_secure_practices(file_map),
            'compliance_status': self._check_compliance(context),
        }
        
    async def _scan_file_vulnerabilities(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Scan a file for security vulnerabilities"""
        vulnerabilities = []
        lines = content.split('\n')
        
        for vuln_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        vulnerabilities.append({
                            'file': file_path,
                            'line': line_num,
                            'type': vuln_type,
                            'pattern': pattern,
                            'severity': self._get_severity(vuln_type),
                            'description': self._get_description(vuln_type),
                            'recommendation': self._get_recommendation(vuln_type)
                        })
                        
        return vulnerabilities
        
    def _calculate_danger_level(self, vulnerabilities: List[Dict]) -> int:
        """Calculate danger level (0-10) based on vulnerabilities"""
        if not vulnerabilities:
            return 0
            
        max_severity = 0
        for vuln in vulnerabilities:
            if vuln['severity'] == 'critical':
                max_severity = max(max_severity, 10)
            elif vuln['severity'] == 'high':
                max_severity = max(max_severity, 8)
            elif vuln['severity'] == 'medium':
                max_severity = max(max_severity, 5)
            else:
                max_severity = max(max_severity, 3)
                
        return max_severity
        
    def _get_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type"""
        severity_map = {
            'hardcoded_secrets': 'critical',
            'sql_injection': 'critical',
            'command_injection': 'critical',
            'path_traversal': 'high',
            'weak_crypto': 'medium',
        }
        return severity_map.get(vuln_type, 'low')
        
    def _get_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type"""
        descriptions = {
            'hardcoded_secrets': 'Hardcoded credentials or API keys detected',
            'sql_injection': 'Potential SQL injection vulnerability',
            'command_injection': 'Potential command injection vulnerability',
            'path_traversal': 'Potential path traversal vulnerability',
            'weak_crypto': 'Weak cryptographic algorithm usage',
        }
        return descriptions.get(vuln_type, 'Security vulnerability detected')
        
    def _get_recommendation(self, vuln_type: str) -> str:
        """Get recommendation for fixing vulnerability"""
        recommendations = {
            'hardcoded_secrets': 'Use environment variables or secure credential storage',
            'sql_injection': 'Use parameterized queries or prepared statements',
            'command_injection': 'Avoid shell=True, sanitize inputs, use subprocess with list arguments',
            'path_traversal': 'Validate and sanitize file paths, use safe path joining',
            'weak_crypto': 'Use strong cryptographic algorithms (SHA-256, AES, etc.)',
        }
        return recommendations.get(vuln_type, 'Review and fix the security issue')
        
    def _aggregate_vulnerabilities(self, vulnerabilities: List[Dict]) -> Dict[str, Any]:
        """Aggregate vulnerabilities by type and severity"""
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        
        for vuln in vulnerabilities:
            by_type[vuln['type']] += 1
            by_severity[vuln['severity']] += 1
            
        return {
            'by_type': dict(by_type),
            'by_severity': dict(by_severity)
        }
        
    def _check_secure_practices(self, file_map: Dict[str, Path]) -> Dict[str, bool]:
        """Check for secure coding practices"""
        practices = {
            'has_security_headers': False,
            'uses_https': False,
            'has_csp': False,
            'input_validation': False,
            'uses_secure_random': False,
        }
        
        # Simple checks based on file existence and patterns
        file_names = {Path(p).name for p in file_map.keys()}
        
        # Check for security-related files
        if any(f in file_names for f in ['.env.example', 'security.md', 'SECURITY.md']):
            practices['has_security_policy'] = True
            
        return practices
        
    def _check_compliance(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Check compliance with security standards"""
        return {
            'owasp_top_10': 'partial',
            'pci_dss': 'unknown',
            'gdpr': 'unknown',
            'soc2': 'unknown',
        }

class PatternDetectionAgent(BaseAgent):
    """Agent for detecting code patterns and anti-patterns"""
    
    def __init__(self):
        super().__init__(AgentRole.PATTERN)
        self.patterns = {
            'design_patterns': {
                'singleton': r'class\s+\w+.*:\s*\n\s*_instance\s*=\s*None',
                'factory': r'class\s+\w*Factory',
                'observer': r'(subscribe|notify|observe)',
                'decorator': r'@\w+|function.*decorator',
                'strategy': r'class\s+\w*Strategy',
            },
            'anti_patterns': {
                'god_class': self._check_god_class,
                'long_method': self._check_long_method,
                'duplicate_code': self._check_duplicate_code,
                'dead_code': r'(TODO|FIXME|HACK|XXX)',
                'magic_numbers': r'\b\d{2,}\b(?!\s*[:\]\)])',
            },
            'code_smells': {
                'nested_loops': self._check_nested_loops,
                'deep_nesting': self._check_deep_nesting,
                'long_parameter_list': r'def\s+\w+\s*\([^)]{100,}\)',
                'large_class': self._check_large_class,
            }
        }
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns and anti-patterns in the codebase"""
        file_map = context.get('file_map', {})
        entities = context.get('entities', {})
        
        detected_patterns = defaultdict(list)
        anti_patterns = defaultdict(list)
        code_quality_score = 100
        
        # Analyze each file
        for rel_path, full_path in file_map.items():
            try:
                async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    
                # Check design patterns
                for pattern_name, pattern in self.patterns['design_patterns'].items():
                    if isinstance(pattern, str) and re.search(pattern, content):
                        detected_patterns[pattern_name].append(rel_path)
                        
                # Check anti-patterns
                for anti_pattern_name, checker in self.patterns['anti_patterns'].items():
                    result = None
                    if callable(checker):
                        result = checker(content, rel_path)
                    elif isinstance(checker, str) and re.search(checker, content):
                        result = True
                        
                    if result:
                        anti_patterns[anti_pattern_name].append(rel_path)
                        code_quality_score -= 5
                        
            except Exception as e:
                logger.warning(f"Failed to analyze patterns in {rel_path}: {e}")
                
        code_quality_score = max(0, code_quality_score)
        
        return {
            'design_patterns_found': dict(detected_patterns),
            'anti_patterns_found': dict(anti_patterns),
            'code_quality_score': code_quality_score / 10,
            'refactoring_suggestions': self._generate_refactoring_suggestions(anti_patterns),
            'best_practices_score': self._calculate_best_practices_score(context),
        }
        
    def _check_god_class(self, content: str, file_path: str) -> bool:
        """Check if a class is too large (god class)"""
        # Count methods in a class
        method_count = len(re.findall(r'def\s+\w+\s*\(', content))
        return method_count > 20
        
    def _check_long_method(self, content: str, file_path: str) -> bool:
        """Check for methods that are too long"""
        # Simple heuristic: methods over 50 lines
        methods = re.split(r'def\s+\w+\s*\([^)]*\):', content)
        for method in methods[1:]:  # Skip first split (before any method)
            lines = method.split('\n')
            # Count non-empty, non-comment lines
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            if len(code_lines) > 50:
                return True
        return False
        
    def _check_duplicate_code(self, content: str, file_path: str) -> bool:
        """Check for duplicate code blocks"""
        # Simple check: look for repeated blocks of 5+ lines
        lines = content.split('\n')
        for i in range(len(lines) - 10):
            block = '\n'.join(lines[i:i+5])
            if block.strip() and content.count(block) > 1:
                return True
        return False
        
    def _check_nested_loops(self, content: str, file_path: str) -> bool:
        """Check for deeply nested loops"""
        # Look for multiple 'for' or 'while' keywords with increasing indentation
        return bool(re.search(r'for.*:\s*\n\s+.*for.*:|while.*:\s*\n\s+.*while.*:', content))
        
    def _check_deep_nesting(self, content: str, file_path: str) -> bool:
        """Check for deep nesting levels"""
        max_indent = 0
        for line in content.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        return max_indent > 20  # More than 5 levels of indentation (4 spaces each)
        
    def _check_large_class(self, content: str, file_path: str) -> bool:
        """Check if file contains overly large classes"""
        lines = content.split('\n')
        return len(lines) > 500
        
    def _generate_refactoring_suggestions(self, anti_patterns: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Generate refactoring suggestions based on detected anti-patterns"""
        suggestions = []
        
        if 'god_class' in anti_patterns:
            for file in anti_patterns['god_class']:
                suggestions.append({
                    'file': file,
                    'issue': 'God Class',
                    'suggestion': 'Consider breaking this class into smaller, focused classes'
                })
                
        if 'long_method' in anti_patterns:
            for file in anti_patterns['long_method']:
                suggestions.append({
                    'file': file,
                    'issue': 'Long Method',
                    'suggestion': 'Extract smaller methods from long methods'
                })
                
        return suggestions
        
    def _calculate_best_practices_score(self, context: Dict[str, Any]) -> float:
        """Calculate overall best practices score"""
        # Simple scoring based on various factors
        score = 10.0
        
        # Deduct for anti-patterns
        # Additional scoring logic here
        
        return max(0, score)

class VersionCompatibilityAgent(BaseAgent):
    """Agent for checking version compatibility and requirements"""
    
    def __init__(self):
        super().__init__(AgentRole.VERSION)
        self.version_extractors = {
            'python': self._extract_python_versions,
            'javascript': self._extract_js_versions,
            'java': self._extract_java_versions,
        }
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze version requirements and compatibility"""
        file_map = context.get('file_map', {})
        
        # Detect package files
        package_files = self._find_package_files(file_map)
        
        # Extract version requirements
        requirements = {}
        compatibility_issues = []
        
        for file_type, file_path in package_files.items():
            if file_path:
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        
                    reqs = self._extract_requirements(content, file_type)
                    requirements[file_type] = reqs
                    
                    # Check for compatibility issues
                    issues = self._check_compatibility(reqs)
                    compatibility_issues.extend(issues)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
                    
        return {
            'package_managers': list(package_files.keys()),
            'requirements': requirements,
            'compatibility_issues': compatibility_issues,
            'outdated_packages': self._check_outdated_packages(requirements),
            'security_vulnerabilities': self._check_known_vulnerabilities(requirements),
        }
        
    def _find_package_files(self, file_map: Dict[str, Path]) -> Dict[str, Optional[Path]]:
        """Find package management files"""
        package_files = {
            'npm': None,
            'pip': None,
            'maven': None,
            'gradle': None,
        }
        
        for rel_path, full_path in file_map.items():
            filename = Path(rel_path).name
            
            if filename == 'package.json':
                package_files['npm'] = full_path
            elif filename in ['requirements.txt', 'Pipfile', 'pyproject.toml']:
                package_files['pip'] = full_path
            elif filename == 'pom.xml':
                package_files['maven'] = full_path
            elif filename in ['build.gradle', 'build.gradle.kts']:
                package_files['gradle'] = full_path
                
        return package_files
        
    def _extract_requirements(self, content: str, file_type: str) -> Dict[str, str]:
        """Extract package requirements from content"""
        requirements = {}
        
        if file_type == 'npm':
            try:
                data = json.loads(content)
                requirements.update(data.get('dependencies', {}))
                requirements.update(data.get('devDependencies', {}))
            except:
                pass
                
        elif file_type == 'pip':
            # Parse requirements.txt format
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    match = re.match(r'([a-zA-Z0-9\-_]+)([><=~!]+)(.+)', line)
                    if match:
                        requirements[match.group(1)] = match.group(2) + match.group(3)
                        
        return requirements
        
    def _extract_python_versions(self, content: str) -> Optional[str]:
        """Extract Python version requirements"""
        # Look for python_requires in setup.py or pyproject.toml
        match = re.search(r'python_requires\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
        return None
        
    def _extract_js_versions(self, content: str) -> Optional[str]:
        """Extract Node.js version requirements"""
        try:
            data = json.loads(content)
            return data.get('engines', {}).get('node')
        except:
            return None
            
    def _extract_java_versions(self, content: str) -> Optional[str]:
        """Extract Java version requirements"""
        # Look for source/target compatibility
        match = re.search(r'<source>(\d+)</source>', content)
        if match:
            return f"Java {match.group(1)}"
        return None
        
    def _check_compatibility(self, requirements: Dict[str, str]) -> List[Dict[str, str]]:
        """Check for known compatibility issues"""
        issues = []
        
        # Example: Check for conflicting versions
        # This would be more sophisticated in production
        
        return issues
        
    def _check_outdated_packages(self, requirements: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
        """Check for outdated packages"""
        # In production, this would check against package registries
        outdated = []
        
        # Example check
        for pkg_type, reqs in requirements.items():
            for package, version in reqs.items():
                # Placeholder logic
                if 'old' in package.lower():
                    outdated.append({
                        'package': package,
                        'current': version,
                        'latest': 'unknown',
                        'type': pkg_type
                    })
                    
        return outdated
        
    def _check_known_vulnerabilities(self, requirements: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
        """Check for known security vulnerabilities"""
        # In production, this would check vulnerability databases
        vulnerabilities = []
        
        # Known vulnerable packages (example)
        vulnerable_packages = {
            'lodash': ['< 4.17.21', 'CVE-2021-23337'],
            'axios': ['< 0.21.1', 'CVE-2020-28168'],
        }
        
        for pkg_type, reqs in requirements.items():
            for package, version in reqs.items():
                if package in vulnerable_packages:
                    vulnerabilities.append({
                        'package': package,
                        'version': version,
                        'vulnerability': vulnerable_packages[package][1],
                        'severity': 'high'
                    })
                    
        return vulnerabilities

class ArchitectureAnalysisAgent(BaseAgent):
    """Agent for analyzing software architecture"""
    
    def __init__(self):
        super().__init__(AgentRole.ARCHITECTURE)
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the software architecture"""
        file_map = context.get('file_map', {})
        entities = context.get('entities', {})
        
        # Analyze directory structure
        dir_structure = self._analyze_directory_structure(file_map)
        
        # Detect architecture patterns
        patterns = self._detect_architecture_patterns(dir_structure, file_map)
        
        # Analyze layers
        layers = self._analyze_layers(file_map)
        
        # Component analysis
        components = self._analyze_components(entities)
        
        return {
            'architecture_style': patterns['style'],
            'patterns_detected': patterns['patterns'],
            'directory_structure': dir_structure,
            'layers': layers,
            'components': components,
            'modularity_score': self._calculate_modularity_score(components),
            'coupling_analysis': self._analyze_coupling(entities),
        }
        
    def _analyze_directory_structure(self, file_map: Dict[str, Path]) -> Dict[str, Any]:
        """Analyze the directory structure"""
        dir_counts = defaultdict(int)
        max_depth = 0
        
        for rel_path in file_map.keys():
            parts = Path(rel_path).parts
            max_depth = max(max_depth, len(parts))
            
            # Count files per directory
            if len(parts) > 1:
                dir_counts[parts[0]] += 1
                
        return {
            'max_depth': max_depth,
            'top_level_dirs': dict(sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)),
            'total_directories': len(dir_counts),
        }
        
    def _detect_architecture_patterns(self, dir_structure: Dict, file_map: Dict[str, Path]) -> Dict[str, Any]:
        """Detect common architecture patterns"""
        patterns = []
        style = 'monolithic'  # default
        
        top_dirs = set(dir_structure['top_level_dirs'].keys())
        
        # MVC pattern
        if {'models', 'views', 'controllers'} & top_dirs:
            patterns.append('MVC')
            style = 'mvc'
            
        # Microservices pattern
        if 'services' in top_dirs or 'microservices' in top_dirs:
            patterns.append('Microservices')
            style = 'microservices'
            
        # Layered architecture
        if {'presentation', 'business', 'data'} & top_dirs or {'ui', 'api', 'db'} & top_dirs:
            patterns.append('Layered')
            style = 'layered'
            
        # Clean architecture
        if {'domain', 'application', 'infrastructure'} & top_dirs:
            patterns.append('Clean Architecture')
            style = 'clean'
            
        return {
            'style': style,
            'patterns': patterns
        }
        
    def _analyze_layers(self, file_map: Dict[str, Path]) -> Dict[str, List[str]]:
        """Analyze architectural layers"""
        layers = {
            'presentation': [],
            'business': [],
            'data': [],
            'infrastructure': [],
        }
        
        for rel_path in file_map.keys():
            lower_path = rel_path.lower()
            
            if any(x in lower_path for x in ['ui', 'view', 'component', 'page', 'template']):
                layers['presentation'].append(rel_path)
            elif any(x in lower_path for x in ['service', 'business', 'logic', 'domain']):
                layers['business'].append(rel_path)
            elif any(x in lower_path for x in ['model', 'entity', 'schema', 'db', 'repository']):
                layers['data'].append(rel_path)
            elif any(x in lower_path for x in ['util', 'helper', 'config', 'middleware']):
                layers['infrastructure'].append(rel_path)
                
        return {k: len(v) for k, v in layers.items() if v}
        
    def _analyze_components(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze software components"""
        components = defaultdict(list)
        
        for path, entity in entities.items():
            # Group by directory
            parts = Path(path).parts
            if len(parts) > 1:
                component = parts[0]
                components[component].append(path)
                
        return {
            'count': len(components),
            'sizes': {k: len(v) for k, v in components.items()},
            'largest': max(components.items(), key=lambda x: len(x[1]))[0] if components else None,
        }
        
    def _calculate_modularity_score(self, components: Dict[str, Any]) -> float:
        """Calculate modularity score"""
        if not components['count']:
            return 0.0
            
        # Simple scoring based on component count and size distribution
        score = min(10, components['count'] / 2)  # More components = better modularity
        
        # Penalize uneven distribution
        sizes = list(components['sizes'].values())
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
            if variance > avg_size:
                score -= 2
                
        return max(0, min(10, score))
        
    def _analyze_coupling(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coupling between components"""
        # Simple coupling analysis based on dependencies
        coupling_score = 10.0
        
        # Count cross-component dependencies
        cross_deps = 0
        for entity in entities.values():
            if hasattr(entity, 'dependencies'):
                # Check if dependencies cross component boundaries
                # Simplified logic
                cross_deps += len(entity.dependencies) * 0.1
                
        coupling_score -= min(5, cross_deps)
        
        return {
            'coupling_score': max(0, coupling_score),
            'recommendation': 'Low coupling detected' if coupling_score > 7 else 'Consider reducing dependencies'
        }

class PerformanceAnalysisAgent(BaseAgent):
    """Agent for analyzing performance characteristics"""
    
    def __init__(self):
        super().__init__(AgentRole.PERFORMANCE)
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance characteristics of the codebase"""
        file_map = context.get('file_map', {})
        
        performance_issues = []
        optimization_opportunities = []
        
        # Analyze each file for performance patterns
        for rel_path, full_path in file_map.items():
            try:
                async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    
                # Check for performance anti-patterns
                issues = self._check_performance_issues(content, rel_path)
                performance_issues.extend(issues)
                
                # Find optimization opportunities
                opts = self._find_optimization_opportunities(content, rel_path)
                optimization_opportunities.extend(opts)
                
            except Exception as e:
                logger.warning(f"Failed to analyze performance for {rel_path}: {e}")
                
        return {
            'performance_score': self._calculate_performance_score(performance_issues),
            'issues_found': len(performance_issues),
            'top_issues': performance_issues[:10],
            'optimization_opportunities': optimization_opportunities[:10],
            'complexity_analysis': self._analyze_complexity(file_map),
            'resource_usage_patterns': self._analyze_resource_usage(file_map),
        }
        
    def _check_performance_issues(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Check for performance anti-patterns"""
        issues = []
        
        # N+1 query pattern
        if 'for' in content and any(x in content for x in ['query', 'find', 'select']):
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'for' in line:
                    # Check next few lines for queries
                    for j in range(i+1, min(i+10, len(lines))):
                        if any(x in lines[j] for x in ['query', 'find', 'select']):
                            issues.append({
                                'type': 'n_plus_one_query',
                                'file': file_path,
                                'line': j+1,
                                'severity': 'high',
                                'description': 'Potential N+1 query problem'
                            })
                            break
                            
        # Inefficient string concatenation
        if '+=' in content and ('"' in content or "'" in content):
            issues.append({
                'type': 'string_concatenation',
                'file': file_path,
                'severity': 'medium',
                'description': 'Inefficient string concatenation detected'
            })
            
        # Large list comprehensions
        comprehensions = re.findall(r'\[.{100,}?\]', content)
        if comprehensions:
            issues.append({
                'type': 'large_list_comprehension',
                'file': file_path,
                'severity': 'medium',
                'description': 'Large list comprehension may impact memory'
            })
            
        return issues
        
    def _find_optimization_opportunities(self, content: str, file_path: str) -> List[Dict[str, str]]:
        """Find potential optimization opportunities"""
        opportunities = []
        
        # Check for repeated computations
        function_calls = re.findall(r'(\w+)\s*\([^)]*\)', content)
        call_counts = defaultdict(int)
        for call in function_calls:
            call_counts[call] += 1
            
        for call, count in call_counts.items():
            if count > 5:
                opportunities.append({
                    'type': 'repeated_computation',
                    'file': file_path,
                    'description': f'Function {call} called {count} times - consider caching',
                    'impact': 'medium'
                })
                
        return opportunities
        
    def _calculate_performance_score(self, issues: List[Dict]) -> float:
        """Calculate overall performance score"""
        score = 10.0
        
        for issue in issues:
            if issue['severity'] == 'high':
                score -= 0.5
            elif issue['severity'] == 'medium':
                score -= 0.2
            else:
                score -= 0.1
                
        return max(0, score)
        
    def _analyze_complexity(self, file_map: Dict[str, Path]) -> Dict[str, Any]:
        """Analyze code complexity"""
        total_complexity = 0
        file_complexities = []
        
        # Simplified complexity calculation
        for rel_path, full_path in list(file_map.items())[:50]:  # Sample files
            complexity = 0
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count decision points
                complexity += len(re.findall(r'\b(if|for|while|case)\b', content))
                complexity += len(re.findall(r'\b(and|or)\b', content))
                
                file_complexities.append({
                    'file': rel_path,
                    'complexity': complexity
                })
                total_complexity += complexity
                
            except:
                pass
                
        avg_complexity = total_complexity / len(file_complexities) if file_complexities else 0
        
        return {
            'average_complexity': round(avg_complexity, 2),
            'most_complex_files': sorted(file_complexities, key=lambda x: x['complexity'], reverse=True)[:5]
        }
        
    def _analyze_resource_usage(self, file_map: Dict[str, Path]) -> Dict[str, List[str]]:
        """Analyze resource usage patterns"""
        patterns = {
            'file_operations': [],
            'network_calls': [],
            'database_operations': [],
            'memory_intensive': [],
        }
        
        # Sample analysis
        for rel_path in list(file_map.keys())[:50]:
            if any(x in rel_path for x in ['file', 'fs', 'io']):
                patterns['file_operations'].append(rel_path)
            if any(x in rel_path for x in ['http', 'request', 'api']):
                patterns['network_calls'].append(rel_path)
            if any(x in rel_path for x in ['db', 'query', 'model']):
                patterns['database_operations'].append(rel_path)
                
        return {k: len(v) for k, v in patterns.items()}