#!/usr/bin/env python3
"""
Documentation Agent for CodebaseIQ Pro
Analyzes documentation quality and coverage
"""

import logging
import re
from typing import Dict, Any, List
from pathlib import Path
import aiofiles

from ..core import BaseAgent, AgentRole

logger = logging.getLogger(__name__)

class DocumentationAgent(BaseAgent):
    """Agent for analyzing documentation and comments"""
    
    def __init__(self):
        super().__init__(AgentRole.DOCUMENTATION)
        self.doc_patterns = {
            'python': {
                'docstring': re.compile(r'"""(.*?)"""', re.DOTALL),
                'comment': re.compile(r'#\s*(.+)$', re.MULTILINE),
                'type_hint': re.compile(r':\s*([A-Za-z_][A-Za-z0-9_\[\], ]*)\s*[=)]')
            },
            'javascript': {
                'jsdoc': re.compile(r'/\*\*(.*?)\*/', re.DOTALL),
                'comment': re.compile(r'//\s*(.+)$', re.MULTILINE)
            },
            'typescript': {
                'jsdoc': re.compile(r'/\*\*(.*?)\*/', re.DOTALL),
                'comment': re.compile(r'//\s*(.+)$', re.MULTILINE),
                'type_annotation': re.compile(r':\s*([A-Za-z_][A-Za-z0-9_<>, ]*)')
            },
            'java': {
                'javadoc': re.compile(r'/\*\*(.*?)\*/', re.DOTALL),
                'comment': re.compile(r'//\s*(.+)$', re.MULTILINE)
            }
        }
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze documentation quality and coverage"""
        file_map = context.get('file_map', {})
        entities = context.get('entities', {})
        
        doc_scores = {}
        total_score = 0
        analyzed_files = 0
        documentation_issues = []
        
        # Sample files for analysis (limit for performance)
        files_to_analyze = list(file_map.items())[:200]
        
        for rel_path, full_path in files_to_analyze:
            language = self._detect_language(full_path)
            if language not in self.doc_patterns:
                continue
                
            try:
                async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    
                # Calculate documentation score
                score, issues = await self._calculate_doc_score(content, rel_path, language)
                doc_scores[rel_path] = score
                total_score += score
                analyzed_files += 1
                
                # Update entity
                if rel_path in entities:
                    entities[rel_path].documentation_score = score
                    
                # Collect issues
                if issues:
                    documentation_issues.extend(issues)
                    
            except Exception as e:
                logger.warning(f"Failed to analyze docs for {rel_path}: {e}")
                
        avg_score = total_score / analyzed_files if analyzed_files > 0 else 0
        
        # Categorize files by documentation quality
        poorly_documented = []
        well_documented = []
        for path, score in doc_scores.items():
            if score < 0.3:
                poorly_documented.append(path)
            elif score > 0.8:
                well_documented.append(path)
                
        return {
            'average_doc_score': round(avg_score, 2),
            'analyzed_files': analyzed_files,
            'total_files': len(file_map),
            'poorly_documented': poorly_documented[:10],  # Top 10
            'well_documented': well_documented[:10],  # Top 10
            'documentation_issues': documentation_issues[:20],  # Top 20 issues
            'doc_coverage_percentage': round((analyzed_files / len(file_map) * 100), 2) if file_map else 0,
            'recommendations': self._generate_recommendations(avg_score, documentation_issues)
        }
        
    def _detect_language(self, path: Path) -> str:
        """Detect programming language from file extension"""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php'
        }
        return ext_to_lang.get(path.suffix.lower(), 'unknown')
        
    async def _calculate_doc_score(self, content: str, file_path: str, language: str) -> tuple:
        """Calculate documentation score for a file"""
        lines = content.split('\n')
        total_lines = len(lines)
        doc_lines = 0
        code_lines = 0
        issues = []
        
        patterns = self.doc_patterns.get(language, {})
        
        # Count documentation
        for pattern_name, pattern in patterns.items():
            matches = pattern.findall(content)
            for match in matches:
                doc_lines += len(match.split('\n'))
                
        # Count actual code lines (non-empty, non-comment)
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith(('#', '//', '/*', '*')):
                code_lines += 1
                
        # Calculate base score
        if code_lines > 0:
            doc_ratio = doc_lines / code_lines
            base_score = min(doc_ratio, 1.0)  # Cap at 1.0
        else:
            base_score = 0.5  # Default for files with no code
            
        # Boost score for specific documentation patterns
        boost = 0
        
        # Check for module-level docstring (Python)
        if language == 'python' and content.strip().startswith('"""'):
            boost += 0.1
            
        # Check for class and function documentation
        if language == 'python':
            # Count documented functions
            func_pattern = re.compile(r'def\s+\w+\s*\([^)]*\):\s*\n\s*"""')
            documented_funcs = len(func_pattern.findall(content))
            
            # Count total functions
            total_funcs = len(re.findall(r'def\s+\w+\s*\([^)]*\):', content))
            
            if total_funcs > 0:
                func_doc_ratio = documented_funcs / total_funcs
                boost += func_doc_ratio * 0.2
                
                if func_doc_ratio < 0.5:
                    issues.append({
                        'file': file_path,
                        'type': 'missing_function_docs',
                        'severity': 'medium',
                        'message': f'Only {documented_funcs}/{total_funcs} functions have docstrings'
                    })
                    
        # Check for TODO/FIXME comments
        todo_count = len(re.findall(r'(TODO|FIXME|XXX|HACK)', content, re.IGNORECASE))
        if todo_count > 5:
            issues.append({
                'file': file_path,
                'type': 'excessive_todos',
                'severity': 'low',
                'message': f'Found {todo_count} TODO/FIXME comments'
            })
            
        # Final score
        final_score = min(base_score + boost, 1.0)
        
        # Generate issues for poor documentation
        if final_score < 0.3:
            issues.append({
                'file': file_path,
                'type': 'poor_documentation',
                'severity': 'high',
                'message': 'File has very little documentation'
            })
            
        return final_score, issues
        
    def _generate_recommendations(self, avg_score: float, issues: List[Dict]) -> List[str]:
        """Generate documentation improvement recommendations"""
        recommendations = []
        
        if avg_score < 0.3:
            recommendations.append("Critical: Documentation is severely lacking. Consider adding docstrings to all public functions and classes.")
            
        if avg_score < 0.5:
            recommendations.append("Add module-level documentation to explain the purpose of each file.")
            recommendations.append("Document all public APIs with parameter descriptions and return values.")
            
        # Analyze issues
        issue_types = {}
        for issue in issues:
            issue_type = issue['type']
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
        if issue_types.get('missing_function_docs', 0) > 10:
            recommendations.append("Many functions lack documentation. Consider using a documentation generator tool.")
            
        if issue_types.get('excessive_todos', 0) > 5:
            recommendations.append("Convert TODO/FIXME comments into tracked issues for better visibility.")
            
        if avg_score > 0.8:
            recommendations.append("Good documentation coverage! Consider adding more examples in complex functions.")
            
        return recommendations[:5]  # Top 5 recommendations