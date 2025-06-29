#!/usr/bin/env python3
"""
Test Coverage Agent for CodebaseIQ Pro
Analyzes test coverage and maps tests to source files
"""

import logging
import re
from typing import Dict, Any, List, Set
from pathlib import Path
from collections import defaultdict

from ..core import BaseAgent, AgentRole

logger = logging.getLogger(__name__)

class TestCoverageAgent(BaseAgent):
    """Agent for analyzing test coverage and test quality"""
    
    def __init__(self):
        super().__init__(AgentRole.TEST_COVERAGE)
        self.test_patterns = [
            'test_', '_test', 'spec.', '.spec', 
            'test/', 'tests/', '__tests__/', 'spec/'
        ]
        self.test_frameworks = {
            'python': ['pytest', 'unittest', 'nose'],
            'javascript': ['jest', 'mocha', 'jasmine', 'vitest'],
            'typescript': ['jest', 'mocha', 'jasmine', 'vitest'],
            'java': ['junit', 'testng'],
            'go': ['testing'],
            'rust': ['cargo test']
        }
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test coverage and map tests to source files"""
        file_map = context.get('file_map', {})
        entities = context.get('entities', {})
        
        # Separate test files from source files
        test_files = {}
        source_files = {}
        test_types = defaultdict(int)
        
        for path, full_path in file_map.items():
            if self._is_test_file(path):
                test_files[path] = full_path
                test_type = self._detect_test_type(path)
                test_types[test_type] += 1
            else:
                source_files[path] = full_path
                
        # Map tests to source files
        test_mapping = await self._map_tests_to_sources(test_files, source_files)
        
        # Calculate coverage metrics
        covered_files = set(test_mapping.keys())
        uncovered_files = set(source_files.keys()) - covered_files
        coverage_percentage = len(covered_files) / len(source_files) * 100 if source_files else 0
        
        # Analyze test quality
        test_quality = await self._analyze_test_quality(test_files, source_files)
        
        # Update entities with coverage info
        for path, entity in entities.items():
            if path in covered_files:
                entity.test_coverage = 1.0  # Has tests
            else:
                entity.test_coverage = 0.0  # No tests
                
        # Find critical uncovered files
        critical_uncovered = self._find_critical_uncovered(uncovered_files, entities)
        
        return {
            'test_file_count': len(test_files),
            'source_file_count': len(source_files),
            'covered_files': len(covered_files),
            'uncovered_files': len(uncovered_files),
            'coverage_percentage': round(coverage_percentage, 2),
            'test_mapping': dict(list(test_mapping.items())[:20]),  # Sample for display
            'critical_uncovered_files': critical_uncovered[:10],  # Top 10
            'test_types': dict(test_types),
            'test_quality': test_quality,
            'recommendations': self._generate_recommendations(coverage_percentage, critical_uncovered)
        }
        
    def _is_test_file(self, path: str) -> bool:
        """Check if a file is a test file"""
        path_lower = path.lower()
        return any(pattern in path_lower for pattern in self.test_patterns)
        
    def _detect_test_type(self, path: str) -> str:
        """Detect the type of test from the file path"""
        path_lower = path.lower()
        
        if 'unit' in path_lower:
            return 'unit'
        elif 'integration' in path_lower or 'e2e' in path_lower:
            return 'integration'
        elif 'spec' in path_lower:
            return 'spec'
        elif 'test' in path_lower:
            return 'test'
        else:
            return 'unknown'
            
    async def _map_tests_to_sources(self, test_files: Dict, source_files: Dict) -> Dict[str, List[str]]:
        """Map test files to their corresponding source files"""
        mapping = defaultdict(list)
        
        for test_path in test_files:
            # Extract base name from test file
            base_name = self._extract_base_name(test_path)
            
            # Find matching source files
            matches = []
            for source_path in source_files:
                if self._files_match(base_name, source_path, test_path):
                    matches.append(test_path)
                    mapping[source_path].append(test_path)
                    
            # If no direct match, try directory-based matching
            if not matches:
                test_dir = Path(test_path).parent
                for source_path in source_files:
                    source_dir = Path(source_path).parent
                    if self._directories_match(test_dir, source_dir):
                        mapping[source_path].append(test_path)
                        
        return dict(mapping)
        
    def _extract_base_name(self, test_path: str) -> str:
        """Extract base name from test file path"""
        base_name = Path(test_path).stem
        
        # Remove common test prefixes/suffixes
        for pattern in ['test_', '_test', '.spec', '.test']:
            base_name = base_name.replace(pattern, '')
            
        return base_name
        
    def _files_match(self, base_name: str, source_path: str, test_path: str) -> bool:
        """Check if a test file matches a source file"""
        source_name = Path(source_path).stem
        
        # Direct name match
        if base_name == source_name:
            return True
            
        # Check if base name is in source path
        if base_name in source_path:
            return True
            
        # Check if they're in related directories
        test_parts = Path(test_path).parts
        source_parts = Path(source_path).parts
        
        # Remove test directory from path
        test_parts_filtered = [p for p in test_parts if p not in ['test', 'tests', '__tests__', 'spec']]
        
        # Check for similarity
        common_parts = set(test_parts_filtered) & set(source_parts)
        if len(common_parts) >= 2:  # At least 2 common path components
            return True
            
        return False
        
    def _directories_match(self, test_dir: Path, source_dir: Path) -> bool:
        """Check if test and source directories are related"""
        test_parts = test_dir.parts
        source_parts = source_dir.parts
        
        # Remove test-specific parts
        test_parts_filtered = [p for p in test_parts if p not in ['test', 'tests', '__tests__', 'spec']]
        
        # Check if the paths are similar
        return test_parts_filtered == list(source_parts)
        
    async def _analyze_test_quality(self, test_files: Dict, source_files: Dict) -> Dict[str, Any]:
        """Analyze the quality of tests"""
        quality_metrics = {
            'test_to_source_ratio': len(test_files) / len(source_files) if source_files else 0,
            'average_tests_per_source': 0,
            'test_naming_score': 0,
            'test_organization_score': 0
        }
        
        # Calculate average tests per source file
        test_counts = defaultdict(int)
        for test_path in test_files:
            base_name = self._extract_base_name(test_path)
            test_counts[base_name] += 1
            
        if test_counts:
            quality_metrics['average_tests_per_source'] = sum(test_counts.values()) / len(test_counts)
            
        # Test naming score (check for descriptive names)
        good_names = 0
        for test_path in test_files:
            filename = Path(test_path).name
            # Good test names are descriptive
            if len(filename) > 10 and ('_' in filename or '-' in filename):
                good_names += 1
                
        quality_metrics['test_naming_score'] = good_names / len(test_files) if test_files else 0
        
        # Test organization score (tests in proper directories)
        organized_tests = sum(1 for path in test_files if any(
            test_dir in path for test_dir in ['test/', 'tests/', '__tests__/', 'spec/']
        ))
        quality_metrics['test_organization_score'] = organized_tests / len(test_files) if test_files else 0
        
        return quality_metrics
        
    def _find_critical_uncovered(self, uncovered_files: Set[str], entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find critical files that lack test coverage"""
        critical = []
        
        for file_path in uncovered_files:
            entity = entities.get(file_path)
            if not entity:
                continue
                
            # Calculate criticality score
            criticality = 0
            
            # High importance files are critical
            if hasattr(entity, 'importance_score') and entity.importance_score > 0.7:
                criticality += 3
                
            # Security-sensitive files are critical
            if hasattr(entity, 'danger_level') and entity.danger_level > 5:
                criticality += 5
                
            # Core business logic is critical
            if any(keyword in file_path.lower() for keyword in ['service', 'controller', 'model', 'auth', 'payment']):
                criticality += 2
                
            # Large files are more critical
            if hasattr(entity, 'size') and entity.size > 500:  # Lines of code
                criticality += 1
                
            if criticality > 0:
                critical.append({
                    'path': file_path,
                    'type': entity.type,
                    'criticality_score': criticality,
                    'danger_level': getattr(entity, 'danger_level', 0),
                    'importance': getattr(entity, 'importance_score', 0)
                })
                
        # Sort by criticality
        critical.sort(key=lambda x: x['criticality_score'], reverse=True)
        
        return critical
        
    def _generate_recommendations(self, coverage_percentage: float, critical_uncovered: List[Dict]) -> List[str]:
        """Generate test coverage recommendations"""
        recommendations = []
        
        if coverage_percentage < 20:
            recommendations.append("Critical: Test coverage is very low. Start by adding tests for core business logic.")
            
        if coverage_percentage < 50:
            recommendations.append("Add unit tests for all public methods and functions.")
            recommendations.append("Focus on testing error handling and edge cases.")
            
        if critical_uncovered:
            high_danger_files = [f['path'] for f in critical_uncovered if f['danger_level'] > 7]
            if high_danger_files:
                recommendations.append(f"Priority: Add tests for high-risk files: {', '.join(high_danger_files[:3])}")
                
        if coverage_percentage > 70:
            recommendations.append("Good coverage! Consider adding integration tests for complex workflows.")
            
        recommendations.append("Use code coverage tools to identify untested code paths.")
        
        return recommendations[:5]  # Top 5 recommendations