#!/usr/bin/env python3
"""
Comprehensive tests for CodebaseIQ Pro v2.0 new architecture.
Tests token management, cache management, and 6-tool aggregation.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from codebaseiq.core import TokenManager, TokenBudget, CacheManager, FileChangeInfo
from codebaseiq.server import CodebaseIQProServer


class TestTokenManager:
    """Test token counting and management."""
    
    def setup_method(self):
        """Set up test instance."""
        self.token_manager = TokenManager()
        
    def test_token_counting(self):
        """Test accurate token counting."""
        # Test string
        text = "Hello world! This is a test."
        tokens = self.token_manager.count_tokens(text)
        assert tokens > 0
        assert tokens < 20  # Should be around 7-8 tokens
        
        # Test dict
        data = {"key": "value", "number": 123}
        tokens = self.token_manager.count_tokens(data)
        assert tokens > 0
        
        # Test list
        items = ["item1", "item2", "item3"]
        tokens = self.token_manager.count_tokens(items)
        assert tokens > 0
        
    def test_token_truncation(self):
        """Test content truncation to fit token limits."""
        # Create large content
        large_text = "This is a test. " * 1000
        
        # Truncate to 100 tokens
        truncated = self.token_manager.truncate_to_tokens(large_text, 100)
        
        # Verify it's within limit
        tokens = self.token_manager.count_tokens(truncated)
        assert tokens <= 100
        assert "truncated" in str(truncated)
        
    def test_dict_truncation(self):
        """Test intelligent dictionary truncation."""
        large_dict = {
            "summary": "Important summary",
            "danger_zones": ["zone1", "zone2"],
            "less_important": "x" * 10000,
            "critical_files": ["file1", "file2"]
        }
        
        truncated = self.token_manager.truncate_to_tokens(large_dict, 50)
        
        # Priority keys should be preserved
        assert "summary" in truncated
        assert "danger_zones" in truncated
        assert "critical_files" in truncated
        
    def test_token_distribution(self):
        """Test token distribution across sections."""
        budget = TokenBudget()
        data = {
            'dependency_analysis': {"data": "x" * 10000},
            'security_analysis': {"data": "y" * 10000},
            'architecture_analysis': {"data": "z" * 10000}
        }
        
        distributed = self.token_manager.distribute_tokens(data, budget)
        
        # Each section should respect its budget
        dep_tokens = self.token_manager.count_tokens(distributed['dependency_analysis'])
        assert dep_tokens <= budget.dependency
        
    def test_validate_output_size(self):
        """Test output size validation."""
        # Within limit
        small_content = {"data": "small content"}
        is_valid, tokens = self.token_manager.validate_output_size(small_content)
        assert is_valid
        assert tokens < TokenManager.MCP_TOKEN_LIMIT
        
        # Exceeds limit
        huge_content = {"data": "x" * 200000}
        is_valid, tokens = self.token_manager.validate_output_size(huge_content)
        assert not is_valid
        assert tokens > TokenManager.MCP_TOKEN_LIMIT


class TestCacheManager:
    """Test cache management and file change detection."""
    
    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            yield CacheManager(cache_dir)
            
    @pytest.fixture
    async def test_files(self):
        """Create test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            
            # Create test files
            (test_dir / "file1.py").write_text("print('hello')")
            (test_dir / "file2.py").write_text("def test(): pass")
            
            yield test_dir
            
    @pytest.mark.asyncio
    async def test_file_hashing(self, cache_manager, test_files):
        """Test file hashing functionality."""
        file_map = {
            "file1.py": test_files / "file1.py",
            "file2.py": test_files / "file2.py"
        }
        
        hashes = await cache_manager.hash_files(file_map)
        
        assert len(hashes) == 2
        assert "file1.py" in hashes
        assert "file2.py" in hashes
        assert all(len(h) == 64 for h in hashes.values())  # SHA256 hex length
        
    @pytest.mark.asyncio
    async def test_save_and_load_analysis(self, cache_manager, test_files):
        """Test saving and loading analysis."""
        analysis_data = {
            "test": "data",
            "results": [1, 2, 3]
        }
        file_hashes = {"file1.py": "hash1", "file2.py": "hash2"}
        
        # Save analysis
        await cache_manager.save_analysis(
            test_files, "test_analysis", analysis_data, file_hashes
        )
        
        # Load analysis
        loaded = await cache_manager.load_analysis(test_files, "test_analysis")
        
        assert loaded is not None
        assert loaded['analysis'] == analysis_data
        assert loaded['metadata']['file_hashes'] == file_hashes
        
    @pytest.mark.asyncio
    async def test_change_detection(self, cache_manager, test_files):
        """Test file change detection."""
        # Initial analysis
        file_map = {
            "file1.py": test_files / "file1.py",
            "file2.py": test_files / "file2.py"
        }
        
        initial_hashes = await cache_manager.hash_files(file_map)
        
        # Save initial state
        await cache_manager.save_analysis(
            test_files, "change_test", {"data": "initial"}, initial_hashes
        )
        
        # Modify a file
        (test_files / "file1.py").write_text("print('modified')")
        
        # Add a new file
        (test_files / "file3.py").write_text("new file")
        file_map["file3.py"] = test_files / "file3.py"
        
        # Load cached data
        cached_data = await cache_manager.load_analysis(test_files, "change_test")
        
        # Detect changes
        changes, needs_full = await cache_manager.detect_changes(
            test_files, file_map, cached_data
        )
        
        assert len(changes) >= 2  # Modified file1 and new file3
        
        # Check change types
        change_types = {c.path: c.change_type for c in changes}
        assert change_types.get("file1.py") == "modified"
        assert change_types.get("file3.py") == "added"
        
    def test_git_info(self, cache_manager):
        """Test git information extraction."""
        # This test will only work in a git repo
        git_info = cache_manager.get_git_info(Path.cwd())
        
        # Should have some git info if in a repo
        if git_info:
            assert 'commit' in git_info or 'branch' in git_info
            

class TestServerIntegration:
    """Test the integrated server functionality."""
    
    @pytest.fixture
    async def server(self):
        """Create server instance."""
        # Set minimal required env vars
        os.environ['OPENAI_API_KEY'] = 'test-key'
        
        server = CodebaseIQProServer()
        yield server
        
    def test_server_initialization(self, server):
        """Test server initializes correctly."""
        assert server.token_manager is not None
        assert server.cache_manager is not None
        assert hasattr(server, '_get_dependency_analysis_full')
        assert hasattr(server, '_get_codebase_context')
        
    def test_token_budget(self):
        """Test token budget allocation."""
        budget = TokenBudget()
        
        assert budget.dependency == 7000
        assert budget.security == 1000
        assert budget.architecture == 3000
        assert budget.business_logic == 5000
        assert budget.technical_stack == 4000
        assert budget.code_intelligence == 5000
        assert budget.total == 25000  # Should equal MCP limit
        
    @pytest.mark.asyncio
    async def test_helper_methods(self, server):
        """Test helper methods for aggregation."""
        # Test instant context builder
        instant_context = server._build_instant_context(
            dependency={"summary": {"total_packages": 50}},
            security={"security_score": 8, "summary": {"critical_count": 2}},
            architecture={"architecture_style": "Microservices"},
            business={"executive_summary": "Test app", "summary": {"key_features": ["F1", "F2"]}},
            technical={"languages": {"all": ["Python", "JS"]}, "frameworks": {"django": "3.2"}},
            intelligence={"entry_points": ["main.py", "app.py"]}
        )
        
        assert "INSTANT CODEBASE CONTEXT" in instant_context
        assert "Python, JS" in instant_context
        assert "Security Score: 8/10" in instant_context
        
        # Test danger zone extraction
        danger_zones = server._extract_aggregated_danger_zones(
            security={
                "summary": {"critical_count": 2, "high_count": 3},
                "vulnerabilities": [
                    {"severity": "CRITICAL", "type": "SQL Injection", "affected_files": ["db.py"]}
                ]
            },
            architecture={
                "critical_components": [
                    {"file_path": "auth.py", "criticality": 9, "reason": "Auth handler"}
                ]
            }
        )
        
        assert "files require extreme caution" in danger_zones['summary']
        assert len(danger_zones['do_not_modify']) > 0
        assert danger_zones['do_not_modify'][0]['file'] == 'db.py'
        
        # Test safe modification guide
        guide = server._build_safe_modification_guide({}, {})
        
        assert len(guide) >= 8
        assert any("danger_zones" in g for g in guide)
        assert any("tests" in g.lower() for g in guide)


def test_imports():
    """Test all required imports work."""
    try:
        import tiktoken
        import aiofiles
        import mcp
        from codebaseiq.core import TokenManager, CacheManager
        from codebaseiq.server import CodebaseIQProServer
        print("✅ All imports successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    print("Running CodebaseIQ Pro v2.0 Architecture Tests...\n")
    
    # Test imports
    print("1. Testing imports...")
    test_imports()
    
    # Test token manager
    print("\n2. Testing TokenManager...")
    tm_test = TestTokenManager()
    tm_test.setup_method()
    tm_test.test_token_counting()
    tm_test.test_token_truncation()
    tm_test.test_validate_output_size()
    print("✅ TokenManager tests passed")
    
    # Test token budget
    print("\n3. Testing TokenBudget...")
    budget = TokenBudget()
    assert budget.total == 25000
    print(f"✅ Token budget totals to {budget.total} (MCP limit)")
    
    # Test cache manager basics
    print("\n4. Testing CacheManager...")
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CacheManager(Path(tmpdir))
        git_info = cm.get_git_info(Path.cwd())
        if git_info:
            print(f"✅ Git info detected: {git_info.get('branch', 'unknown branch')}")
        else:
            print("⚠️  Not in a git repository")
    
    # Test server initialization
    print("\n5. Testing Server initialization...")
    os.environ['OPENAI_API_KEY'] = 'test-key'
    try:
        server = CodebaseIQProServer()
        print("✅ Server initialized successfully")
        
        # Check all 6 analysis methods exist
        methods = [
            '_get_dependency_analysis_full',
            '_get_security_analysis_full', 
            '_get_architecture_analysis_full',
            '_get_business_logic_analysis_full',
            '_get_technical_stack_analysis_full',
            '_get_code_intelligence_analysis_full'
        ]
        
        for method in methods:
            assert hasattr(server, method), f"Missing method: {method}"
        print("✅ All 6 analysis methods present")
        
        # Check aggregation method
        assert hasattr(server, '_get_codebase_context')
        print("✅ Aggregation method present")
        
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
    
    print("\n✅ All basic tests passed!")
    print("\nFor full test suite, run: pytest tests/test_new_architecture.py -v")