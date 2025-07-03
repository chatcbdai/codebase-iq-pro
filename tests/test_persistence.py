#!/usr/bin/env python3
"""
Focused test suite for state persistence functionality.
Tests that the server correctly saves and restores state on startup.
"""

import os
import sys
import time
import json
import asyncio
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from codebaseiq.server import CodebaseIQProServer as CodebaseIQServer
from codebaseiq.core.cache_manager import CacheManager

# Test codebase path
TEST_CODEBASE_PATH = Path(__file__).parent.parent / "codebase_tester_for_codebaseIQ-pro"


class TestPersistence:
    """Test suite for state persistence functionality."""
    
    def __init__(self):
        self.cache_dir = Path(".codebaseiq_cache")
        self.test_results = []
        
    def log_test(self, name: str, passed: bool, message: str = ""):
        """Log test result."""
        result = {
            "name": name,
            "passed": passed,
            "message": message,
            "timestamp": datetime.now()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if message:
            print(f"         {message}")
            
    async def test_initial_analysis_creates_persistence(self):
        """Test that initial analysis creates persistence files."""
        print("\n🧪 Test 1: Initial Analysis Creates Persistence")
        print("-" * 50)
        
        # Clean up any existing cache
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            
        # Create server and run analysis
        server = CodebaseIQServer()
        await server.run()
        
        # Run initial analysis
        result = await server.handle_call_tool(
            "get_and_set_the_codebase_knowledge_foundation",
            {
                "path": str(TEST_CODEBASE_PATH),
                "force_refresh": True,
                "enable_embeddings": True
            }
        )
        
        # Check if analysis completed
        if result and result.get("status") == "completed":
            self.log_test("Analysis completed", True)
        else:
            self.log_test("Analysis completed", False, "Analysis did not complete successfully")
            return
            
        # Check if cache directory was created
        if self.cache_dir.exists():
            self.log_test("Cache directory created", True)
        else:
            self.log_test("Cache directory created", False, f"Directory {self.cache_dir} not found")
            return
            
        # Check for persistence files
        persistence_files = list(self.cache_dir.glob("analysis_state_*.json"))
        if persistence_files:
            self.log_test("Persistence files created", True, f"Found {len(persistence_files)} files")
            
            # Verify file content
            latest_file = max(persistence_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_file, 'r') as f:
                    state_data = json.load(f)
                    
                # Check required fields
                required_fields = ["timestamp", "analysis", "embeddings_ready", "git_commit"]
                missing_fields = [field for field in required_fields if field not in state_data]
                
                if not missing_fields:
                    self.log_test("Persistence file format", True)
                else:
                    self.log_test("Persistence file format", False, f"Missing fields: {missing_fields}")
                    
            except Exception as e:
                self.log_test("Persistence file format", False, f"Error reading file: {str(e)}")
        else:
            self.log_test("Persistence files created", False, "No persistence files found")
            
    async def test_new_instance_restores_state(self):
        """Test that a new server instance restores state from persistence."""
        print("\n🧪 Test 2: New Instance Restores State")
        print("-" * 50)
        
        # Ensure we have persistence files from previous test
        persistence_files = list(self.cache_dir.glob("analysis_state_*.json"))
        if not persistence_files:
            self.log_test("Prerequisites check", False, "No persistence files found from previous test")
            return
            
        # Get the original analysis data for comparison
        latest_file = max(persistence_files, key=lambda p: p.stat().st_mtime)
        with open(latest_file, 'r') as f:
            original_state = json.load(f)
            
        # Create a new server instance
        print("\nCreating new server instance...")
        new_server = CodebaseIQServer()
        
        # Measure restoration time
        start_time = time.time()
        await new_server.run()
        restoration_time = time.time() - start_time
        
        print(f"Restoration completed in {restoration_time:.3f}s")
        
        # Check if state was restored
        if hasattr(new_server, 'current_analysis') and new_server.current_analysis:
            self.log_test("State restored", True)
            
            # Verify the restored state matches
            if new_server.embeddings_ready == original_state.get("embeddings_ready"):
                self.log_test("Embeddings state matches", True)
            else:
                self.log_test("Embeddings state matches", False, 
                            f"Expected {original_state.get('embeddings_ready')}, got {new_server.embeddings_ready}")
                
            # Test that analysis summary is available without re-analysis
            summary_result = await new_server.handle_call_tool("get_analysis_summary", {})
            
            if summary_result and summary_result.get("has_analysis"):
                self.log_test("Analysis available after restore", True)
                
                # Verify file count matches
                original_files = original_state["analysis"].get("total_files", 0)
                restored_files = summary_result.get("total_files", 0)
                
                if original_files == restored_files:
                    self.log_test("File count matches", True, f"{restored_files} files")
                else:
                    self.log_test("File count matches", False, 
                                f"Expected {original_files}, got {restored_files}")
            else:
                self.log_test("Analysis available after restore", False, "No analysis found")
        else:
            self.log_test("State restored", False, "No current_analysis found")
            
    async def test_restoration_performance(self):
        """Test that restoration completes within 5 seconds."""
        print("\n🧪 Test 3: Restoration Performance (<5 seconds)")
        print("-" * 50)
        
        # Run multiple restoration tests
        restoration_times = []
        
        for i in range(3):
            print(f"\nRun {i+1}/3:")
            server = CodebaseIQServer()
            
            start_time = time.time()
            await server.run()
            restoration_time = time.time() - start_time
            
            restoration_times.append(restoration_time)
            print(f"  Restoration time: {restoration_time:.3f}s")
            
        # Calculate statistics
        avg_time = sum(restoration_times) / len(restoration_times)
        max_time = max(restoration_times)
        
        print(f"\nAverage restoration time: {avg_time:.3f}s")
        print(f"Maximum restoration time: {max_time:.3f}s")
        
        # Check if all runs were under 5 seconds
        if all(t < 5.0 for t in restoration_times):
            self.log_test("Performance requirement", True, 
                         f"All restorations < 5s (avg: {avg_time:.3f}s)")
        else:
            failed_runs = sum(1 for t in restoration_times if t >= 5.0)
            self.log_test("Performance requirement", False, 
                         f"{failed_runs}/3 runs exceeded 5s limit")
            
    async def test_cache_expiration(self):
        """Test that old cache files are ignored."""
        print("\n🧪 Test 4: Cache Expiration (7-day limit)")
        print("-" * 50)
        
        # Create an old cache file (8 days old)
        old_timestamp = datetime.now() - timedelta(days=8)
        old_cache_file = self.cache_dir / f"analysis_state_{int(old_timestamp.timestamp())}.json"
        
        # Create fake old state
        old_state = {
            "timestamp": old_timestamp.isoformat(),
            "analysis": {"old": True},
            "embeddings_ready": True,
            "git_commit": "old_commit"
        }
        
        with open(old_cache_file, 'w') as f:
            json.dump(old_state, f)
            
        # Create new server instance
        server = CodebaseIQServer()
        await server.run()
        
        # Check if old state was ignored
        if hasattr(server, 'current_analysis'):
            if server.current_analysis and server.current_analysis.get("old"):
                self.log_test("Old cache ignored", False, "Server loaded 8-day old cache")
            else:
                self.log_test("Old cache ignored", True, "Old cache was not loaded")
        else:
            self.log_test("Old cache ignored", True, "No state loaded (as expected)")
            
        # Clean up old file
        old_cache_file.unlink()
        
    async def test_git_aware_caching(self):
        """Test that cache is invalidated on git commit change."""
        print("\n🧪 Test 5: Git-Aware Caching")
        print("-" * 50)
        
        # This test would require actual git operations
        # For now, we'll simulate by checking the mechanism
        
        cache_manager = CacheManager(base_path=Path.cwd())
        
        # Get current git commit
        current_commit = cache_manager._get_git_commit()
        
        if current_commit:
            self.log_test("Git commit detection", True, f"Current commit: {current_commit[:8]}")
            
            # Test that cache would be invalidated with different commit
            # (This is a simulation - in real scenario, git commit would change)
            persistence_files = list(self.cache_dir.glob("analysis_state_*.json"))
            if persistence_files:
                latest_file = max(persistence_files, key=lambda p: p.stat().st_mtime)
                
                with open(latest_file, 'r') as f:
                    state_data = json.load(f)
                    
                stored_commit = state_data.get("git_commit")
                
                if stored_commit == current_commit:
                    self.log_test("Git commit stored", True)
                else:
                    self.log_test("Git commit stored", False, 
                                f"Stored: {stored_commit}, Current: {current_commit}")
        else:
            self.log_test("Git commit detection", True, "No git repository (cache always valid)")
            
    async def test_embeddings_not_regenerated(self):
        """Test that embeddings are not regenerated on restart."""
        print("\n🧪 Test 6: Embeddings Not Regenerated")
        print("-" * 50)
        
        # Create new server instance
        server = CodebaseIQServer()
        await server.run()
        
        # Try to run embedding agent again
        result = await server.handle_call_tool(
            "get_and_set_the_codebase_knowledge_foundation",
            {"path": str(TEST_CODEBASE_PATH)}
        )
        
        if result:
            # Check phase 4 (embedding) results
            phase_4 = result.get("phase_4", {})
            embeddings_exists = phase_4.get("embeddings_exists", False)
            embeddings_created = phase_4.get("embeddings_created", 0)
            
            if embeddings_exists and embeddings_created == 0:
                self.log_test("Embeddings reuse", True, "Existing embeddings were reused")
            else:
                self.log_test("Embeddings reuse", False, 
                            f"Created {embeddings_created} new embeddings")
                
    async def test_search_works_after_restore(self):
        """Test that search functionality works after state restore."""
        print("\n🧪 Test 7: Search Works After Restore")
        print("-" * 50)
        
        # Create new server instance
        server = CodebaseIQServer()
        await server.run()
        
        # Test semantic search
        search_result = await server.handle_call_tool(
            "semantic_code_search",
            {
                "query": "authentication",
                "top_k": 5
            }
        )
        
        if search_result and "results" in search_result:
            if len(search_result["results"]) > 0:
                self.log_test("Semantic search after restore", True, 
                            f"Found {len(search_result['results'])} results")
            else:
                self.log_test("Semantic search after restore", False, "No results found")
        else:
            self.log_test("Semantic search after restore", False, "Search failed")
            
        # Test find similar code
        similar_result = await server.handle_call_tool(
            "find_similar_code",
            {
                "entity_path": "src/main.ts",
                "top_k": 3
            }
        )
        
        if similar_result and "similar_entities" in similar_result:
            self.log_test("Find similar after restore", True, 
                         f"Found {len(similar_result['similar_entities'])} similar files")
        else:
            self.log_test("Find similar after restore", False, "Find similar failed")
            
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 PERSISTENCE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["passed"])
        failed_tests = total_tests - passed_tests
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"\nSuccess Rate: {(passed_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print("\nFailed Tests:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"  - {result['name']}: {result['message']}")
                    
        print("\n" + "=" * 60)
        
    async def run_all_tests(self):
        """Run all persistence tests."""
        print("\n🚀 Starting State Persistence Tests")
        print("=" * 60)
        print(f"Test Codebase: {TEST_CODEBASE_PATH}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Run test suites in order
        await self.test_initial_analysis_creates_persistence()
        await self.test_new_instance_restores_state()
        await self.test_restoration_performance()
        await self.test_cache_expiration()
        await self.test_git_aware_caching()
        await self.test_embeddings_not_regenerated()
        await self.test_search_works_after_restore()
        
        self.print_summary()


async def main():
    """Main test runner."""
    tester = TestPersistence()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())