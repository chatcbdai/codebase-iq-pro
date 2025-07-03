#!/usr/bin/env python3
"""
Comprehensive test suite for CodebaseIQ Pro MCP Server.
Tests all MCP tools against a real codebase (codebase_tester_for_codebaseIQ-pro).
"""

import os
import sys
import json
import time
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from codebaseiq.server import CodebaseIQProServer as CodebaseIQServer

# Test codebase path
TEST_CODEBASE_PATH = Path(__file__).parent.parent / "codebase_tester_for_codebaseIQ-pro"

class TestMCPServer:
    """Comprehensive test suite for CodebaseIQ Pro MCP Server."""
    
    def __init__(self):
        self.server = None
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "performance_metrics": {},
            "tool_results": {}
        }
        
    async def setup(self):
        """Set up the test environment."""
        print("🔧 Setting up test environment...")
        
        # Initialize the server
        self.server = CodebaseIQServer()
        await self.server.run()
        
        # Clear any existing cache for clean testing
        cache_dir = Path(".codebaseiq_cache")
        if cache_dir.exists():
            print("  Clearing existing cache...")
            shutil.rmtree(cache_dir)
            
        print("✅ Test environment ready\n")
        
    async def teardown(self):
        """Clean up after tests."""
        print("\n🧹 Cleaning up test environment...")
        # Server cleanup is handled by the server itself
        
    async def test_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single MCP tool and return results."""
        self.test_results["total_tests"] += 1
        
        print(f"\n📋 Testing tool: {tool_name}")
        print(f"   Arguments: {json.dumps(arguments, indent=2)}")
        
        start_time = time.time()
        
        try:
            # Call the tool through the server's handle_call_tool method
            result = await self.server.handle_call_tool(tool_name, arguments)
            
            elapsed_time = time.time() - start_time
            
            # Store performance metrics
            if tool_name not in self.test_results["performance_metrics"]:
                self.test_results["performance_metrics"][tool_name] = []
            self.test_results["performance_metrics"][tool_name].append(elapsed_time)
            
            # Store tool results
            self.test_results["tool_results"][tool_name] = result
            
            print(f"✅ Tool {tool_name} completed in {elapsed_time:.2f}s")
            self.test_results["passed"] += 1
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Tool {tool_name} failed after {elapsed_time:.2f}s: {str(e)}"
            print(f"❌ {error_msg}")
            self.test_results["failed"] += 1
            self.test_results["errors"].append(error_msg)
            return None
            
    async def test_initial_analysis(self):
        """Test the complete codebase analysis."""
        print("\n🧪 TEST 1: Complete Codebase Analysis")
        print("=" * 50)
        
        result = await self.test_tool(
            "get_and_set_the_codebase_knowledge_foundation",
            {
                "path": str(TEST_CODEBASE_PATH),
                "force_refresh": True,
                "enable_embeddings": True
            }
        )
        
        if result:
            # Verify key components are present
            assert "analysis_id" in result, "Missing analysis_id"
            assert "status" in result, "Missing status"
            assert result["status"] == "completed", f"Analysis not completed: {result['status']}"
            
            # Check phase results
            for phase in ["phase_1", "phase_2", "phase_3", "phase_4"]:
                assert phase in result, f"Missing {phase} results"
                assert result[phase]["status"] == "completed", f"{phase} not completed"
                
            print(f"\n📊 Analysis Summary:")
            print(f"   Total files analyzed: {result.get('total_files', 'N/A')}")
            print(f"   Total entities: {result.get('total_entities', 'N/A')}")
            print(f"   Embeddings created: {result['phase_4'].get('embeddings_created', 'N/A')}")
            
    async def test_state_persistence(self):
        """Test state persistence and restoration."""
        print("\n🧪 TEST 2: State Persistence & Restoration")
        print("=" * 50)
        
        # First, ensure we have an analysis to persist
        print("📌 Ensuring analysis exists...")
        await self.test_tool(
            "get_and_set_the_codebase_knowledge_foundation",
            {"path": str(TEST_CODEBASE_PATH)}
        )
        
        # Now test restoration by creating a new server instance
        print("\n📌 Creating new server instance to test restoration...")
        
        # Save current server state
        original_server = self.server
        
        # Create new server instance
        new_server = CodebaseIQServer()
        self.server = new_server
        
        # Measure restoration time
        start_time = time.time()
        await new_server.run()
        restoration_time = time.time() - start_time
        
        print(f"⏱️  Restoration completed in {restoration_time:.2f}s")
        
        # Verify restoration was successful
        if restoration_time < 5.0:
            print("✅ Performance requirement met: restoration < 5 seconds")
        else:
            print(f"⚠️  Performance requirement NOT met: {restoration_time:.2f}s > 5s")
            
        # Test that analysis is available
        summary_result = await self.test_tool("get_analysis_summary", {})
        
        if summary_result and summary_result.get("has_analysis"):
            print("✅ State successfully restored")
            print(f"   Analysis timestamp: {summary_result.get('analysis_timestamp', 'N/A')}")
            print(f"   Total files: {summary_result.get('total_files', 'N/A')}")
        else:
            print("❌ State restoration failed")
            
        # Restore original server
        self.server = original_server
        
    async def test_semantic_search(self):
        """Test semantic code search functionality."""
        print("\n🧪 TEST 3: Semantic Code Search")
        print("=" * 50)
        
        test_queries = [
            "authentication logic",
            "database connection",
            "JWT token generation",
            "user registration",
            "error handling"
        ]
        
        for query in test_queries:
            result = await self.test_tool(
                "semantic_code_search",
                {
                    "query": query,
                    "top_k": 5
                }
            )
            
            if result and "results" in result:
                print(f"\n🔍 Query: '{query}'")
                print(f"   Found {len(result['results'])} results")
                
                # Show top result
                if result["results"]:
                    top_result = result["results"][0]
                    print(f"   Top match: {top_result.get('file_path', 'N/A')} (score: {top_result.get('score', 0):.3f})")
                    
    async def test_find_similar_code(self):
        """Test find similar code functionality."""
        print("\n🧪 TEST 4: Find Similar Code")
        print("=" * 50)
        
        # Test with a known file
        test_file = "src/app/routes/auth/auth.service.ts"
        
        result = await self.test_tool(
            "find_similar_code",
            {
                "entity_path": test_file,
                "top_k": 5,
                "similarity_threshold": 0.7
            }
        )
        
        if result and "similar_entities" in result:
            print(f"\n🔍 Similar to: {test_file}")
            print(f"   Found {len(result['similar_entities'])} similar files")
            
            for entity in result["similar_entities"][:3]:
                print(f"   - {entity.get('path', 'N/A')} (similarity: {entity.get('similarity', 0):.3f})")
                
    async def test_individual_analysis_tools(self):
        """Test all individual analysis tools."""
        print("\n🧪 TEST 5: Individual Analysis Tools")
        print("=" * 50)
        
        analysis_tools = [
            ("get_dependency_analysis", {"force_refresh": False}),
            ("get_security_analysis", {"severity_filter": "all"}),
            ("get_architecture_analysis", {"include_diagrams": True}),
            ("get_business_logic_analysis", {"include_workflows": True}),
            ("get_technical_stack_analysis", {"include_configs": True}),
            ("get_code_intelligence_analysis", {"include_patterns": True})
        ]
        
        for tool_name, args in analysis_tools:
            result = await self.test_tool(tool_name, args)
            
            if result:
                # Verify token count is within limits
                token_count = result.get("metadata", {}).get("token_count", 0)
                if token_count > 0:
                    print(f"   Token count: {token_count:,}")
                    if token_count <= 25000:
                        print(f"   ✅ Within 25K token limit")
                    else:
                        print(f"   ⚠️  Exceeds 25K token limit!")
                        
    async def test_ai_knowledge_package(self):
        """Test AI knowledge package generation."""
        print("\n🧪 TEST 6: AI Knowledge Package")
        print("=" * 50)
        
        result = await self.test_tool("get_ai_knowledge_package", {})
        
        if result:
            sections = [
                "instant_context",
                "danger_zones",
                "modification_guidelines",
                "business_understanding",
                "ai_instructions"
            ]
            
            for section in sections:
                if section in result:
                    print(f"   ✅ {section} present")
                else:
                    print(f"   ❌ {section} missing")
                    
    async def test_modification_guidance(self):
        """Test modification guidance for specific files."""
        print("\n🧪 TEST 7: Modification Guidance")
        print("=" * 50)
        
        test_files = [
            "src/app/routes/auth/auth.service.ts",
            "src/prisma/schema.prisma",
            "src/main.ts"
        ]
        
        for file_path in test_files:
            result = await self.test_tool(
                "get_modification_guidance",
                {"file_path": file_path}
            )
            
            if result:
                print(f"\n📄 File: {file_path}")
                print(f"   Risk Level: {result.get('risk_level', 'N/A')}")
                print(f"   Impacts {len(result.get('impacts', []))} other files")
                
                # Show safety checklist
                checklist = result.get("safety_checklist", [])
                if checklist:
                    print("   Safety Checklist:")
                    for item in checklist[:3]:
                        print(f"     - {item}")
                        
    async def test_codebase_context(self):
        """Test codebase context retrieval."""
        print("\n🧪 TEST 8: Codebase Context")
        print("=" * 50)
        
        result = await self.test_tool(
            "get_codebase_context",
            {"refresh": False}
        )
        
        if result:
            print(f"\n📊 Context Summary:")
            print(f"   Has analysis: {result.get('has_analysis', False)}")
            print(f"   Danger zones: {len(result.get('danger_zones', []))}")
            print(f"   Business context available: {'business_context' in result}")
            
    async def test_check_understanding(self):
        """Test AI understanding verification."""
        print("\n🧪 TEST 9: Check Understanding")
        print("=" * 50)
        
        result = await self.test_tool(
            "check_understanding",
            {
                "implementation_plan": "Add a new endpoint to retrieve user profile statistics",
                "files_to_modify": [
                    "src/app/routes/profile/profile.controller.ts",
                    "src/app/routes/profile/profile.service.ts"
                ],
                "understanding_points": [
                    "Profile routes use Express Router",
                    "Authentication is handled via JWT middleware",
                    "Prisma is used for database operations"
                ]
            }
        )
        
        if result:
            print(f"\n🎯 Understanding Score: {result.get('approval_score', 0):.1f}/10")
            print(f"   Status: {result.get('approval_status', 'N/A')}")
            
            if "suggestions" in result:
                print("   Suggestions:")
                for suggestion in result["suggestions"][:3]:
                    print(f"     - {suggestion}")
                    
    async def test_error_handling(self):
        """Test error handling and graceful degradation."""
        print("\n🧪 TEST 10: Error Handling")
        print("=" * 50)
        
        # Test with invalid path
        print("\n📌 Testing with invalid path...")
        result = await self.test_tool(
            "get_and_set_the_codebase_knowledge_foundation",
            {"path": "/nonexistent/path"}
        )
        
        # Test with invalid file for modification guidance
        print("\n📌 Testing with invalid file...")
        result = await self.test_tool(
            "get_modification_guidance",
            {"file_path": "nonexistent.ts"}
        )
        
        # Test search without analysis (should still work)
        print("\n📌 Testing search functionality...")
        result = await self.test_tool(
            "semantic_code_search",
            {"query": "test query"}
        )
        
        if result:
            print("   ✅ Search works even without full analysis")
        else:
            print("   ❌ Search failed without analysis")
            
    async def run_all_tests(self):
        """Run all tests in sequence."""
        print("\n🚀 Starting Comprehensive MCP Server Tests")
        print("=" * 70)
        print(f"Test Codebase: {TEST_CODEBASE_PATH}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        await self.setup()
        
        # Run all test suites
        test_suites = [
            self.test_initial_analysis,
            self.test_state_persistence,
            self.test_semantic_search,
            self.test_find_similar_code,
            self.test_individual_analysis_tools,
            self.test_ai_knowledge_package,
            self.test_modification_guidance,
            self.test_codebase_context,
            self.test_check_understanding,
            self.test_error_handling
        ]
        
        for test_suite in test_suites:
            try:
                await test_suite()
            except Exception as e:
                print(f"\n❌ Test suite {test_suite.__name__} failed: {str(e)}")
                self.test_results["errors"].append(f"Suite {test_suite.__name__}: {str(e)}")
                
        await self.teardown()
        
        # Print final summary
        self.print_summary()
        
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 70)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"\nTotal Tests: {self.test_results['total_tests']}")
        print(f"✅ Passed: {self.test_results['passed']}")
        print(f"❌ Failed: {self.test_results['failed']}")
        
        success_rate = (self.test_results['passed'] / self.test_results['total_tests'] * 100) if self.test_results['total_tests'] > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        if self.test_results['errors']:
            print("\n⚠️  Errors:")
            for error in self.test_results['errors']:
                print(f"   - {error}")
                
        # Performance summary
        print("\n⏱️  Performance Metrics:")
        for tool, times in self.test_results['performance_metrics'].items():
            avg_time = sum(times) / len(times)
            print(f"   {tool}: {avg_time:.2f}s average")
            
        print("\n" + "=" * 70)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


async def main():
    """Main test runner."""
    tester = TestMCPServer()
    await tester.run_all_tests()


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())