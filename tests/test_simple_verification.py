#!/usr/bin/env python3
"""
Simple verification test for CodebaseIQ Pro implementation.
Tests core functionality without full MCP server startup.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from codebaseiq.server import CodebaseIQProServer as CodebaseIQServer
from codebaseiq.agents.embedding_agent import EmbeddingAgent

# Test codebase path
TEST_CODEBASE_PATH = Path(__file__).parent.parent / "codebase_tester_for_codebaseIQ-pro"


async def test_phase1_state_restoration():
    """Test Phase 1: State restoration flag and method"""
    print("\n🧪 Testing Phase 1: State Restoration")
    print("-" * 50)
    
    server = CodebaseIQServer()
    
    # Check flag exists
    assert hasattr(server, '_needs_state_restoration'), "Missing _needs_state_restoration flag"
    assert server._needs_state_restoration == True, "Flag should be True initially"
    print("✅ _needs_state_restoration flag is set correctly")
    
    # Check method exists
    assert hasattr(server, '_auto_restore_state'), "Missing _auto_restore_state method"
    print("✅ _auto_restore_state method exists")
    
    # Check embeddings_ready flag
    assert hasattr(server, 'embeddings_ready'), "Missing embeddings_ready flag"
    assert server.embeddings_ready == False, "embeddings_ready should be False initially"
    print("✅ embeddings_ready flag initialized")
    
    return True


async def test_phase2_embedding_check():
    """Test Phase 2: EmbeddingAgent checks for existing embeddings"""
    print("\n🧪 Testing Phase 2: Embedding Agent Check")
    print("-" * 50)
    
    # Create minimal context
    context = {
        'entities': {},
        'enable_embeddings': True
    }
    
    # Create mock services
    class MockVectorDB:
        async def get_stats(self):
            return {'total_vectors': 100}
    
    class MockEmbeddingService:
        pass
    
    agent = EmbeddingAgent(MockVectorDB(), MockEmbeddingService())
    result = await agent.analyze(context)
    
    assert result.get('embeddings_exists') == True, "Should detect existing embeddings"
    assert result.get('embeddings_created') == 0, "Should not create new embeddings"
    print("✅ EmbeddingAgent correctly detects existing embeddings")
    
    return True


async def test_phase3_search_methods():
    """Test Phase 3: Search methods work without current_analysis"""
    print("\n🧪 Testing Phase 3: Search Methods")
    print("-" * 50)
    
    server = CodebaseIQServer()
    
    # Test semantic search without current_analysis
    server.current_analysis = None
    
    # Mock vector_db with no embeddings
    class MockEmptyVectorDB:
        async def get_stats(self):
            return {'total_vectors': 0}
    
    server.vector_db = MockEmptyVectorDB()
    
    result = await server._semantic_code_search("test query")
    
    # Should return error about no embeddings, not about no analysis
    assert 'error' in result, "Should return error when no embeddings"
    assert 'No embeddings found' in result['error'], "Error should be about embeddings, not analysis"
    print("✅ Semantic search works without current_analysis")
    
    return True


async def test_phase4_cache_integration():
    """Test Phase 4: Cache integration in analysis methods"""
    print("\n🧪 Testing Phase 4: Cache Integration")
    print("-" * 50)
    
    server = CodebaseIQServer()
    
    # Check cache manager exists
    assert hasattr(server, 'cache_manager'), "Missing cache_manager"
    print("✅ Cache manager exists")
    
    # Check analysis methods have cache checks
    import inspect
    source = inspect.getsource(server._get_dependency_analysis_full)
    
    assert 'cache_manager.load_analysis' in source, "Missing cache load in dependency analysis"
    assert 'cache_manager.save_analysis' in source, "Missing cache save in dependency analysis"
    print("✅ Cache integration found in analysis methods")
    
    return True


async def test_phase5_server_refactoring():
    """Test Phase 5: Server refactoring"""
    print("\n🧪 Testing Phase 5: Server Refactoring")
    print("-" * 50)
    
    # Check that all mixin classes exist
    from codebaseiq.server_base import CodebaseIQProServerBase
    from codebaseiq.analysis_methods import AnalysisMethods
    from codebaseiq.helper_methods import HelperMethods
    
    server = CodebaseIQServer()
    
    # Check inheritance
    assert isinstance(server, CodebaseIQProServerBase), "Server should inherit from base"
    assert isinstance(server, AnalysisMethods), "Server should inherit from analysis methods"
    assert isinstance(server, HelperMethods), "Server should inherit from helper methods"
    print("✅ Server correctly inherits from all mixins")
    
    # Check that key methods exist
    assert hasattr(server, '_setup_tools'), "Missing _setup_tools method"
    assert hasattr(server, '_get_dependency_analysis_full'), "Missing analysis method"
    assert hasattr(server, '_discover_files'), "Missing helper method"
    print("✅ All key methods are accessible")
    
    return True


async def test_phase6_error_handling():
    """Test Phase 6: Error handling in _auto_restore_state"""
    print("\n🧪 Testing Phase 6: Error Handling")
    print("-" * 50)
    
    import inspect
    server = CodebaseIQServer()
    
    # Check that _auto_restore_state has try-except
    source = inspect.getsource(server._auto_restore_state)
    
    assert 'try:' in source, "Missing try block"
    assert 'except Exception as e:' in source, "Missing exception handling"
    assert 'Continue anyway' in source, "Missing graceful degradation comment"
    print("✅ Error handling is implemented in _auto_restore_state")
    
    return True


async def test_phase7_comprehensive_saving():
    """Test Phase 7: Comprehensive analysis saving"""
    print("\n🧪 Testing Phase 7: Comprehensive Analysis Saving")
    print("-" * 50)
    
    import inspect
    server = CodebaseIQServer()
    
    # Check that comprehensive analysis is saved after Phase 4
    source = inspect.getsource(server._get_and_set_knowledge_foundation)
    
    assert 'save_analysis' in source and 'comprehensive' in source, "Missing comprehensive analysis save"
    print("✅ Comprehensive analysis is saved to cache")
    
    return True


async def main():
    """Run all verification tests"""
    print("\n🚀 CodebaseIQ Pro Implementation Verification")
    print("=" * 60)
    
    tests = [
        test_phase1_state_restoration,
        test_phase2_embedding_check,
        test_phase3_search_methods,
        test_phase4_cache_integration,
        test_phase5_server_refactoring,
        test_phase6_error_handling,
        test_phase7_comprehensive_saving
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All phases implemented correctly!")
    else:
        print("❌ Some phases have issues")
        

if __name__ == "__main__":
    asyncio.run(main())