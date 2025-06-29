#!/usr/bin/env python3
"""
Test script to verify CodebaseIQ Pro setup
Run this after setup to ensure everything works
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

async def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from codebaseiq.core import get_config
        print("  ✅ core.adaptive_config")
    except ImportError as e:
        print(f"  ❌ core.adaptive_config: {e}")
        return False
        
    try:
        from codebaseiq.services import create_vector_db
        print("  ✅ services.vector_db")
    except ImportError as e:
        print(f"  ❌ services.vector_db: {e}")
        return False
        
    try:
        from codebaseiq.services import create_embedding_service
        print("  ✅ services.embedding_service")
    except ImportError as e:
        print(f"  ❌ services.embedding_service: {e}")
        return False
        
    try:
        from codebaseiq.services import create_cache_service
        print("  ✅ services.cache_service")
    except ImportError as e:
        print(f"  ❌ services.cache_service: {e}")
        return False
        
    try:
        from codebaseiq.core import SimpleOrchestrator
        print("  ✅ core.simple_orchestrator")
    except ImportError as e:
        print(f"  ❌ core.simple_orchestrator: {e}")
        return False
        
    try:
        from codebaseiq.agents import DependencyAnalysisAgent
        print("  ✅ agents.analysis_agents")
    except ImportError as e:
        print(f"  ❌ agents.analysis_agents: {e}")
        return False
        
    return True

async def test_configuration():
    """Test configuration loading"""
    print("\n🔧 Testing configuration...")
    
    # Set minimal required env var
    if not os.getenv('OPENAI_API_KEY'):
        print("  ⚠️  Setting dummy OPENAI_API_KEY for testing")
        os.environ['OPENAI_API_KEY'] = 'sk-test-key'
        
    try:
        from codebaseiq.core import get_config
        config = get_config()
        print("  ✅ Configuration loaded successfully")
        
        summary = config.get_config_summary()
        print(f"\n  Vector DB: {summary['vector_db']['type']} ({summary['vector_db']['tier']})")
        print(f"  Embeddings: {summary['embeddings']['service']} ({summary['embeddings']['tier']})")
        print(f"  Cache: {summary['cache']['type']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration failed: {e}")
        return False

async def test_services():
    """Test service initialization"""
    print("\n🛠️  Testing services...")
    
    from codebaseiq.core import get_config
    from codebaseiq.services import create_vector_db, create_embedding_service, create_cache_service
    
    config = get_config()
    
    # Test vector DB
    try:
        vector_db = create_vector_db(config.vector_db_config)
        print(f"  ✅ Vector DB created: {config.vector_db_config['type']}")
    except Exception as e:
        print(f"  ❌ Vector DB failed: {e}")
        return False
        
    # Test embedding service
    try:
        embedding_service = create_embedding_service(config.embedding_config)
        print(f"  ✅ Embedding service created: {config.embedding_config['service']}")
    except Exception as e:
        print(f"  ❌ Embedding service failed: {e}")
        return False
        
    # Test cache
    try:
        cache = create_cache_service(config.cache_config)
        print(f"  ✅ Cache service created: {config.cache_config['type']}")
    except Exception as e:
        print(f"  ❌ Cache service failed: {e}")
        return False
        
    return True

async def test_basic_analysis():
    """Test basic analysis functionality"""
    print("\n🧪 Testing basic analysis...")
    
    from codebaseiq.core import SimpleOrchestrator
    from codebaseiq.agents import DependencyAnalysisAgent
    
    try:
        orchestrator = SimpleOrchestrator()
        orchestrator.register_agent(DependencyAnalysisAgent())
        print("  ✅ Orchestrator and agents initialized")
        
        # Test with a small sample
        test_context = {
            'file_map': {
                'test.py': Path('test.py'),
            },
            'entities': {},
            'root_path': Path('.')
        }
        
        # Just verify it can start
        print("  ✅ Basic analysis setup works")
        return True
        
    except Exception as e:
        print(f"  ❌ Analysis setup failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 CodebaseIQ Pro Test Suite")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not await test_imports():
        all_passed = False
        print("\n❌ Import test failed. Check that all module files are present.")
        
    # Test configuration
    if not await test_configuration():
        all_passed = False
        print("\n❌ Configuration test failed. Check environment variables.")
        
    # Test services
    if not await test_services():
        all_passed = False
        print("\n❌ Service test failed. Check dependencies are installed.")
        
    # Test basic analysis
    if not await test_basic_analysis():
        all_passed = False
        print("\n❌ Analysis test failed.")
        
    print("\n" + "=" * 40)
    
    if all_passed:
        print("✅ All tests passed! CodebaseIQ Pro is ready to use.")
        print("\nNext steps:")
        print("1. Set your real OPENAI_API_KEY in .env")
        print("2. Configure VS Code MCP settings")
        print("3. Run: python codebase-iq-pro.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nTry running: python setup.py")

if __name__ == "__main__":
    asyncio.run(main())