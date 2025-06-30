#!/usr/bin/env python3
"""
Test CodebaseIQ Pro on itself
Demonstrates the enhanced understanding capabilities
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import server components
from codebaseiq.server import CodebaseIQProServer

# Set test API key
os.environ['OPENAI_API_KEY'] = 'test-key-12345'

async def test_self_analysis():
    """Test CodebaseIQ Pro analyzing itself"""
    print("🔍 CodebaseIQ Pro Self-Analysis")
    print("=" * 60)
    
    # Initialize server
    server = CodebaseIQProServer()
    
    # Analyze the codebase
    print("\n📊 Analyzing CodebaseIQ Pro codebase...")
    result = await server._analyze_codebase(
        path=os.path.dirname(os.path.abspath(__file__)),
        analysis_type="full",
        enable_embeddings=False
    )
    
    if result['status'] == 'success':
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Files analyzed: {result['files_analyzed']}")
        print(f"🤖 AI Ready: {result['ai_ready']}")
        
        # Get AI knowledge package
        ai_package = await server._get_ai_knowledge_package()
        
        print("\n📦 AI Knowledge Package Summary:")
        print("-" * 60)
        
        # Show instant context
        print(ai_package['instant_context'])
        
        # Show danger zones
        danger_zones = ai_package['danger_zones']
        print(f"\n⚠️ Danger Zones: {danger_zones['summary']}")
        
        if danger_zones.get('do_not_modify'):
            print("\n🔴 DO NOT MODIFY Files:")
            for item in danger_zones['do_not_modify'][:5]:
                print(f"  - {item['file']}")
                print(f"    Reason: {item['reason']}")
                
        if danger_zones.get('extreme_caution'):
            print("\n🟡 EXTREME CAUTION Files:")
            for item in danger_zones['extreme_caution'][:5]:
                print(f"  - {item['file']}")
                print(f"    Impact: {item['impact']} files affected")
                
        # Get business context
        business = await server._get_business_context()
        print(f"\n💼 Business Understanding:")
        print(f"  Domain entities: {len(business['domain_model'].get('entities', {}))}")
        print(f"  Key features: {', '.join(business['key_features'][:5])}")
        print(f"  Business rules: {len(business['business_rules'])}")
        
        # Test modification guidance for a critical file
        test_file = "src/codebaseiq/server.py"
        guidance = await server._get_modification_guidance(test_file)
        
        print(f"\n🛡️ Modification Guidance for {test_file}:")
        print(f"  Risk Level: {guidance.get('risk_level', 'N/A')}")
        print(f"  Impact: {guidance.get('impact_summary', 'N/A')}")
        print(f"  Purpose: {guidance.get('file_purpose', 'N/A')}")
        
        if guidance.get('safer_alternatives'):
            print("\n💡 Safer Alternatives:")
            for alt in guidance['safer_alternatives'][:3]:
                print(f"  - {alt}")
                
        # Show AI instructions preview
        print("\n🤖 AI Instructions Preview:")
        print("-" * 60)
        ai_instructions = ai_package.get('ai_instructions', '')
        print(ai_instructions[:500] + "..." if len(ai_instructions) > 500 else ai_instructions)
        
    else:
        print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
    print("\n" + "=" * 60)
    print("✨ CodebaseIQ Pro provides immediate, comprehensive understanding")
    print("🎯 AI assistants can now make safe, informed modifications")
    print("🛡️ Critical files are protected with clear guidance")

if __name__ == "__main__":
    asyncio.run(test_self_analysis())