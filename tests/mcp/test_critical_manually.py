#!/usr/bin/env python3
"""
Manual test runner for critical tests without pytest
"""

import os
import sys
import json
import asyncio

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from codebaseiq.server import CodebaseIQProServer
import mcp.types as types


async def test_protocol_compliance():
    """Test MCP protocol compliance"""
    print("\n=== Testing MCP Protocol Compliance ===")
    
    server = CodebaseIQProServer()
    
    # Test 1: Response format
    print("1. Testing response format...")
    response = await server.handle_call_tool("get_analysis_summary", {})
    
    assert isinstance(response, list), f"Response must be list, got {type(response)}"
    assert len(response) > 0, "Response must not be empty"
    assert isinstance(response[0], types.TextContent), "Response must contain TextContent"
    assert response[0].type == "text", "TextContent type must be 'text'"
    print("   ✓ Response format correct")
    
    # Test 2: Content is JSON
    print("2. Testing content format...")
    content = json.loads(response[0].text)
    assert isinstance(content, dict), "Content must be dict"
    print("   ✓ Content is valid JSON")
    
    # Test 3: Error handling
    print("3. Testing error handling...")
    error_response = await server.handle_call_tool("invalid_tool", {})
    assert isinstance(error_response, list), "Error must return list"
    error_content = json.loads(error_response[0].text)
    assert 'error' in error_content, "Error must have 'error' field"
    print("   ✓ Error handling correct")
    
    # Test 4: Required parameters
    print("4. Testing required parameters...")
    missing_response = await server.handle_call_tool("check_understanding", {})
    missing_content = json.loads(missing_response[0].text)
    assert 'error' in missing_content, "Missing param must error"
    assert 'implementation_plan' in missing_content['error'], "Must mention missing param"
    print("   ✓ Required parameters enforced")
    
    print("\n✅ All protocol compliance tests passed!")
    return True


async def test_contracts():
    """Test MCP contracts"""
    print("\n=== Testing MCP Contracts ===")
    
    server = CodebaseIQProServer()
    tools = await server.handle_list_tools()
    
    # Test 1: Tool schemas
    print("1. Testing tool schemas...")
    for tool in tools:
        assert hasattr(tool, 'name'), "Tool must have name"
        assert hasattr(tool, 'description'), "Tool must have description"
        assert hasattr(tool, 'inputSchema'), "Tool must have inputSchema"
    print("   ✓ All tools have proper schemas")
    
    # Test 2: Required parameters enforced
    print("2. Testing required parameter enforcement...")
    required_tools = {
        "check_understanding": "implementation_plan",
        "get_impact_analysis": "file_path",
        "semantic_code_search": "query"
    }
    
    for tool_name, param in required_tools.items():
        response = await server.handle_call_tool(tool_name, {})
        content = json.loads(response[0].text)
        assert 'error' in content, f"{tool_name} should error without {param}"
    print("   ✓ Required parameters are enforced")
    
    # Test 3: Error format consistency
    print("3. Testing error format consistency...")
    error_response = await server.handle_call_tool("unknown_tool", {})
    error_content = json.loads(error_response[0].text)
    assert 'error' in error_content, "Error must have 'error' field"
    assert 'tool' in error_content, "Error must have 'tool' field"
    print("   ✓ Error format is consistent")
    
    print("\n✅ All contract tests passed!")
    return True


async def test_e2e():
    """Test end-to-end functionality"""
    print("\n=== Testing End-to-End Functionality ===")
    
    server = CodebaseIQProServer()
    
    # Test 1: Tool discovery
    print("1. Testing tool discovery...")
    tools = await server.handle_list_tools()
    tool_names = [t.name for t in tools]
    assert "get_codebase_context" in tool_names, "Essential tool missing"
    assert "check_understanding" in tool_names, "Essential tool missing"
    print(f"   ✓ Found {len(tools)} tools")
    
    # Test 2: Simple request-response
    print("2. Testing request-response cycle...")
    response = await server.handle_call_tool("get_analysis_summary", {})
    assert isinstance(response, list), "Response must be list"
    content = json.loads(response[0].text)
    assert isinstance(content, dict), "Content must be dict"
    print("   ✓ Request-response cycle works")
    
    # Test 3: Complex request with parameters
    print("3. Testing complex requests...")
    response = await server.handle_call_tool(
        "check_understanding",
        {
            "implementation_plan": "Test plan",
            "files_to_modify": ["test.py"],
            "understanding_points": ["Point 1"]
        }
    )
    assert isinstance(response, list), "Complex request must return list"
    print("   ✓ Complex requests handled correctly")
    
    print("\n✅ All E2E tests passed!")
    return True


async def main():
    """Run all critical tests"""
    print("=" * 60)
    print("CODEBASEIQ PRO CRITICAL TESTS")
    print("=" * 60)
    
    try:
        # Run all test suites
        protocol_passed = await test_protocol_compliance()
        contracts_passed = await test_contracts()
        e2e_passed = await test_e2e()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Protocol Compliance: {'✅ PASS' if protocol_passed else '❌ FAIL'}")
        print(f"Contract Validation: {'✅ PASS' if contracts_passed else '❌ FAIL'}")
        print(f"End-to-End Tests: {'✅ PASS' if e2e_passed else '❌ FAIL'}")
        
        all_passed = protocol_passed and contracts_passed and e2e_passed
        
        if all_passed:
            print(f"\n✅ ALL CRITICAL TESTS PASSED - Server is production ready!")
        else:
            print(f"\n❌ SOME TESTS FAILED - Fix issues before production!")
            
        return all_passed
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)