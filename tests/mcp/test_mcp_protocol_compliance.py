#!/usr/bin/env python3
"""
MCP Protocol Compliance Tests

Tests the ACTUAL MCP protocol format responses, not just the internal methods.
This ensures all responses match the Model Context Protocol specification.
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any
import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from codebaseiq.server import CodebaseIQProServer
import mcp.types as types


class TestMCPProtocolCompliance:
    """Test that all MCP responses follow the protocol specification"""
    
    @pytest.fixture
    async def server(self):
        """Create a server instance for testing"""
        server = CodebaseIQProServer()
        # Don't run the server, just use the instance
        return server
    
    @pytest.mark.asyncio
    async def test_response_format_is_list_of_content(self, server):
        """Test that all responses return List[types.TextContent]"""
        # Test a simple tool that should always work
        response = await server.handle_call_tool("get_analysis_summary", {})
        
        # PROTOCOL REQUIREMENT: Response must be a list
        assert isinstance(response, list), f"Response must be a list, got {type(response)}"
        
        # PROTOCOL REQUIREMENT: All items must be TextContent objects
        for item in response:
            assert isinstance(item, types.TextContent), f"Response items must be TextContent, got {type(item)}"
            assert hasattr(item, 'type'), "TextContent must have 'type' attribute"
            assert hasattr(item, 'text'), "TextContent must have 'text' attribute"
            assert item.type == "text", f"TextContent type must be 'text', got {item.type}"
            assert isinstance(item.text, str), f"TextContent.text must be string, got {type(item.text)}"
    
    @pytest.mark.asyncio
    async def test_response_content_is_valid_json(self, server):
        """Test that response content is valid JSON"""
        response = await server.handle_call_tool("get_analysis_summary", {})
        
        assert len(response) > 0, "Response must not be empty"
        
        # PROTOCOL REQUIREMENT: Content must be valid JSON
        try:
            content = json.loads(response[0].text)
            assert isinstance(content, dict), "Response content should be a dictionary"
        except json.JSONDecodeError as e:
            pytest.fail(f"Response content must be valid JSON: {e}")
    
    @pytest.mark.asyncio
    async def test_error_response_format(self, server):
        """Test that errors also follow MCP protocol format"""
        # Call with invalid tool name
        response = await server.handle_call_tool("invalid_tool_name", {})
        
        # PROTOCOL REQUIREMENT: Errors must still return List[TextContent]
        assert isinstance(response, list), "Error response must be a list"
        assert len(response) == 1, "Error response should have one item"
        assert isinstance(response[0], types.TextContent), "Error must be TextContent"
        
        # Verify error content
        error_content = json.loads(response[0].text)
        assert 'error' in error_content, "Error response must contain 'error' field"
        assert 'tool' in error_content, "Error response must identify the tool"
    
    @pytest.mark.asyncio
    async def test_validation_error_format(self, server):
        """Test that validation errors follow protocol"""
        # Call check_understanding without required parameter
        response = await server.handle_call_tool("check_understanding", {})
        
        # PROTOCOL REQUIREMENT: Validation errors must return List[TextContent]
        assert isinstance(response, list), "Validation error must be a list"
        assert isinstance(response[0], types.TextContent), "Validation error must be TextContent"
        
        error_content = json.loads(response[0].text)
        assert error_content['type'] == 'validation_error', "Should identify as validation error"
        assert 'implementation_plan' in error_content['error'], "Should mention missing parameter"
    
    @pytest.mark.asyncio
    async def test_all_tools_return_correct_format(self, server):
        """Test that every defined tool returns correct format"""
        # Get list of all tools
        tools = await server.handle_list_tools()
        
        # Test each tool with minimal valid arguments
        test_args = {
            "get_codebase_context": {"refresh": False},
            "check_understanding": {"implementation_plan": "test plan"},
            "get_impact_analysis": {"file_path": "test.py"},
            "get_and_set_the_codebase_knowledge_foundation": {"path": "."},
            "update_cached_knowledge_foundation": {"path": "."},
            "semantic_code_search": {"query": "test"},
            "find_similar_code": {"entity_path": "test.py"},
            "get_analysis_summary": {},
            "get_danger_zones": {},
            "get_dependencies": {},
            "get_ai_knowledge_package": {},
            "get_business_context": {},
            "get_modification_guidance": {"file_path": "test.py"},
            "get_dependency_analysis": {},
            "get_security_analysis": {},
            "get_architecture_analysis": {},
            "get_business_logic_analysis": {},
            "get_technical_stack_analysis": {},
            "get_code_intelligence_analysis": {}
        }
        
        for tool in tools:
            if tool.name in test_args:
                print(f"Testing protocol compliance for: {tool.name}")
                response = await server.handle_call_tool(tool.name, test_args[tool.name])
                
                # Every tool must return List[TextContent]
                assert isinstance(response, list), f"{tool.name} must return a list"
                assert all(isinstance(item, types.TextContent) for item in response), \
                    f"{tool.name} must return TextContent objects"
    
    @pytest.mark.asyncio
    async def test_empty_arguments_handling(self, server):
        """Test protocol compliance with None and empty arguments"""
        # Test with None arguments
        response = await server.handle_call_tool("get_analysis_summary", None)
        assert isinstance(response, list), "Must handle None arguments"
        assert isinstance(response[0], types.TextContent), "Must return TextContent"
        
        # Test with empty dict
        response = await server.handle_call_tool("get_analysis_summary", {})
        assert isinstance(response, list), "Must handle empty dict"
        assert isinstance(response[0], types.TextContent), "Must return TextContent"
    
    @pytest.mark.asyncio 
    async def test_malformed_arguments_handling(self, server):
        """Test protocol compliance with malformed arguments"""
        # This should trigger TypeError handling
        # Note: The server converts non-dict to error, not passing it through
        response = await server.handle_call_tool("get_analysis_summary", "not a dict")
        
        # Must still return proper format
        assert isinstance(response, list), "Malformed args must return list"
        assert isinstance(response[0], types.TextContent), "Malformed args must return TextContent"
        
        error_content = json.loads(response[0].text)
        assert 'error' in error_content, "Should indicate error"
        assert 'type' in error_content, "Should indicate error type"
    
    @pytest.mark.asyncio
    async def test_large_response_handling(self, server):
        """Test protocol compliance with large responses"""
        # get_dependency_analysis can return large responses
        response = await server.handle_call_tool("get_dependency_analysis", {"force_refresh": False})
        
        # Must still follow protocol even for large responses
        assert isinstance(response, list), "Large responses must be list"
        assert isinstance(response[0], types.TextContent), "Large responses must be TextContent"
        
        # Content should be parseable JSON even if large
        content = json.loads(response[0].text)
        assert isinstance(content, dict), "Large response must be valid JSON dict"
    
    def test_protocol_never_returns_raw_strings(self):
        """Ensure we NEVER return raw strings (the original bug)"""
        # Check the source code to ensure no json.dumps returns
        import inspect
        # Get the internal handler source
        server_module = sys.modules['codebaseiq.server']
        source_file = inspect.getsourcefile(server_module.CodebaseIQProServer)
        
        with open(source_file, 'r') as f:
            source = f.read()
        
        # Check the handle_call_tool section
        # Should not have any direct string returns
        assert "return json.dumps(result, indent=2)" not in source, \
            "handle_call_tool must never return raw JSON strings"
        
        # Should always return list of TextContent
        assert "return [" in source and "types.TextContent(" in source, \
            "handle_call_tool should return list of content objects"


if __name__ == "__main__":
    # Run with detailed output
    pytest.main([__file__, "-v", "-s"])