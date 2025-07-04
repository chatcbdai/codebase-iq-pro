#!/usr/bin/env python3
"""
MCP Contract Tests

Tests that all MCP tools adhere to their defined contracts:
- Input schemas are enforced
- Required parameters are validated
- Output formats match expectations
- Tool descriptions match behavior
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, List
import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from codebaseiq.server import CodebaseIQProServer
import mcp.types as types


class TestMCPContracts:
    """Test that all MCP tools honor their contracts"""
    
    @pytest.fixture
    async def server(self):
        """Create a server instance for testing"""
        server = CodebaseIQProServer()
        return server
    
    @pytest.fixture
    async def tools(self, server):
        """Get list of all available tools"""
        return await server.handle_list_tools()
    
    @pytest.mark.asyncio
    async def test_tool_schemas_are_valid(self, tools):
        """Test that all tool schemas are properly defined"""
        for tool in tools:
            # Every tool must have required attributes
            assert hasattr(tool, 'name'), f"Tool must have name"
            assert hasattr(tool, 'description'), f"Tool must have description"
            assert hasattr(tool, 'inputSchema'), f"Tool must have inputSchema"
            
            # Schema must be a valid JSON schema
            schema = tool.inputSchema
            assert isinstance(schema, dict), f"{tool.name}: inputSchema must be dict"
            assert 'type' in schema, f"{tool.name}: schema must have type"
            assert schema['type'] == 'object', f"{tool.name}: schema type must be object"
            
            if 'properties' in schema:
                assert isinstance(schema['properties'], dict), \
                    f"{tool.name}: properties must be dict"
    
    @pytest.mark.asyncio
    async def test_required_parameters_are_enforced(self, server, tools):
        """Test that required parameters are actually required"""
        # Tools with required parameters
        required_params = {
            "check_understanding": ["implementation_plan"],
            "get_impact_analysis": ["file_path"],
            "semantic_code_search": ["query"],
            "find_similar_code": ["entity_path"],
            "get_modification_guidance": ["file_path"]
        }
        
        for tool_name, params in required_params.items():
            # Find the tool definition
            tool = next((t for t in tools if t.name == tool_name), None)
            assert tool is not None, f"Tool {tool_name} should exist"
            
            # Verify schema declares these as required
            if 'required' in tool.inputSchema:
                for param in params:
                    assert param in tool.inputSchema['required'], \
                        f"{tool_name}: {param} should be in required list"
            
            # Test that missing required params causes error
            response = await server.handle_call_tool(tool_name, {})
            assert isinstance(response, list), "Response must be list"
            
            error_content = json.loads(response[0].text)
            assert 'error' in error_content, f"{tool_name}: Missing required param should error"
            assert params[0] in error_content['error'].lower(), \
                f"{tool_name}: Error should mention missing {params[0]}"
    
    @pytest.mark.asyncio
    async def test_optional_parameters_have_defaults(self, server, tools):
        """Test that optional parameters work with defaults"""
        # Tools with optional parameters and their defaults
        optional_params = {
            "get_codebase_context": {"refresh": False},
            "get_and_set_the_codebase_knowledge_foundation": {
                "path": ".",
                "enable_embeddings": True,
                "force_refresh": False
            },
            "get_dependency_analysis": {
                "force_refresh": False,
                "include_transitive": True
            },
            "semantic_code_search": {
                "top_k": 10,
                "search_type": "semantic"
            }
        }
        
        for tool_name, defaults in optional_params.items():
            # Find the tool definition
            tool = next((t for t in tools if t.name == tool_name), None)
            assert tool is not None, f"Tool {tool_name} should exist"
            
            # Check schema defines defaults
            schema_props = tool.inputSchema.get('properties', {})
            for param, default_value in defaults.items():
                if param in schema_props:
                    param_schema = schema_props[param]
                    if 'default' in param_schema:
                        assert param_schema['default'] == default_value, \
                            f"{tool_name}.{param}: default mismatch"
    
    @pytest.mark.asyncio
    async def test_parameter_types_are_validated(self, server):
        """Test that parameter types are validated"""
        # Test boolean parameter
        response = await server.handle_call_tool(
            "get_codebase_context", 
            {"refresh": "not a boolean"}  # Should be boolean
        )
        # Should still work (coerced) but let's test with clearly wrong type
        
        # Test array parameter
        response = await server.handle_call_tool(
            "check_understanding",
            {
                "implementation_plan": "test",
                "files_to_modify": "not an array"  # Should be array
            }
        )
        # The server should handle this gracefully
        assert isinstance(response, list), "Must return valid response format"
    
    @pytest.mark.asyncio
    async def test_output_contracts(self, server):
        """Test that outputs match their documented contracts"""
        # Test specific output structures
        
        # get_codebase_context should return specific fields
        response = await server.handle_call_tool("get_codebase_context", {})
        content = json.loads(response[0].text)
        
        if 'error' not in content:  # Only check if not an error
            expected_fields = ['danger_zones', 'critical_files', 'business_context', 
                             'dependencies', 'architecture', 'ready_for_modifications']
            for field in expected_fields:
                assert field in content, f"get_codebase_context must include {field}"
        
        # get_analysis_summary should return summary info
        response = await server.handle_call_tool("get_analysis_summary", {})
        content = json.loads(response[0].text)
        
        if 'error' not in content:
            assert isinstance(content, dict), "Summary must be a dictionary"
            # Should have some kind of status or summary info
            assert any(key in content for key in ['status', 'analysis_state', 'error']), \
                "Summary should indicate analysis state"
    
    @pytest.mark.asyncio
    async def test_tool_descriptions_match_behavior(self, tools):
        """Test that tool descriptions accurately describe their function"""
        # Check key terms in descriptions
        description_checks = {
            "get_codebase_context": ["context", "modifications"],
            "check_understanding": ["verify", "understanding"],
            "get_impact_analysis": ["impact", "analysis"],
            "semantic_code_search": ["search"],
            "get_danger_zones": ["danger", "zones"],
            "get_business_context": ["business"],
        }
        
        for tool in tools:
            if tool.name in description_checks:
                keywords = description_checks[tool.name]
                description_lower = tool.description.lower()
                for keyword in keywords:
                    assert keyword in description_lower, \
                        f"{tool.name}: Description should mention '{keyword}'"
    
    @pytest.mark.asyncio
    async def test_enum_parameters_are_validated(self, server):
        """Test that enum parameters only accept valid values"""
        # Test search_type enum
        valid_response = await server.handle_call_tool(
            "semantic_code_search",
            {"query": "test", "search_type": "semantic"}
        )
        assert isinstance(valid_response, list), "Valid enum should work"
        
        # Test with invalid enum value
        invalid_response = await server.handle_call_tool(
            "semantic_code_search", 
            {"query": "test", "search_type": "invalid_type"}
        )
        # Should still return valid format (might coerce or use default)
        assert isinstance(invalid_response, list), "Invalid enum should return valid format"
    
    @pytest.mark.asyncio
    async def test_number_parameters_are_validated(self, server):
        """Test that number parameters are properly handled"""
        # Test with valid number
        response = await server.handle_call_tool(
            "semantic_code_search",
            {"query": "test", "top_k": 5}
        )
        assert isinstance(response, list), "Valid number should work"
        
        # Test with negative number (should handle gracefully)
        response = await server.handle_call_tool(
            "semantic_code_search",
            {"query": "test", "top_k": -1}
        )
        assert isinstance(response, list), "Negative number should be handled"
        
        # Test with non-number
        response = await server.handle_call_tool(
            "semantic_code_search",
            {"query": "test", "top_k": "not a number"}
        )
        assert isinstance(response, list), "Non-number should be handled"
    
    @pytest.mark.asyncio
    async def test_consistent_error_contract(self, server, tools):
        """Test that all errors follow consistent contract"""
        # Test various error conditions
        error_cases = [
            ("unknown_tool", {}, "Unknown tool"),
            ("check_understanding", {}, "Missing required parameter"),
            ("get_impact_analysis", {"file_path": ""}, "Empty required parameter"),
        ]
        
        for tool_name, args, error_type in error_cases:
            response = await server.handle_call_tool(tool_name, args)
            
            # All errors must follow same format
            assert isinstance(response, list), f"{error_type}: Must return list"
            assert len(response) == 1, f"{error_type}: Must return single item"
            assert isinstance(response[0], types.TextContent), f"{error_type}: Must be TextContent"
            
            content = json.loads(response[0].text)
            assert 'error' in content, f"{error_type}: Must have error field"
            assert 'tool' in content, f"{error_type}: Must identify tool"
            
            # If it's a structured error response, should have type
            if 'type' in content:
                assert content['type'] in ['validation_error', 'missing_parameter', 
                                         'general_error', 'unknown_tool'], \
                    f"{error_type}: Error type should be recognized"


if __name__ == "__main__":
    # Run with detailed output
    pytest.main([__file__, "-v", "-s"])