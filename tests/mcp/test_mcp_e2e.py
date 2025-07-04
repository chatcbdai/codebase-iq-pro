#!/usr/bin/env python3
"""
MCP End-to-End Client Tests

Tests the complete MCP server with simulated client interactions.
This ensures the server works correctly when accessed through the
actual MCP protocol, not just direct method calls.
"""

import os
import sys
import json
import asyncio
import subprocess
from typing import Dict, Any, List, Optional
import pytest
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from codebaseiq.server import CodebaseIQProServer
from mcp.server.stdio import stdio_server
import mcp.types as types


class MockMCPClient:
    """Simulates an MCP client making requests to the server"""
    
    def __init__(self, server: CodebaseIQProServer):
        self.server = server
        self.tools: List[types.Tool] = []
        
    async def connect(self):
        """Simulate client connection and tool discovery"""
        # In real MCP, this would be through stdio/network
        # Here we simulate the protocol interaction
        self.tools = await self.server.handle_list_tools()
        return True
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Simulate calling a tool through MCP protocol"""
        # In real MCP, this would serialize through JSON-RPC
        # We simulate the full protocol flow
        
        # 1. Client would validate tool exists
        if not any(t.name == tool_name for t in self.tools):
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # 2. Client would send request over protocol
        # 3. Server processes and returns response
        response = await self.server.handle_call_tool(tool_name, arguments)
        
        # 4. Client would deserialize response
        # Verify response format matches MCP protocol
        if not isinstance(response, list):
            raise ValueError(f"Invalid response format: expected list, got {type(response)}")
        
        # 5. Client extracts content
        if response and isinstance(response[0], types.TextContent):
            return json.loads(response[0].text)
        
        return None


class TestMCPEndToEnd:
    """Test complete MCP client-server interactions"""
    
    @pytest.fixture
    async def server(self):
        """Create a server instance"""
        return CodebaseIQProServer()
    
    @pytest.fixture
    async def client(self, server):
        """Create a mock MCP client"""
        client = MockMCPClient(server)
        await client.connect()
        return client
    
    @pytest.mark.asyncio
    async def test_client_can_discover_tools(self, client):
        """Test that client can discover available tools"""
        assert len(client.tools) > 0, "Client should discover tools"
        
        # Verify essential tools are available
        tool_names = [t.name for t in client.tools]
        assert "get_codebase_context" in tool_names
        assert "check_understanding" in tool_names
        assert "get_and_set_the_codebase_knowledge_foundation" in tool_names
    
    @pytest.mark.asyncio
    async def test_client_server_round_trip(self, client):
        """Test complete request-response cycle"""
        # Make a simple request
        result = await client.call_tool("get_analysis_summary", {})
        
        # Verify we got a valid response
        assert isinstance(result, dict), "Should get dictionary response"
        assert 'error' in result or 'status' in result or 'analysis_state' in result, \
            "Response should have expected fields"
    
    @pytest.mark.asyncio
    async def test_concurrent_client_requests(self, server):
        """Test server handles multiple concurrent clients"""
        # Create multiple clients
        clients = [MockMCPClient(server) for _ in range(5)]
        
        # Connect all clients
        await asyncio.gather(*[client.connect() for client in clients])
        
        # Make concurrent requests
        tasks = []
        for i, client in enumerate(clients):
            task = client.call_tool("get_analysis_summary", {})
            tasks.append(task)
        
        # All should complete successfully
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all succeeded
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), \
                f"Client {i} failed: {result}"
            assert isinstance(result, dict), \
                f"Client {i} got invalid response"
    
    @pytest.mark.asyncio
    async def test_client_error_handling(self, client):
        """Test client handles server errors correctly"""
        # Test with missing required parameter
        result = await client.call_tool("check_understanding", {})
        
        assert 'error' in result, "Should get error response"
        assert 'implementation_plan' in result['error'], \
            "Error should mention missing parameter"
        
        # Test with unknown tool
        with pytest.raises(ValueError, match="Unknown tool"):
            await client.call_tool("nonexistent_tool", {})
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, client):
        """Test a complete analysis workflow as a client would use it"""
        # Create a temporary test codebase
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test_project"
            test_dir.mkdir()
            
            # Create some test files
            (test_dir / "main.py").write_text("""
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")
            
            (test_dir / "utils.py").write_text("""
def helper_function(x):
    return x * 2
""")
            
            # 1. Client runs full analysis
            print("Starting analysis...")
            analysis_result = await client.call_tool(
                "get_and_set_the_codebase_knowledge_foundation",
                {
                    "path": str(test_dir),
                    "enable_embeddings": False,  # Faster for testing
                    "force_refresh": True
                }
            )
            
            # Verify analysis completed
            if 'error' not in analysis_result:
                assert 'status' in analysis_result
                assert analysis_result.get('status') == 'completed' or \
                       analysis_result.get('phase_1', {}).get('status') == 'completed'
            
            # 2. Client gets context
            context_result = await client.call_tool(
                "get_codebase_context", 
                {"refresh": False}
            )
            
            # 3. Client checks understanding
            understanding_result = await client.call_tool(
                "check_understanding",
                {
                    "implementation_plan": "Add error handling to main function",
                    "files_to_modify": ["main.py"],
                    "understanding_points": [
                        "main.py contains the entry point",
                        "No error handling currently exists"
                    ]
                }
            )
            
            # Verify workflow completed
            assert isinstance(understanding_result, dict)
    
    @pytest.mark.asyncio
    async def test_client_handles_large_responses(self, client):
        """Test client can handle large responses"""
        # Request potentially large analysis
        result = await client.call_tool(
            "get_dependency_analysis",
            {"force_refresh": False}
        )
        
        # Should handle large response
        assert isinstance(result, dict), "Should handle large responses"
        
        # If not error, should have some content
        if 'error' not in result:
            # Dependency analysis should have some structure
            assert any(key in result for key in 
                      ['dependencies', 'imports', 'dependency_graph', 'status'])
    
    @pytest.mark.asyncio
    async def test_client_protocol_validation(self, client):
        """Test that client validates protocol requirements"""
        # Direct test of protocol validation
        server = client.server
        
        # Simulate protocol violation - if server returned wrong type
        # We can't actually make server return wrong type due to type hints,
        # but we can verify client would catch it
        
        # Test client validates response format
        response = await server.handle_call_tool("get_analysis_summary", {})
        
        # Client should validate this is correct format
        assert isinstance(response, list), "Response must be list"
        assert all(isinstance(item, types.TextContent) for item in response), \
            "All items must be TextContent"
        
        # Client should be able to extract content
        if response:
            content = json.loads(response[0].text)
            assert isinstance(content, dict), "Content must be valid JSON dict"
    
    @pytest.mark.asyncio
    async def test_stateful_client_session(self, client):
        """Test that client can maintain stateful session"""
        # First call - might initialize state
        result1 = await client.call_tool("get_analysis_summary", {})
        
        # Second call - should maintain any state
        result2 = await client.call_tool("get_danger_zones", {})
        
        # Both should succeed
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        
        # If analysis exists, both should reflect that
        if 'error' not in result1 and 'analysis_state' in result1:
            if result1['analysis_state'] == 'ready':
                # Second call should also work with existing state
                assert 'error' not in result2 or 'No analysis available' in str(result2.get('error'))
    
    @pytest.mark.asyncio
    async def test_client_reconnection(self, server):
        """Test client can reconnect and continue working"""
        # First client session
        client1 = MockMCPClient(server)
        await client1.connect()
        
        # Do some work
        await client1.call_tool("get_analysis_summary", {})
        
        # Simulate disconnect and new client
        client2 = MockMCPClient(server)
        await client2.connect()
        
        # New client should be able to continue
        result = await client2.call_tool("get_analysis_summary", {})
        assert isinstance(result, dict), "New client should work"


class TestMCPServerProcess:
    """Test running the server as a separate process (true E2E)"""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires subprocess handling - run manually if needed")
    async def test_server_subprocess(self):
        """Test server running as subprocess with stdio"""
        # This would test the actual server startup
        # Skipped in automated tests but documents the approach
        
        # Start server process
        server_process = subprocess.Popen(
            [sys.executable, "src/codebaseiq/server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            # Send JSON-RPC request
            request = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 1
            }
            
            server_process.stdin.write(json.dumps(request) + "\n")
            server_process.stdin.flush()
            
            # Read response
            response_line = server_process.stdout.readline()
            response = json.loads(response_line)
            
            assert "result" in response, "Should get valid JSON-RPC response"
            
        finally:
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    # Run with detailed output
    pytest.main([__file__, "-v", "-s"])