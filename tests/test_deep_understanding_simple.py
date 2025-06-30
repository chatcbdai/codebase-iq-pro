#!/usr/bin/env python3
"""Simple test for Deep Understanding Agent without dependencies"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import directly from the module file, bypassing __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "deep_understanding_agent",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                 "src/codebaseiq/agents/deep_understanding_agent.py")
)
deep_understanding_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deep_understanding_module)

DeepUnderstandingAgent = deep_understanding_module.DeepUnderstandingAgent
CodeContext = deep_understanding_module.CodeContext

def test_simple():
    """Simple test of the Deep Understanding Agent"""
    
    agent = DeepUnderstandingAgent()
    
    # Test with a simple Python function
    test_code = '''
"""Authentication module for user login"""

def login(username, password):
    """Authenticate user and return token"""
    # Check credentials
    if username == "admin" and password == "secret":
        return {"token": "abc123", "user": username}
    return None
'''
    
    context = agent.analyze_file("auth.py", test_code)
    
    print("✅ Deep Understanding Agent Test Results:")
    print(f"  Purpose: {context.purpose}")
    print(f"  Business Logic: {context.business_logic}")
    print(f"  Language: {context.language}")
    print(f"  Modification Risk: {context.modification_risk}")
    print(f"  AI Guidance: {context.ai_guidance}")
    
    # Basic assertions
    assert context.language == "python"
    assert "auth" in context.purpose.lower() or "login" in context.purpose.lower()
    assert context.modification_risk is not None
    
    print("\n✅ Basic test passed! The Deep Understanding Agent is working.")

if __name__ == "__main__":
    test_simple()