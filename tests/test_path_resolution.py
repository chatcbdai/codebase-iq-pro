#!/usr/bin/env python3
"""
Test script to demonstrate path resolution in CodebaseIQ Pro
This shows how relative paths are resolved based on the current working directory
"""

import os
from pathlib import Path

def test_path_resolution():
    """Show how paths are resolved in different contexts"""
    
    print("=== Path Resolution Test ===\n")
    
    # Current working directory
    cwd = os.getcwd()
    print(f"Current Working Directory: {cwd}")
    
    # Test relative path resolution
    test_paths = [".", "./src", "../", "~/Desktop"]
    
    print("\nPath Resolution Examples:")
    print("-" * 50)
    
    for test_path in test_paths:
        try:
            # This is how the server resolves paths
            resolved = Path(test_path).resolve()
            expanded = Path(test_path).expanduser().resolve()
            
            print(f"\nInput path: '{test_path}'")
            print(f"  → Resolved: {resolved}")
            if test_path.startswith("~"):
                print(f"  → Expanded: {expanded}")
                
        except Exception as e:
            print(f"  → Error: {e}")
    
    print("\n" + "=" * 50)
    print("\nKey Insight:")
    print("When you run 'analyze_codebase path: \".\"' in Claude Code,")
    print("it analyzes the VS Code workspace directory, not the MCP server directory!")

if __name__ == "__main__":
    test_path_resolution()