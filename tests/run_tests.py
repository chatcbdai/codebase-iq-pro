#!/usr/bin/env python3
"""
Test runner for CodebaseIQ Pro MCP Server tests.
Provides options to run different test suites.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"🧪 {title}")
    print("=" * 70 + "\n")


async def run_persistence_tests():
    """Run persistence-focused tests."""
    print_header("Running Persistence Tests")
    
    from test_persistence import TestPersistence
    tester = TestPersistence()
    await tester.run_all_tests()


async def run_full_mcp_tests():
    """Run comprehensive MCP server tests."""
    print_header("Running Full MCP Server Tests")
    
    from test_mcp_server_full import TestMCPServer
    tester = TestMCPServer()
    await tester.run_all_tests()


async def run_quick_validation():
    """Run a quick validation of core functionality."""
    print_header("Running Quick Validation")
    
    from test_mcp_server_full import TestMCPServer
    tester = TestMCPServer()
    
    await tester.setup()
    
    # Run only essential tests
    print("Running essential tests only...")
    await tester.test_initial_analysis()
    await tester.test_semantic_search()
    await tester.test_ai_knowledge_package()
    
    await tester.teardown()
    tester.print_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CodebaseIQ Pro Test Runner")
    parser.add_argument(
        "test_suite",
        choices=["persistence", "full", "quick", "all"],
        help="Which test suite to run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
    
    print("\n🚀 CodebaseIQ Pro Test Runner")
    print(f"📁 Test Codebase: codebase_tester_for_codebaseIQ-pro")
    
    # Run selected test suite
    if args.test_suite == "persistence":
        asyncio.run(run_persistence_tests())
    elif args.test_suite == "full":
        asyncio.run(run_full_mcp_tests())
    elif args.test_suite == "quick":
        asyncio.run(run_quick_validation())
    elif args.test_suite == "all":
        # Run all test suites
        asyncio.run(run_persistence_tests())
        print("\n" + "-" * 70 + "\n")
        asyncio.run(run_full_mcp_tests())
    
    print("\n✅ Test run completed!")


if __name__ == "__main__":
    main()