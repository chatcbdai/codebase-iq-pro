#!/usr/bin/env python3
"""
CodebaseIQ Pro - Main Entry Point

Run this file to start the CodebaseIQ Pro MCP server.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from codebaseiq import CodebaseIQProServer
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    try:
        server = CodebaseIQProServer()
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n👋 CodebaseIQ Pro Server stopped")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()