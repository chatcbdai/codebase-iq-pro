#!/usr/bin/env python3
"""
Setup and dependency checker for CodebaseIQ Pro
Helps users install required dependencies and configure the environment
"""

import subprocess
import sys
import os
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_colored(message, color):
    print(f"{color}{message}{Colors.END}")

def check_python_version():
    """Check if Python version is 3.9+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_colored("❌ Python 3.9+ is required", Colors.RED)
        print(f"   Current version: {sys.version}")
        return False
    print_colored(f"✅ Python {version.major}.{version.minor} detected", Colors.GREEN)
    return True

def check_package_installed(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def install_requirements():
    """Install requirements from requirements.txt"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print_colored("❌ requirements.txt not found", Colors.RED)
        return False
        
    print_colored("\n📦 Installing required packages...", Colors.BLUE)
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        
        print_colored("✅ All packages installed successfully", Colors.GREEN)
        return True
        
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to install packages: {e}", Colors.RED)
        return False

def check_required_packages():
    """Check if core packages are installed"""
    core_packages = [
        ("mcp", "MCP (Model Context Protocol)"),
        ("openai", "OpenAI (for embeddings)"),
        ("tree_sitter", "Tree-sitter (for code analysis)"),
        ("networkx", "NetworkX (for dependency graphs)"),
        ("numpy", "NumPy (for embeddings)"),
        ("aiofiles", "aiofiles (for async file operations)"),
        ("cryptography", "Cryptography (for security)")
    ]
    
    optional_packages = [
        ("qdrant_client", "Qdrant (free vector database)"),
        ("pinecone", "Pinecone (premium vector database)"),
        ("voyageai", "Voyage AI (premium embeddings)"),
        ("redis", "Redis (distributed caching)")
    ]
    
    print_colored("\n🔍 Checking installed packages...", Colors.BLUE)
    
    # Check core packages
    missing_core = []
    for package, name in core_packages:
        if check_package_installed(package):
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")
            missing_core.append(package)
            
    # Check optional packages
    print_colored("\n📦 Optional packages:", Colors.BLUE)
    for package, name in optional_packages:
        if check_package_installed(package):
            print(f"  ✅ {name}")
        else:
            print(f"  ⚪ {name} (not installed)")
            
    return len(missing_core) == 0

def check_environment_variables():
    """Check if required environment variables are set"""
    print_colored("\n🔑 Checking environment variables...", Colors.BLUE)
    
    # Required
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print_colored("  ✅ OPENAI_API_KEY is set", Colors.GREEN)
    else:
        print_colored("  ❌ OPENAI_API_KEY is not set (REQUIRED)", Colors.RED)
        
    # Optional premium features
    voyage_key = os.getenv('VOYAGE_API_KEY')
    if voyage_key:
        print_colored("  ✅ VOYAGE_API_KEY is set (premium embeddings enabled)", Colors.GREEN)
    else:
        print_colored("  ⚪ VOYAGE_API_KEY not set (using OpenAI embeddings)", Colors.YELLOW)
        
    pinecone_key = os.getenv('PINECONE_API_KEY')
    if pinecone_key:
        print_colored("  ✅ PINECONE_API_KEY is set (premium vector DB enabled)", Colors.GREEN)
    else:
        print_colored("  ⚪ PINECONE_API_KEY not set (using local Qdrant)", Colors.YELLOW)
        
    redis_url = os.getenv('REDIS_URL')
    if redis_url:
        print_colored("  ✅ REDIS_URL is set (distributed caching enabled)", Colors.GREEN)
    else:
        print_colored("  ⚪ REDIS_URL not set (using in-memory cache)", Colors.YELLOW)
        
    return openai_key is not None

def create_env_template():
    """Create a .env.example file"""
    env_template = """# CodebaseIQ Pro Environment Configuration

# REQUIRED: OpenAI API Key for embeddings
OPENAI_API_KEY=your-openai-api-key-here

# OPTIONAL: Premium Features

# Voyage AI for optimized code embeddings (better than OpenAI for code)
# Get your API key from: https://www.voyageai.com/
VOYAGE_API_KEY=

# Pinecone for cloud vector database (faster than local Qdrant)
# Get your API key from: https://www.pinecone.io/
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=us-east-1

# Redis for distributed caching (optional)
# Use local Redis: redis://localhost:6379
# Or cloud Redis: redis://user:password@host:port
REDIS_URL=

# Performance settings (optional)
MAX_WORKERS=10
BATCH_SIZE=100
MAX_FILE_SIZE_MB=10
ANALYSIS_TIMEOUT=300

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
"""
    
    env_file = Path(__file__).parent / ".env.example"
    with open(env_file, 'w') as f:
        f.write(env_template)
        
    print_colored(f"\n✅ Created .env.example file", Colors.GREEN)
    print(f"   Copy to .env and add your API keys:")
    print(f"   cp .env.example .env")

def main():
    """Main setup process"""
    print_colored("🚀 CodebaseIQ Pro Setup", Colors.BLUE)
    print_colored("=" * 40, Colors.BLUE)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
        
    # Check if packages need to be installed
    if not check_required_packages():
        response = input("\n❓ Install missing packages? (y/n): ")
        if response.lower() == 'y':
            if not install_requirements():
                sys.exit(1)
            # Re-check packages
            check_required_packages()
                
    # Check environment variables
    env_ok = check_environment_variables()
    
    # Create .env.example
    create_env_template()
    
    # Final status
    print_colored("\n" + "=" * 40, Colors.BLUE)
    
    if not env_ok:
        print_colored("⚠️  Setup incomplete!", Colors.YELLOW)
        print("\nNext steps:")
        print("1. Copy .env.example to .env")
        print("2. Add your OPENAI_API_KEY (required)")
        print("3. Optionally add premium service keys")
        print("\nThen run: python codebase-iq-pro.py")
    else:
        print_colored("✅ Setup complete!", Colors.GREEN)
        print("\nYou can now run: python codebase-iq-pro.py")
        print("\nOptional: Add premium service keys to .env for enhanced features")

if __name__ == "__main__":
    main()