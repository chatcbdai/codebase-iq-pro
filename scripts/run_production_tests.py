#!/usr/bin/env python3
"""
Production Test Runner for CodebaseIQ Pro MCP Server

Runs all critical tests and verifies the server is production-ready.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class ProductionTestRunner:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'critical_failures': [],
            'warnings': [],
            'production_ready': False
        }
        
    def print_header(self, text, color=BLUE):
        """Print a formatted header"""
        print(f"\n{color}{BOLD}{'=' * 60}{RESET}")
        print(f"{color}{BOLD}{text.center(60)}{RESET}")
        print(f"{color}{BOLD}{'=' * 60}{RESET}\n")
        
    def print_test(self, name, status, message=""):
        """Print test result"""
        if status == "PASS":
            print(f"  {GREEN}✓{RESET} {name} {GREEN}[PASS]{RESET}")
        elif status == "FAIL":
            print(f"  {RED}✗{RESET} {name} {RED}[FAIL]{RESET}")
            if message:
                print(f"    {RED}→ {message}{RESET}")
        elif status == "WARN":
            print(f"  {YELLOW}⚠{RESET} {name} {YELLOW}[WARN]{RESET}")
            if message:
                print(f"    {YELLOW}→ {message}{RESET}")
        elif status == "SKIP":
            print(f"  {YELLOW}-{RESET} {name} {YELLOW}[SKIP]{RESET}")
            
    def run_test_file(self, test_file, test_name, critical=True):
        """Run a specific test file"""
        print(f"\n{BOLD}Running {test_name}...{RESET}")
        
        try:
            # Run the test with pytest
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True
            )
            
            self.results['tests_run'] += 1
            
            if result.returncode == 0:
                self.results['tests_passed'] += 1
                self.print_test(test_name, "PASS")
                return True
            else:
                self.results['tests_failed'] += 1
                error_msg = self._extract_error(result.stdout + result.stderr)
                self.print_test(test_name, "FAIL", error_msg)
                
                if critical:
                    self.results['critical_failures'].append({
                        'test': test_name,
                        'error': error_msg
                    })
                return False
                
        except Exception as e:
            self.results['tests_failed'] += 1
            self.print_test(test_name, "FAIL", str(e))
            if critical:
                self.results['critical_failures'].append({
                    'test': test_name,
                    'error': str(e)
                })
            return False
            
    def _extract_error(self, output):
        """Extract meaningful error from pytest output"""
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if 'FAILED' in line or 'ERROR' in line:
                # Get the next few lines for context
                return ' '.join(lines[i:i+3])
        return "Test failed (check output for details)"
        
    def check_dependencies(self):
        """Check all required dependencies"""
        print(f"\n{BOLD}Checking Dependencies...{RESET}")
        
        dependencies = {
            'mcp': 'MCP package',
            'pytest': 'Testing framework',
            'openai': 'OpenAI API',
            'qdrant_client': 'Vector database',
            'tiktoken': 'Token counting'
        }
        
        all_good = True
        for module, name in dependencies.items():
            try:
                __import__(module)
                self.print_test(f"{name} ({module})", "PASS")
            except ImportError:
                self.print_test(f"{name} ({module})", "FAIL", "Not installed")
                all_good = False
                self.results['critical_failures'].append({
                    'test': 'Dependencies',
                    'error': f'{module} not installed'
                })
                
        return all_good
        
    def check_environment(self):
        """Check environment configuration"""
        print(f"\n{BOLD}Checking Environment...{RESET}")
        
        # Check for .env file
        env_file = Path('.env')
        if env_file.exists():
            self.print_test(".env file", "PASS")
        else:
            self.print_test(".env file", "WARN", "Not found - using defaults")
            self.results['warnings'].append("No .env file found")
            
        # Check critical environment variables
        critical_vars = ['OPENAI_API_KEY']
        for var in critical_vars:
            if os.getenv(var):
                self.print_test(f"{var}", "PASS", "Set")
            else:
                self.print_test(f"{var}", "FAIL", "Not set")
                self.results['critical_failures'].append({
                    'test': 'Environment',
                    'error': f'{var} not set'
                })
                
        # Check optional environment variables
        optional_vars = ['VOYAGE_API_KEY', 'PINECONE_API_KEY', 'REDIS_URL']
        for var in optional_vars:
            if os.getenv(var):
                self.print_test(f"{var}", "PASS", "Set (Premium)")
            else:
                self.print_test(f"{var}", "SKIP", "Not set (Using free tier)")
                
    def run_critical_tests(self):
        """Run all critical tests"""
        self.print_header("CRITICAL TESTS", BLUE)
        
        critical_tests = [
            ("tests/test_mcp_protocol_compliance.py", "MCP Protocol Compliance"),
            ("tests/test_mcp_contracts.py", "MCP Contract Validation"),
            ("tests/test_mcp_e2e.py", "End-to-End MCP Client Tests"),
        ]
        
        all_passed = True
        for test_file, test_name in critical_tests:
            if Path(test_file).exists():
                passed = self.run_test_file(test_file, test_name, critical=True)
                all_passed = all_passed and passed
            else:
                self.print_test(test_name, "SKIP", f"File not found: {test_file}")
                self.results['warnings'].append(f"Test file missing: {test_file}")
                
        return all_passed
        
    def run_integration_tests(self):
        """Run integration tests"""
        self.print_header("INTEGRATION TESTS", BLUE)
        
        integration_tests = [
            ("tests/test_integration.py", "Basic Integration"),
            ("tests/test_full_integration.py", "Full Integration"),
            ("tests/test_persistence.py", "State Persistence"),
        ]
        
        for test_file, test_name in integration_tests:
            if Path(test_file).exists():
                self.run_test_file(test_file, test_name, critical=False)
            else:
                self.print_test(test_name, "SKIP", f"File not found: {test_file}")
                
    def check_server_startup(self):
        """Test that server can start"""
        print(f"\n{BOLD}Testing Server Startup...{RESET}")
        
        try:
            # Import and create server instance
            from src.codebaseiq.server import CodebaseIQProServer
            server = CodebaseIQProServer()
            self.print_test("Server instantiation", "PASS")
            
            # Test that handlers are accessible
            if hasattr(server, 'handle_call_tool') and hasattr(server, 'handle_list_tools'):
                self.print_test("Handler methods available", "PASS")
            else:
                self.print_test("Handler methods available", "FAIL", "Methods not exposed")
                self.results['critical_failures'].append({
                    'test': 'Server startup',
                    'error': 'Handler methods not accessible'
                })
                
        except Exception as e:
            self.print_test("Server instantiation", "FAIL", str(e))
            self.results['critical_failures'].append({
                'test': 'Server startup',
                'error': str(e)
            })
            
    def generate_report(self):
        """Generate final test report"""
        self.print_header("TEST REPORT", BLUE)
        
        # Summary statistics
        print(f"{BOLD}Test Summary:{RESET}")
        print(f"  Total Tests Run: {self.results['tests_run']}")
        print(f"  Passed: {GREEN}{self.results['tests_passed']}{RESET}")
        print(f"  Failed: {RED}{self.results['tests_failed']}{RESET}")
        print(f"  Critical Failures: {RED}{len(self.results['critical_failures'])}{RESET}")
        print(f"  Warnings: {YELLOW}{len(self.results['warnings'])}{RESET}")
        
        # Critical failures
        if self.results['critical_failures']:
            print(f"\n{RED}{BOLD}CRITICAL FAILURES:{RESET}")
            for failure in self.results['critical_failures']:
                print(f"  {RED}✗ {failure['test']}: {failure['error']}{RESET}")
                
        # Warnings
        if self.results['warnings']:
            print(f"\n{YELLOW}{BOLD}WARNINGS:{RESET}")
            for warning in self.results['warnings']:
                print(f"  {YELLOW}⚠ {warning}{RESET}")
                
        # Production readiness
        self.results['production_ready'] = (
            self.results['tests_failed'] == 0 and
            len(self.results['critical_failures']) == 0
        )
        
        print(f"\n{BOLD}Production Readiness:{RESET}")
        if self.results['production_ready']:
            print(f"  {GREEN}{BOLD}✓ SERVER IS PRODUCTION READY{RESET}")
        else:
            print(f"  {RED}{BOLD}✗ SERVER IS NOT PRODUCTION READY{RESET}")
            print(f"  {RED}Fix all critical failures before deploying to production{RESET}")
            
        # Save report
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n{BLUE}Report saved to: {report_file}{RESET}")
        
    def run(self):
        """Run all production tests"""
        self.print_header("CODEBASEIQ PRO PRODUCTION TEST SUITE", GREEN)
        print("Running comprehensive tests to verify production readiness...\n")
        
        # Check dependencies
        if not self.check_dependencies():
            print(f"\n{RED}Cannot continue without required dependencies{RESET}")
            return
            
        # Check environment
        self.check_environment()
        
        # Test server startup
        self.check_server_startup()
        
        # Run critical tests
        self.run_critical_tests()
        
        # Run integration tests
        self.run_integration_tests()
        
        # Generate report
        self.generate_report()
        
        # Return success/failure
        return self.results['production_ready']


if __name__ == "__main__":
    runner = ProductionTestRunner()
    success = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)