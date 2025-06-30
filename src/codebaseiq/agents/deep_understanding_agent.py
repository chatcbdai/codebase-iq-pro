#!/usr/bin/env python3
"""
Deep Understanding Agent - Extracts semantic meaning from code
This agent understands WHAT code does and WHY it exists, providing
AI assistants with complete context from the first message.
"""

import ast
import re
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class CodeContext:
    """Represents deep understanding of a code file for AI consumption"""
    file_path: str
    language: str
    purpose: str  # What this file/class/function does
    business_logic: str  # Why it exists in business terms
    dependencies: List[str]  # What it needs to work
    dependents: List[str]  # What needs this to work
    critical_functions: List[Dict[str, Any]]  # Key functions and their purposes
    data_flow: List[str]  # How data moves through this code
    side_effects: List[str]  # External systems it affects
    error_handling: List[str]  # How it handles failures
    security_concerns: List[str]  # Security-sensitive operations
    modification_risk: str  # Risk level of modifying this file
    ai_guidance: str  # Specific guidance for AI assistants
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

class LanguageAnalyzer:
    """Base class for language-specific analyzers"""
    
    def can_analyze(self, file_path: str, content: str) -> bool:
        """Check if this analyzer can handle the file"""
        raise NotImplementedError
        
    def extract_purpose(self, content: str, tree: Any = None) -> str:
        """Extract what this code does"""
        raise NotImplementedError
        
    def extract_dependencies(self, content: str, tree: Any = None) -> List[str]:
        """Extract what this code depends on"""
        raise NotImplementedError
        
    def extract_functions(self, content: str, tree: Any = None) -> List[Dict[str, Any]]:
        """Extract critical functions and their purposes"""
        raise NotImplementedError

class PythonAnalyzer(LanguageAnalyzer):
    """Python-specific code analyzer"""
    
    def can_analyze(self, file_path: str, content: str) -> bool:
        return file_path.endswith('.py')
        
    def extract_purpose(self, content: str, tree: ast.AST = None) -> str:
        """Extract purpose from Python code"""
        purposes = []
        
        if tree:
            # Module-level docstring
            docstring = ast.get_docstring(tree)
            if docstring:
                first_line = docstring.split('\n')[0].strip()
                purposes.append(f"Module: {first_line}")
                
            # Main class or function purposes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node)
                    if class_doc:
                        purposes.append(f"Class {node.name}: {class_doc.split(chr(10))[0]}")
                elif isinstance(node, ast.FunctionDef) and node.name in ['main', '__init__', 'run', 'execute', 'start']:
                    func_doc = ast.get_docstring(node)
                    if func_doc:
                        purposes.append(f"Main function: {func_doc.split(chr(10))[0]}")
                        
        # Extract from comments
        important_comments = re.findall(r'#\s*(PURPOSE|MAIN|TODO|IMPORTANT|NOTE):\s*(.+)', content, re.IGNORECASE)
        for marker, comment in important_comments[:2]:
            purposes.append(f"{marker}: {comment.strip()}")
            
        # If no explicit purpose, infer from structure
        if not purposes:
            if 'test_' in Path(tree.body[0].name if tree and tree.body else '').name:
                purposes.append("Test suite for validating functionality")
            elif 'main' in content:
                purposes.append("Application entry point or main logic")
            elif re.search(r'class\s+\w+Model', content):
                purposes.append("Data model definitions")
            elif re.search(r'@app\.|@router\.', content):
                purposes.append("API endpoint definitions")
                
        return " | ".join(purposes) if purposes else "Purpose needs clarification - add docstrings"
        
    def extract_dependencies(self, content: str, tree: ast.AST = None) -> List[str]:
        """Extract Python dependencies with context"""
        deps = []
        
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        deps.append(f"import:{alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        deps.append(f"from:{module}.{alias.name}")
                        
                # Track external API/service calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            call_str = f"{node.func.value.id}.{node.func.attr}"
                            if any(api in call_str for api in ['requests.', 'http.', 'api.', 'client.']):
                                deps.append(f"api_call:{call_str}")
                                
        return list(set(deps))[:30]  # Top 30 unique dependencies
        
    def extract_functions(self, content: str, tree: ast.AST = None) -> List[Dict[str, Any]]:
        """Extract critical functions with deep understanding"""
        functions = []
        
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip test functions unless they test critical functionality
                    if node.name.startswith('test_') and 'critical' not in node.name.lower():
                        continue
                        
                    # Skip private methods unless important
                    if node.name.startswith('_') and node.name not in ['__init__', '__call__', '__enter__', '__exit__']:
                        continue
                        
                    func_info = self._analyze_function(node, content)
                    if func_info['importance_score'] > 0:
                        functions.append(func_info)
                        
        # Sort by importance
        functions.sort(key=lambda f: f['importance_score'], reverse=True)
        return functions[:15]  # Top 15 most important functions
        
    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Deep analysis of a single function"""
        docstring = ast.get_docstring(node) or "No documentation"
        
        func_info = {
            "name": node.name,
            "purpose": docstring.split('\n')[0],
            "parameters": [arg.arg for arg in node.args.args],
            "calls_external": False,
            "modifies_state": False,
            "handles_errors": False,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "complexity": self._calculate_complexity(node),
            "importance_score": 0,
            "ai_notes": []
        }
        
        # Analyze function body
        for child in ast.walk(node):
            # External calls
            if isinstance(child, ast.Call):
                func_info["calls_external"] = True
                if isinstance(child.func, ast.Attribute) and hasattr(child.func.value, 'id'):
                    if child.func.value.id in ['requests', 'http', 'api']:
                        func_info["ai_notes"].append("Makes external API calls - test with mocks")
                        
            # State modifications
            if isinstance(child, (ast.Assign, ast.AugAssign)):
                func_info["modifies_state"] = True
                
            # Error handling
            if isinstance(child, (ast.Try, ast.ExceptHandler)):
                func_info["handles_errors"] = True
                func_info["ai_notes"].append("Has error handling - preserve exception logic")
                
            # Database operations
            if isinstance(child, ast.Str) and re.search(r'(SELECT|INSERT|UPDATE|DELETE)', child.s, re.IGNORECASE):
                func_info["ai_notes"].append("Contains SQL - validate query changes carefully")
                
        # Calculate importance score
        importance = 0
        if node.name in ['__init__', 'main', 'run', 'execute', 'start']:
            importance += 10
        if func_info["handles_errors"]:
            importance += 5
        if func_info["calls_external"]:
            importance += 3
        if func_info["modifies_state"]:
            importance += 2
        if len(func_info["parameters"]) > 3:
            importance += 2  # Complex interface
        if func_info["is_async"]:
            importance += 2  # Async complexity
            
        func_info["importance_score"] = importance
        
        # Generate AI-specific guidance
        if importance >= 10:
            func_info["ai_notes"].append("CRITICAL FUNCTION - Changes require extensive testing")
        elif importance >= 5:
            func_info["ai_notes"].append("Important function - verify all callers after changes")
            
        return func_info
        
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

class JavaScriptAnalyzer(LanguageAnalyzer):
    """JavaScript analyzer"""
    
    def can_analyze(self, file_path: str, content: str) -> bool:
        return file_path.endswith(('.js', '.jsx'))
        
    def extract_purpose(self, content: str, tree: Any = None) -> str:
        """Extract purpose from JavaScript/TypeScript code"""
        purposes = []
        
        # JSDoc comments
        jsdoc_matches = re.findall(r'/\*\*\s*\n\s*\*\s*([^@\n]+)', content)
        if jsdoc_matches:
            purposes.append(f"Module: {jsdoc_matches[0].strip()}")
            
        # React components
        if re.search(r'export\s+(?:default\s+)?(?:function|const)\s+(\w+).*?(?:React\.FC|Component)', content):
            purposes.append("React component for UI rendering")
            
        # Express routes
        if re.search(r'(?:app|router)\.\w+\([\'"`]([^\'"`]+)', content):
            purposes.append("API endpoint handlers")
            
        # Redux reducers/actions
        if 'reducer' in content.lower() or 'dispatch' in content:
            purposes.append("State management logic")
            
        return " | ".join(purposes) if purposes else "JavaScript/TypeScript module"
        
    def extract_dependencies(self, content: str, tree: Any = None) -> List[str]:
        """Extract JavaScript dependencies"""
        deps = []
        
        # ES6 imports
        import_matches = re.findall(r'import\s+(?:{[^}]+}|\w+)\s+from\s+[\'"`]([^\'"`]+)[\'"`]', content)
        deps.extend([f"import:{imp}" for imp in import_matches])
        
        # CommonJS requires
        require_matches = re.findall(r'require\([\'"`]([^\'"`]+)[\'"`]\)', content)
        deps.extend([f"require:{req}" for req in require_matches])
        
        # API calls
        if 'fetch(' in content or 'axios' in content:
            deps.append("api_call:http_requests")
            
        return list(set(deps))[:30]
        
    def extract_functions(self, content: str, tree: Any = None) -> List[Dict[str, Any]]:
        """Extract JavaScript functions"""
        functions = []
        
        # Function declarations and expressions
        func_patterns = [
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)',
            r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',
            r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?function\s*\([^)]*\)',
        ]
        
        for pattern in func_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                func_name = match.group(1)
                # Get the function body context
                start = match.start()
                context_start = max(0, start - 100)
                context_end = min(len(content), start + 500)
                func_context = content[context_start:context_end]
                
                func_info = {
                    "name": func_name,
                    "purpose": self._infer_js_function_purpose(func_name, func_context),
                    "is_async": 'async' in match.group(0),
                    "is_exported": 'export' in match.group(0),
                    "importance_score": 5 if 'export' in match.group(0) else 3,
                    "ai_notes": []
                }
                
                if func_info["is_exported"]:
                    func_info["ai_notes"].append("Exported function - part of public API")
                if func_info["is_async"]:
                    func_info["ai_notes"].append("Async function - maintain promise handling")
                    
                functions.append(func_info)
                
        return functions[:15]
        
    def _infer_js_function_purpose(self, func_name: str, context: str) -> str:
        """Infer function purpose from name and context"""
        # Common patterns
        if func_name.startswith('handle'):
            return f"Event handler for {func_name[6:]} events"
        elif func_name.startswith('get'):
            return f"Getter for {func_name[3:]} data"
        elif func_name.startswith('set'):
            return f"Setter for {func_name[3:]} data"
        elif func_name.startswith('render'):
            return f"Renders {func_name[6:]} UI component"
        elif 'Component' in func_name:
            return "React component definition"
        elif 'reducer' in func_name.lower():
            return "Redux reducer for state management"
        else:
            return f"Function {func_name}"

class TypeScriptAnalyzer(LanguageAnalyzer):
    """TypeScript analyzer with enhanced type-aware analysis"""
    
    def can_analyze(self, file_path: str, content: str) -> bool:
        return file_path.endswith(('.ts', '.tsx'))
        
    def extract_purpose(self, content: str, tree: Any = None) -> str:
        """Extract purpose from TypeScript code with type awareness"""
        purposes = []
        
        # Check for decorators (common in Angular/NestJS)
        if '@Component' in content or '@Injectable' in content:
            purposes.append("Angular component/service")
        elif '@Controller' in content or '@Module' in content:
            purposes.append("NestJS module/controller")
            
        # Interface/Type definitions
        interface_count = len(re.findall(r'interface\s+\w+', content))
        type_count = len(re.findall(r'type\s+\w+\s*=', content))
        if interface_count + type_count > 3:
            purposes.append(f"Type definitions ({interface_count} interfaces, {type_count} types)")
            
        # Generic TypeScript patterns
        if re.search(r'export\s+(?:default\s+)?(?:function|const)\s+(\w+).*?:\s*React\.FC', content):
            purposes.append("Typed React component")
            
        # Async patterns with types
        if 'Promise<' in content:
            purposes.append("Async operations with typed promises")
            
        return " | ".join(purposes) if purposes else "TypeScript module with strong typing"
        
    def extract_dependencies(self, content: str, tree: Any = None) -> List[str]:
        """Extract dependencies including type imports"""
        deps = []
        
        # Regular imports
        import_matches = re.findall(r'import\s+.*?from\s+[\'"`]([^\'"`]+)', content)
        deps.extend(import_matches)
        
        # Type-only imports
        type_imports = re.findall(r'import\s+type\s+.*?from\s+[\'"`]([^\'"`]+)', content)
        deps.extend([f"{imp} (type)" for imp in type_imports])
        
        return list(set(deps))
        
    def extract_functions(self, content: str, tree: Any = None) -> List[Dict[str, Any]]:
        """Extract functions with type signatures"""
        functions = []
        
        # Function patterns with return types
        func_patterns = [
            # Regular functions with types
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)\s*:\s*([^{]+){',
            # Arrow functions with types
            r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*:\s*([^=]+)=>'
        ]
        
        for pattern in func_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                func_name = match.group(1)
                return_type = match.group(2).strip() if len(match.groups()) > 1 else 'unknown'
                
                functions.append({
                    "name": func_name,
                    "purpose": self._infer_ts_function_purpose(func_name, return_type),
                    "return_type": return_type,
                    "has_generics": '<' in match.group(0),
                    "ai_notes": [f"Typed function with return type: {return_type}"]
                })
                
        return functions[:15]
        
    def _infer_ts_function_purpose(self, func_name: str, return_type: str) -> str:
        """Infer purpose with type information"""
        base_purpose = self._infer_js_function_purpose(func_name, "")
        
        # Enhance with type info
        if 'Observable' in return_type:
            return f"{base_purpose} (reactive stream)"
        elif 'Promise' in return_type:
            return f"{base_purpose} (async operation)"
        elif return_type == 'void':
            return f"{base_purpose} (side effect only)"
        else:
            return base_purpose
            
    def _infer_js_function_purpose(self, func_name: str, context: str) -> str:
        """Reuse JavaScript inference logic"""
        if func_name.startswith('handle'):
            return f"Event handler for {func_name[6:]} events"
        elif func_name.startswith('get'):
            return f"Getter for {func_name[3:]} data"
        else:
            return f"Function {func_name}"

class Web3Analyzer(LanguageAnalyzer):
    """Web3.js analyzer for blockchain/Ethereum development"""
    
    def can_analyze(self, file_path: str, content: str) -> bool:
        # Check for Web3 patterns in JS/TS files
        if file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
            return any(pattern in content for pattern in ['web3', 'Web3', 'ethereum', 'contract', 'wallet'])
        return False
        
    def extract_purpose(self, content: str, tree: Any = None) -> str:
        """Extract Web3-specific purpose"""
        purposes = []
        
        # Smart contract interactions
        if 'abi' in content.lower() and ('contract' in content.lower() or 'Contract' in content):
            purposes.append("Smart contract interface")
            
        # Web3 provider setup
        if 'Web3Provider' in content or 'new Web3(' in content:
            purposes.append("Web3 provider initialization")
            
        # Transaction handling
        if any(method in content for method in ['sendTransaction', 'signTransaction', 'estimateGas']):
            purposes.append("Blockchain transaction handling")
            
        # Wallet operations
        if any(wallet in content for wallet in ['MetaMask', 'WalletConnect', 'wallet.connect']):
            purposes.append("Wallet integration")
            
        # Event listeners
        if '.on(' in content and ('Transfer' in content or 'Approval' in content):
            purposes.append("Blockchain event listener")
            
        # ENS (Ethereum Name Service)
        if 'ens' in content.lower() or 'resolveName' in content:
            purposes.append("ENS name resolution")
            
        # DeFi specific
        if any(defi in content for defi in ['swap', 'liquidity', 'stake', 'yield']):
            purposes.append("DeFi operations")
            
        # Web3 v2.0 specific features (as of June 2025)
        if 'web3.eth.subscribe' in content:
            purposes.append("Web3 v2.0 subscription handling")
        if 'web3.utils.toBigInt' in content:
            purposes.append("Web3 v2.0 BigInt operations")
            
        return " | ".join(purposes) if purposes else "Web3/Blockchain integration"
        
    def extract_dependencies(self, content: str, tree: Any = None) -> List[str]:
        """Extract Web3-related dependencies"""
        deps = []
        
        # Common Web3 packages
        web3_packages = [
            'web3', 'ethers', '@metamask/sdk', '@walletconnect/client',
            'web3-utils', 'web3-eth', 'web3-eth-contract', 'web3-providers',
            '@openzeppelin/contracts', 'hardhat', 'truffle'
        ]
        
        for pkg in web3_packages:
            if pkg in content:
                deps.append(pkg)
                
        # Chain-specific SDKs
        if 'alchemy-sdk' in content:
            deps.append('alchemy-sdk (RPC provider)')
        if 'moralis' in content:
            deps.append('moralis (Web3 data)')
            
        return list(set(deps))
        
    def extract_functions(self, content: str, tree: Any = None) -> List[Dict[str, Any]]:
        """Extract Web3-specific functions"""
        functions = []
        
        # Contract method patterns
        contract_methods = re.findall(r'contract\.methods\.(\w+)\(', content)
        for method in set(contract_methods):
            functions.append({
                "name": f"contract.{method}",
                "purpose": f"Smart contract method: {method}",
                "is_blockchain": True,
                "ai_notes": ["Blockchain transaction - gas fees apply", "Requires user wallet approval"]
            })
            
        # Web3 utility functions
        web3_patterns = [
            (r'async\s+function\s+(\w*[Cc]onnect\w*)', "Wallet connection"),
            (r'async\s+function\s+(\w*[Ss]ign\w*)', "Transaction signing"),
            (r'async\s+function\s+(\w*[Tt]ransfer\w*)', "Token/ETH transfer"),
            (r'function\s+(\w*[Bb]alance\w*)', "Balance checking"),
        ]
        
        for pattern, purpose in web3_patterns:
            matches = re.findall(pattern, content)
            for func_name in matches:
                functions.append({
                    "name": func_name,
                    "purpose": purpose,
                    "is_blockchain": True,
                    "ai_notes": ["Web3 operation - handle errors gracefully", "May require network requests"]
                })
                
        return functions[:15]

class DeepUnderstandingAgent:
    """
    Extracts semantic meaning from code beyond just structure.
    Provides AI assistants with complete understanding from conversation start.
    """
    
    def __init__(self):
        self.contexts: Dict[str, CodeContext] = {}
        self.analyzers = [
            PythonAnalyzer(),
            JavaScriptAnalyzer(),
            TypeScriptAnalyzer(),
            Web3Analyzer(),
        ]
        self.codebase_summary = {}
        
    def analyze_file(self, file_path: str, content: str) -> CodeContext:
        """
        Deeply analyze a single file to understand its purpose and meaning.
        This provides AI with immediate context about what can and cannot be changed.
        """
        logger.info(f"🧠 Deep analysis of {file_path}")
        
        # Determine language and get appropriate analyzer
        analyzer = self._get_analyzer(file_path, content)
        language = self._detect_language(file_path)
        
        # Parse the code if possible
        tree = None
        if language == 'python':
            try:
                tree = ast.parse(content)
            except:
                logger.warning(f"Could not parse {file_path}, analyzing as text")
                
        # Extract all semantic information
        context = CodeContext(
            file_path=file_path,
            language=language,
            purpose=analyzer.extract_purpose(content, tree) if analyzer else "Unknown purpose",
            business_logic=self._extract_business_logic(content, tree),
            dependencies=analyzer.extract_dependencies(content, tree) if analyzer else [],
            dependents=[],  # Will be filled by cross-file analysis
            critical_functions=analyzer.extract_functions(content, tree) if analyzer else [],
            data_flow=self._extract_data_flow(content, tree),
            side_effects=self._extract_side_effects(content),
            error_handling=self._extract_error_handling(content, tree),
            security_concerns=self._extract_security_concerns(content),
            modification_risk=self._assess_modification_risk(file_path, content),
            ai_guidance=self._generate_file_specific_ai_guidance(file_path, content)
        )
        
        self.contexts[file_path] = context
        return context
        
    def _get_analyzer(self, file_path: str, content: str) -> Optional[LanguageAnalyzer]:
        """Get the appropriate language analyzer"""
        for analyzer in self.analyzers:
            if analyzer.can_analyze(file_path, content):
                return analyzer
        return None
        
    def _detect_language(self, file_path: str) -> str:
        """Detect the programming language"""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'objc',
            '.sql': 'sql',
            '.sh': 'bash',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.md': 'markdown',
        }
        return language_map.get(ext, 'unknown')
        
    def _extract_business_logic(self, content: str, tree: Any = None) -> str:
        """Extract WHY this code exists in business terms"""
        business_indicators = []
        
        # Business domain keywords
        business_domains = {
            'authentication': ['login', 'auth', 'session', 'token', 'password', 'user', 'permission'],
            'payment': ['payment', 'charge', 'refund', 'invoice', 'billing', 'subscription', 'price'],
            'user_management': ['user', 'profile', 'account', 'registration', 'member', 'customer'],
            'order_processing': ['order', 'cart', 'checkout', 'shipping', 'delivery', 'product'],
            'communication': ['email', 'notification', 'message', 'alert', 'sms', 'push'],
            'analytics': ['analytics', 'metrics', 'report', 'dashboard', 'statistics', 'tracking'],
            'data_processing': ['process', 'transform', 'aggregate', 'calculate', 'analyze'],
            'integration': ['api', 'webhook', 'sync', 'export', 'import', 'integration'],
            'content_management': ['content', 'article', 'post', 'media', 'document', 'file'],
            'search': ['search', 'filter', 'query', 'find', 'discover', 'recommend'],
        }
        
        content_lower = content.lower()
        
        # Check for business domain matches
        for domain, keywords in business_domains.items():
            if any(keyword in content_lower for keyword in keywords):
                # Count keyword occurrences to determine primary domain
                keyword_count = sum(content_lower.count(keyword) for keyword in keywords)
                if keyword_count > 2:  # Significant presence
                    business_indicators.append(f"Primary: {domain.replace('_', ' ').title()} logic")
                    break
                    
        # API endpoint detection
        api_patterns = [
            (r'@(?:app|router)\.\w+\(["\']([^"\']+)["\']', 'REST API'),
            (r'(?:get|post|put|delete|patch)\s*\(["\']([^"\']+)["\']', 'HTTP endpoint'),
            (r'type\s+\w+\s+{\s*[^}]*}\s*#\s*GraphQL', 'GraphQL schema'),
        ]
        
        for pattern, api_type in api_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                business_indicators.append(f"{api_type}: {len(matches)} endpoints")
                
        # Database operations
        db_operations = {
            'read': ['SELECT', 'FIND', 'GET', 'FETCH', 'QUERY'],
            'write': ['INSERT', 'UPDATE', 'CREATE', 'SAVE', 'PUT'],
            'delete': ['DELETE', 'REMOVE', 'DROP', 'DESTROY'],
        }
        
        for op_type, keywords in db_operations.items():
            if any(re.search(rf'\b{keyword}\b', content, re.IGNORECASE) for keyword in keywords):
                business_indicators.append(f"Database {op_type} operations")
                
        # Business rules detection
        rule_indicators = [
            (r'if.*age.*[<>]=?\s*\d+', 'Age verification rules'),
            (r'if.*price.*[<>]=?\s*\d+', 'Pricing rules'),
            (r'if.*balance.*[<>]=?\s*\d+', 'Balance checks'),
            (r'(?:validate|check|verify|ensure)', 'Validation logic'),
            (r'(?:calculate|compute)', 'Calculation logic'),
        ]
        
        for pattern, rule_type in rule_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                business_indicators.append(rule_type)
                
        return " | ".join(business_indicators) if business_indicators else "Technical implementation - business context needed"
        
    def _extract_data_flow(self, content: str, tree: Any = None) -> List[str]:
        """Trace how data flows through the code"""
        flows = []
        
        # Common data flow patterns
        flow_patterns = [
            (r'def\s+(\w+).*?return\s+', 'Transform and return'),
            (r'async\s+def\s+(\w+).*?await\s+', 'Async data processing'),
            (r'(\w+)\s*=\s*(?:fetch|request|query)', 'Fetch and store'),
            (r'for\s+\w+\s+in\s+(\w+):', 'Iterate and process'),
            (r'map\(|filter\(|reduce\(', 'Functional transformation'),
        ]
        
        for pattern, flow_type in flow_patterns:
            if re.search(pattern, content):
                flows.append(flow_type)
                
        # Input/Output analysis
        if tree and hasattr(tree, 'body'):
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    has_params = len(node.args.args) > 0
                    has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
                    
                    if has_params and has_return:
                        flows.append(f"{node.name}: Input → Process → Output")
                    elif has_params:
                        flows.append(f"{node.name}: Input → Side effects")
                    elif has_return:
                        flows.append(f"{node.name}: Generate → Output")
                        
        return flows[:10]
        
    def _extract_side_effects(self, content: str) -> List[str]:
        """Identify external systems this code affects"""
        effects = set()
        
        # Comprehensive side effect patterns
        effect_patterns = {
            "File I/O": ['open(', 'write(', 'read(', 'file', 'fs.', 'createWriteStream', 'createReadStream'],
            "Network calls": ['requests.', 'urllib', 'http', 'fetch(', 'axios', 'ajax', 'websocket'],
            "Database operations": ['execute(', 'query(', '.save()', '.create()', '.update()', '.delete()', 'INSERT', 'UPDATE', 'DELETE'],
            "Process execution": ['subprocess', 'exec(', 'spawn(', 'system(', 'shell'],
            "Environment access": ['os.environ', 'process.env', 'getenv'],
            "Logging": ['logger.', 'logging.', 'console.log', 'print('],
            "Cache operations": ['cache.', 'redis.', 'memcache', 'localStorage', 'sessionStorage'],
            "Message queues": ['publish(', 'subscribe(', 'rabbitmq', 'kafka', 'sqs', 'pubsub'],
            "Email sending": ['send_email', 'send_mail', 'smtp', 'ses.', 'sendgrid'],
            "Authentication": ['jwt', 'oauth', 'auth0', 'passport'],
            "Payment processing": ['stripe', 'paypal', 'braintree', 'square'],
            "Cloud services": ['aws', 's3', 'gcs', 'azure', 'cloudinary'],
        }
        
        for effect_type, patterns in effect_patterns.items():
            if any(pattern in content.lower() for pattern in patterns):
                effects.add(effect_type)
                
        return sorted(list(effects))
        
    def _extract_error_handling(self, content: str, tree: Any = None) -> List[str]:
        """Identify how this code handles errors"""
        error_patterns = []
        
        # Language-agnostic error patterns
        error_keywords = [
            (r'try\s*{|try:', 'Try-catch blocks'),
            (r'catch\s*\(|except\s+\w+:', 'Exception handling'),
            (r'finally\s*{|finally:', 'Cleanup blocks'),
            (r'throw\s+|raise\s+', 'Error throwing'),
            (r'\.catch\(|\.error\(', 'Promise error handling'),
            (r'on\s*\(\s*["\']error["\']\s*,', 'Event error handling'),
            (r'if\s*\(\s*(?:err|error)\s*\)', 'Error checking'),
        ]
        
        for pattern, error_type in error_keywords:
            if re.search(pattern, content):
                error_patterns.append(error_type)
                
        # Specific error types
        if tree and hasattr(ast, 'ExceptHandler'):
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    if node.type:
                        if isinstance(node.type, ast.Name):
                            error_patterns.append(f"Handles: {node.type.id}")
                        elif isinstance(node.type, ast.Tuple):
                            exceptions = [e.id for e in node.type.elts if isinstance(e, ast.Name)]
                            error_patterns.append(f"Handles: {', '.join(exceptions)}")
                    else:
                        error_patterns.append("Handles: all exceptions")
                        
        return list(set(error_patterns))
        
    def _extract_security_concerns(self, content: str) -> List[str]:
        """Identify security-sensitive operations"""
        concerns = []
        
        # Security patterns to detect
        security_patterns = [
            (r'password|passwd|pwd', 'Password handling'),
            (r'secret|private_key|api_key', 'Secret/API key handling'),
            (r'token|jwt|bearer', 'Authentication tokens'),
            (r'encrypt|decrypt|hash|bcrypt|crypto', 'Cryptographic operations'),
            (r'sql.*where.*=\s*["\']?\s*\+|f["\'].*{.*}.*(?:WHERE|where)', 'Potential SQL injection'),
            (r'eval\(|exec\(|compile\(', 'Dynamic code execution'),
            (r'<script|innerHTML|dangerouslySetInnerHTML', 'Potential XSS vulnerability'),
            (r'sudo|root|admin|privilege', 'Privilege escalation'),
            (r'cors|origin|csrf', 'Cross-origin security'),
            (r'sanitize|escape|validate', 'Input validation'),
        ]
        
        for pattern, concern_type in security_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                concerns.append(concern_type)
                
        return list(set(concerns))
        
    def _assess_modification_risk(self, file_path: str, content: str) -> str:
        """Assess the risk level of modifying this file"""
        risk_score = 0
        risk_factors = []
        
        # File name indicators
        critical_files = ['auth', 'security', 'payment', 'config', 'settings', 'database', 'migration']
        for critical in critical_files:
            if critical in file_path.lower():
                risk_score += 3
                risk_factors.append(f"Critical file type: {critical}")
                
        # Content indicators
        if len(self._extract_security_concerns(content)) > 2:
            risk_score += 4
            risk_factors.append("Multiple security concerns")
            
        if re.search(r'(?:production|prod).*(?:config|settings|env)', content, re.IGNORECASE):
            risk_score += 5
            risk_factors.append("Production configuration")
            
        if re.search(r'CREATE TABLE|ALTER TABLE|DROP TABLE', content):
            risk_score += 4
            risk_factors.append("Database schema changes")
            
        # Determine risk level
        if risk_score >= 8:
            return f"CRITICAL - {', '.join(risk_factors[:2])}"
        elif risk_score >= 5:
            return f"HIGH - {', '.join(risk_factors[:2])}"
        elif risk_score >= 3:
            return f"MEDIUM - {', '.join(risk_factors[:1])}"
        else:
            return "LOW - Safe to modify with standard precautions"
            
    def _generate_file_specific_ai_guidance(self, file_path: str, content: str) -> str:
        """Generate specific guidance for AI when modifying this file"""
        guidance_parts = []
        
        # Check for specific patterns that need careful handling
        if 'test' in file_path.lower():
            guidance_parts.append("Test file - ensure changes maintain test coverage")
            
        if re.search(r'@deprecated|DEPRECATED|TODO.*remove', content, re.IGNORECASE):
            guidance_parts.append("Contains deprecated code - consider removal rather than modification")
            
        if len(content.split('\n')) > 500:
            guidance_parts.append("Large file - consider breaking into smaller modules")
            
        if re.search(r'(?:async|await|Promise|then\()', content):
            guidance_parts.append("Async code - maintain proper error handling and avoid race conditions")
            
        if re.search(r'(?:BEGIN|COMMIT|ROLLBACK)', content):
            guidance_parts.append("Transaction handling - ensure ACID properties are maintained")
            
        # Add language-specific guidance
        if file_path.endswith('.py'):
            if 'class' in content:
                guidance_parts.append("Python classes - maintain inheritance hierarchy and method signatures")
        elif file_path.endswith(('.js', '.ts')):
            if 'export' in content:
                guidance_parts.append("Exported module - changes affect all importers")
                
        return " | ".join(guidance_parts) if guidance_parts else "Standard modification guidelines apply"
        
    def generate_understanding_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary for AI assistants"""
        # Group files by risk level
        risk_groups = defaultdict(list)
        for path, context in self.contexts.items():
            risk_level = context.modification_risk.split(' - ')[0]
            risk_groups[risk_level].append(path)
            
        # Identify entry points
        entry_points = []
        for path, context in self.contexts.items():
            if any(func['name'] in ['main', 'start', 'run', 'index'] for func in context.critical_functions):
                entry_points.append(path)
                
        # Find most connected files (will be updated by cross-file analysis)
        critical_files = [
            path for path, context in self.contexts.items()
            if 'CRITICAL' in context.modification_risk or 'HIGH' in context.modification_risk
        ]
        
        return {
            "total_files_analyzed": len(self.contexts),
            "languages_found": list(set(ctx.language for ctx in self.contexts.values())),
            "file_purposes": {
                path: context.purpose 
                for path, context in self.contexts.items()
            },
            "business_logic_map": {
                path: context.business_logic 
                for path, context in self.contexts.items()
                if "Technical implementation" not in context.business_logic
            },
            "risk_assessment": dict(risk_groups),
            "entry_points": entry_points,
            "critical_files": critical_files,
            "security_sensitive_files": [
                path for path, context in self.contexts.items()
                if context.security_concerns
            ],
            "external_integrations": self._summarize_external_integrations(),
            "immediate_ai_guidance": self._generate_codebase_level_guidance()
        }
        
    def _summarize_external_integrations(self) -> Dict[str, List[str]]:
        """Summarize all external system integrations"""
        integrations = defaultdict(set)
        
        for path, context in self.contexts.items():
            for effect in context.side_effects:
                integrations[effect].add(path)
                
        return {k: list(v) for k, v in integrations.items()}
        
    def _generate_codebase_level_guidance(self) -> str:
        """Generate immediate guidance for AI assistants"""
        total_files = len(self.contexts)
        critical_count = sum(1 for ctx in self.contexts.values() if 'CRITICAL' in ctx.modification_risk)
        security_count = sum(1 for ctx in self.contexts.values() if ctx.security_concerns)
        
        guidance = [
            f"📊 Codebase Analysis Complete: {total_files} files analyzed",
            f"⚠️  Critical files requiring extreme caution: {critical_count}",
            f"🔒 Security-sensitive files: {security_count}",
            "",
            "🤖 BEFORE MAKING ANY CHANGES:",
            "1. Check the file's modification_risk level",
            "2. Review critical_functions to understand impact",
            "3. Examine dependencies and dependents (from cross-file analysis)",
            "4. Follow the ai_guidance for each file",
            "5. Consider security_concerns before modifying",
            "",
            "💡 KEY INSIGHT: This codebase appears to be a " + self._infer_codebase_type(),
        ]
        
        return "\n".join(guidance)
        
    def _infer_codebase_type(self) -> str:
        """Infer the type of codebase from analysis"""
        # Count indicators
        indicators = defaultdict(int)
        
        for context in self.contexts.values():
            if 'REST API' in context.business_logic:
                indicators['api'] += 1
            if 'React component' in context.purpose:
                indicators['frontend'] += 1
            if 'Database' in context.business_logic:
                indicators['backend'] += 1
            if 'test' in context.file_path.lower():
                indicators['testing'] += 1
                
        # Determine primary type
        if indicators['api'] > 5:
            return "REST API service with database backend"
        elif indicators['frontend'] > 5:
            return "Frontend application (likely React/Vue/Angular)"
        elif indicators['backend'] > indicators['frontend']:
            return "Backend service application"
        else:
            return "Full-stack application"