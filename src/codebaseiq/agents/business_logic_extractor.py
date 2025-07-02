#!/usr/bin/env python3
"""
Business Logic Extractor
Translates technical code into business terms for immediate AI understanding.
Provides the "why" behind code, not just the "what".
"""

import re
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import logging
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)

class BusinessLogicExtractor:
    """
    Extracts and explains the business purpose of code.
    This helps AI understand not just WHAT the code does, but WHY it exists.
    Provides immediate business context at conversation startup.
    """
    
    def __init__(self):
        self.business_flows = []
        self.domain_model = {}
        self.user_journeys = []
        self.business_rules = []
        self.compliance_indicators = set()
        self.key_features = set()
        
    def extract_business_logic(self, file_contexts: Dict[str, Any], 
                              cross_file_intel: Dict[str, Any],
                              file_contents: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Extract high-level business logic from technical code.
        This gives AI immediate understanding of the application's purpose.
        """
        logger.info("💼 Extracting business logic for AI understanding...")
        
        # Step 1: Build domain model - what business entities exist
        self.domain_model = self._extract_domain_model(file_contexts, file_contents)
        
        # Step 2: Map user journeys - how users interact with the system
        self.user_journeys = self._map_user_journeys(file_contexts, cross_file_intel, file_contents)
        
        # Step 3: Extract business rules - constraints and logic
        self.business_rules = self._extract_business_rules(file_contexts, file_contents)
        
        # Step 4: Map business processes and workflows
        self.business_flows = self._map_business_flows(file_contexts, cross_file_intel)
        
        # Step 5: Identify key features
        self.key_features = self._identify_key_features(file_contexts, file_contents)
        
        # Step 6: Detect compliance requirements
        self.compliance_indicators = self._identify_compliance_requirements(file_contexts, file_contents)
        
        # Step 7: Generate executive summary for quick understanding
        executive_summary = self._generate_executive_summary()
        
        # Step 8: Create AI-friendly business context
        ai_business_context = self._create_ai_business_context()
        
        return {
            "executive_summary": executive_summary,
            "ai_business_context": ai_business_context,
            "domain_model": self.domain_model,
            "user_journeys": self.user_journeys,
            "business_rules": self.business_rules,
            "business_flows": self.business_flows,
            "key_features": list(self.key_features),
            "compliance_requirements": list(self.compliance_indicators),
            "business_glossary": self._generate_business_glossary(),
            "immediate_context": self._generate_immediate_context()
        }
        
    def _extract_domain_model(self, file_contexts: Dict[str, Any], 
                             file_contents: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Extract business entities and their relationships"""
        
        entities = {}
        relationships = []
        
        # Common business entity patterns
        entity_patterns = [
            # Class definitions
            (r'class\s+(\w*(?:User|Customer|Client|Person|Member|Employee|Staff|Admin)\w*)', 'user_management'),
            (r'class\s+(\w*(?:Product|Item|Article|Service|Offering|SKU)\w*)', 'inventory'),
            (r'class\s+(\w*(?:Order|Purchase|Transaction|Sale|Invoice|Receipt)\w*)', 'transactions'),
            (r'class\s+(\w*(?:Payment|Billing|Subscription|Plan|Price)\w*)', 'financial'),
            (r'class\s+(\w*(?:Account|Profile|Settings|Preferences)\w*)', 'account_management'),
            (r'class\s+(\w*(?:Company|Organization|Team|Department|Group)\w*)', 'organizational'),
            (r'class\s+(\w*(?:Project|Task|Ticket|Issue|Request)\w*)', 'workflow'),
            (r'class\s+(\w*(?:Report|Analytics|Metric|Dashboard)\w*)', 'analytics'),
            (r'class\s+(\w*(?:Message|Notification|Email|Alert|Comment)\w*)', 'communication'),
            (r'class\s+(\w*(?:Document|File|Media|Asset|Content)\w*)', 'content'),
            
            # Database tables
            (r'(?:CREATE TABLE|Table\()\s*["\']?(\w+)["\']?', 'database_entity'),
            
            # Model definitions (various frameworks)
            (r'(?:Model|Schema|Entity)\s*=.*?["\'](\w+)["\']', 'data_model'),
        ]
        
        # Analyze each file
        for file_path, context in file_contexts.items():
            if file_contents and file_path in file_contents:
                content = file_contents[file_path]
                
                # Extract entities
                for pattern, category in entity_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        entity_name = match.strip()
                        if entity_name and len(entity_name) > 2:
                            if entity_name not in entities:
                                entities[entity_name] = {
                                    "name": entity_name,
                                    "category": category,
                                    "files": [],
                                    "operations": [],
                                    "relationships": [],
                                    "attributes": [],
                                    "business_purpose": self._infer_entity_purpose(entity_name, category)
                                }
                            entities[entity_name]["files"].append(file_path)
                            
                # Extract entity operations (CRUD and business operations)
                operation_patterns = [
                    (r'def\s+(\w+).*?(?:' + '|'.join(entities.keys()) + ')', 'method'),
                    (r'(?:GET|POST|PUT|DELETE|PATCH)\s+["\']?/\w*/?(\w+)', 'api_endpoint'),
                    (r'(?:create|add|new|insert)[\s_]?(\w+)', 'create'),
                    (r'(?:get|find|fetch|retrieve|load|read)[\s_]?(\w+)', 'read'),
                    (r'(?:update|edit|modify|change|set)[\s_]?(\w+)', 'update'),
                    (r'(?:delete|remove|destroy|drop)[\s_]?(\w+)', 'delete'),
                    (r'(?:validate|verify|check)[\s_]?(\w+)', 'validation'),
                    (r'(?:process|handle|execute)[\s_]?(\w+)', 'business_logic'),
                ]
                
                for pattern, op_type in operation_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        op_name = match.group(0)
                        # Match operation to entity
                        for entity_name in entities:
                            if entity_name.lower() in op_name.lower():
                                entities[entity_name]["operations"].append({
                                    "name": op_name,
                                    "type": op_type,
                                    "file": file_path
                                })
                                
                # Extract relationships between entities
                relationship_patterns = [
                    (r'(\w+)\.(\w+)_id\s*=', 'belongs_to'),
                    (r'ForeignKey\(["\'](\w+)', 'references'),
                    (r'OneToMany.*?["\'](\w+)', 'has_many'),
                    (r'ManyToOne.*?["\'](\w+)', 'belongs_to'),
                    (r'ManyToMany.*?["\'](\w+)', 'many_to_many'),
                    (r'(\w+)\s*=\s*relationship\(["\'](\w+)', 'related_to'),
                ]
                
                for pattern, rel_type in relationship_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if len(match.groups()) >= 2:
                            from_entity = match.group(1)
                            to_entity = match.group(2)
                        else:
                            to_entity = match.group(1)
                            from_entity = self._infer_current_entity(content, match.start())
                            
                        if from_entity and to_entity:
                            relationships.append({
                                "from": from_entity,
                                "to": to_entity,
                                "type": rel_type,
                                "file": file_path
                            })
                            
        # Enrich entities with business context
        for entity_name, entity_data in entities.items():
            entity_data["importance"] = self._calculate_entity_importance(entity_data)
            entity_data["crud_operations"] = self._summarize_crud_operations(entity_data["operations"])
            
        return {
            "entities": entities,
            "relationships": relationships,
            "entity_summary": self._generate_entity_summary(entities),
            "relationship_graph": self._build_relationship_graph(entities, relationships)
        }
        
    def _infer_entity_purpose(self, entity_name: str, category: str) -> str:
        """Infer the business purpose of an entity"""
        
        purpose_map = {
            'user_management': "Manages system users and their authentication/authorization",
            'inventory': "Represents products or services offered by the business",
            'transactions': "Handles business transactions and order processing",
            'financial': "Manages financial aspects including payments and billing",
            'account_management': "User account settings and preferences",
            'organizational': "Organizational structure and hierarchy",
            'workflow': "Business process and task management",
            'analytics': "Business intelligence and reporting",
            'communication': "Internal and external communication management",
            'content': "Content and document management",
            'database_entity': "Core data storage entity"
        }
        
        base_purpose = purpose_map.get(category, "Business entity")
        
        # Add specific context based on entity name
        if 'admin' in entity_name.lower():
            base_purpose += " with administrative privileges"
        elif 'customer' in entity_name.lower():
            base_purpose += " for external customers"
        elif 'internal' in entity_name.lower():
            base_purpose += " for internal use"
            
        return base_purpose
        
    def _infer_current_entity(self, content: str, position: int) -> Optional[str]:
        """Infer the current entity context from code position"""
        # Look backwards for class definition
        before_position = content[:position]
        class_matches = list(re.finditer(r'class\s+(\w+)', before_position))
        if class_matches:
            return class_matches[-1].group(1)
        return None
        
    def _calculate_entity_importance(self, entity_data: Dict[str, Any]) -> int:
        """Calculate importance score for an entity"""
        score = 0
        score += len(entity_data["files"]) * 10  # Files using it
        score += len(entity_data["operations"]) * 5  # Operations on it
        score += len(entity_data["relationships"]) * 8  # Relationships
        
        # Core entities get bonus
        if any(core in entity_data["name"].lower() for core in ['user', 'order', 'product', 'payment']):
            score += 20
            
        return score
        
    def _summarize_crud_operations(self, operations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize CRUD operations for an entity"""
        crud_count = defaultdict(int)
        for op in operations:
            if op["type"] in ["create", "read", "update", "delete"]:
                crud_count[op["type"]] += 1
        return dict(crud_count)
        
    def _generate_entity_summary(self, entities: Dict[str, Any]) -> str:
        """Generate a summary of the domain model"""
        if not entities:
            return "No clear domain entities identified"
            
        # Sort by importance
        sorted_entities = sorted(entities.items(), 
                               key=lambda x: x[1]["importance"], 
                               reverse=True)
        
        summary_parts = [f"The system manages {len(entities)} core business entities:"]
        
        for name, data in sorted_entities[:10]:  # Top 10
            ops_summary = []
            if data["crud_operations"]:
                ops_summary = [f"{k}({v})" for k, v in data["crud_operations"].items()]
            summary_parts.append(
                f"- **{name}**: {data['business_purpose']} "
                f"[Operations: {', '.join(ops_summary) if ops_summary else 'minimal'}]"
            )
            
        return "\n".join(summary_parts)
        
    def _build_relationship_graph(self, entities: Dict[str, Any], 
                                 relationships: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build a graph of entity relationships"""
        graph = defaultdict(list)
        
        for rel in relationships:
            from_entity = rel["from"]
            to_entity = rel["to"]
            rel_type = rel["type"]
            
            graph[from_entity].append(f"{to_entity} ({rel_type})")
            
        return dict(graph)
        
    def _map_user_journeys(self, file_contexts: Dict[str, Any], 
                          cross_file_intel: Dict[str, Any],
                          file_contents: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Map how users interact with the system"""
        
        journeys = []
        api_boundaries = cross_file_intel.get('api_boundaries', {})
        
        # Common user journey patterns
        journey_patterns = {
            "authentication": {
                "keywords": ["login", "logout", "register", "signup", "auth", "session", "token"],
                "description": "User authentication and session management",
                "typical_flow": ["Registration/Login", "Token Generation", "Session Management", "Logout"]
            },
            "user_onboarding": {
                "keywords": ["onboard", "welcome", "setup", "profile", "wizard", "getting_started"],
                "description": "New user onboarding and initial setup",
                "typical_flow": ["Account Creation", "Profile Setup", "Preferences", "Tutorial"]
            },
            "purchase_flow": {
                "keywords": ["cart", "checkout", "payment", "order", "purchase", "buy", "transaction"],
                "description": "E-commerce purchase workflow",
                "typical_flow": ["Product Selection", "Cart Management", "Checkout", "Payment", "Confirmation"]
            },
            "content_management": {
                "keywords": ["create", "edit", "publish", "draft", "content", "article", "post"],
                "description": "Content creation and management",
                "typical_flow": ["Create Draft", "Edit Content", "Review", "Publish"]
            },
            "search_discovery": {
                "keywords": ["search", "find", "filter", "browse", "discover", "explore", "recommend"],
                "description": "Search and discovery features",
                "typical_flow": ["Search Input", "Filter Results", "View Details", "Take Action"]
            },
            "user_communication": {
                "keywords": ["message", "chat", "email", "notify", "comment", "reply", "send"],
                "description": "Communication between users or with system",
                "typical_flow": ["Compose", "Send", "Receive", "Reply"]
            },
            "reporting_analytics": {
                "keywords": ["report", "analytics", "dashboard", "metrics", "export", "visualize"],
                "description": "Data analysis and reporting",
                "typical_flow": ["Select Data", "Apply Filters", "Generate Report", "Export/Share"]
            },
            "admin_management": {
                "keywords": ["admin", "manage", "configure", "settings", "users", "permissions"],
                "description": "Administrative functions",
                "typical_flow": ["Access Admin", "Select Entity", "Perform Action", "Confirm Changes"]
            },
            "subscription_management": {
                "keywords": ["subscribe", "unsubscribe", "upgrade", "downgrade", "billing", "plan"],
                "description": "Subscription and billing management",
                "typical_flow": ["View Plans", "Select Plan", "Payment Setup", "Manage Subscription"]
            },
            "support_help": {
                "keywords": ["help", "support", "ticket", "issue", "contact", "faq", "documentation"],
                "description": "User support and help system",
                "typical_flow": ["Access Help", "Search/Browse", "Create Ticket", "Resolution"]
            }
        }
        
        # Analyze files for journey indicators
        for journey_type, journey_info in journey_patterns.items():
            journey_files = []
            journey_endpoints = []
            
            # Check file contexts
            for file_path, context in file_contexts.items():
                # Get relevant text to check
                if hasattr(context, 'purpose'):
                    purpose = getattr(context, 'purpose', '')
                elif isinstance(context, dict):
                    purpose = str(context.get('purpose', ''))
                else:
                    purpose = ''
                    
                if hasattr(context, 'business_logic'):
                    business_logic = getattr(context, 'business_logic', '')
                elif isinstance(context, dict):
                    business_logic = str(context.get('business_logic', ''))
                else:
                    business_logic = ''
                combined_text = f"{file_path} {purpose} {business_logic}".lower()
                
                # Check for journey keywords
                matches = sum(1 for keyword in journey_info["keywords"] if keyword in combined_text)
                if matches >= 2:  # At least 2 keyword matches
                    if hasattr(context, 'critical_functions'):
                        critical_funcs = context.critical_functions
                    elif isinstance(context, dict):
                        critical_funcs = context.get('critical_functions', [])
                    else:
                        critical_funcs = []
                    journey_files.append({
                        "file": file_path,
                        "role": purpose,
                        "relevance_score": matches,
                        "key_functions": [f['name'] for f in critical_funcs[:3]]
                    })
                    
            # Check API boundaries for entry points
            for boundary_type, boundaries in api_boundaries.items():
                for boundary in boundaries:
                    if any(keyword in boundary["file"].lower() for keyword in journey_info["keywords"]):
                        journey_endpoints.append({
                            "type": boundary_type,
                            "file": boundary["file"],
                            "risk": boundary.get("risk", "MEDIUM")
                        })
                        
            # Create journey if we found relevant files
            if journey_files:
                # Sort by relevance
                journey_files.sort(key=lambda x: x["relevance_score"], reverse=True)
                
                # Trace execution flow for this journey
                execution_flow = self._trace_journey_flow(
                    journey_files, 
                    cross_file_intel.get('execution_paths', [])
                )
                
                journeys.append({
                    "journey_type": journey_type,
                    "description": journey_info["description"],
                    "typical_flow": journey_info["typical_flow"],
                    "involved_files": journey_files[:10],  # Top 10 most relevant
                    "entry_points": journey_endpoints,
                    "execution_flow": execution_flow,
                    "complexity": self._assess_journey_complexity(journey_files, execution_flow),
                    "critical_points": self._identify_critical_points(journey_files, file_contexts)
                })
                
        # Sort journeys by importance
        journeys.sort(key=lambda j: len(j["involved_files"]) + len(j["entry_points"]), reverse=True)
        
        return journeys
        
    def _trace_journey_flow(self, journey_files: List[Dict[str, Any]], 
                           execution_paths: List[List[str]]) -> List[str]:
        """Trace the execution flow for a user journey"""
        flow = []
        journey_file_paths = {f["file"] for f in journey_files}
        
        # Find execution paths that include journey files
        for path in execution_paths:
            path_files = {step.split(':')[0] for step in path if ':' in step}
            if path_files.intersection(journey_file_paths):
                # This path is part of the journey
                flow.extend([step for step in path if step.split(':')[0] in journey_file_paths])
                
        # Deduplicate while preserving order
        seen = set()
        unique_flow = []
        for item in flow:
            if item not in seen:
                seen.add(item)
                unique_flow.append(item)
                
        return unique_flow
        
    def _assess_journey_complexity(self, journey_files: List[Dict[str, Any]], 
                                  execution_flow: List[str]) -> str:
        """Assess the complexity of a user journey"""
        file_count = len(journey_files)
        flow_steps = len(execution_flow)
        
        if file_count > 10 or flow_steps > 15:
            return "HIGH - Multiple systems and complex flow"
        elif file_count > 5 or flow_steps > 8:
            return "MEDIUM - Moderate number of components"
        else:
            return "LOW - Simple, straightforward flow"
            
    def _identify_critical_points(self, journey_files: List[Dict[str, Any]], 
                                 file_contexts: Dict[str, Any]) -> List[str]:
        """Identify critical points in a user journey"""
        critical_points = []
        
        for jf in journey_files:
            file_path = jf["file"]
            if file_path in file_contexts:
                context = file_contexts[file_path]
                
                # Check for critical indicators
                if hasattr(context, 'security_concerns'):
                    sec_concerns = context.security_concerns
                elif isinstance(context, dict):
                    sec_concerns = context.get('security_concerns', [])
                else:
                    sec_concerns = []
                if sec_concerns:
                    critical_points.append(f"{file_path}: Security validation required")
                    
                if hasattr(context, 'error_handling'):
                    error_handling = context.error_handling
                elif isinstance(context, dict):
                    error_handling = context.get('error_handling', [])
                else:
                    error_handling = []
                if len(error_handling) > 2:
                    critical_points.append(f"{file_path}: Complex error handling")
                    
                if hasattr(context, 'side_effects'):
                    side_effects = context.side_effects
                elif isinstance(context, dict):
                    side_effects = context.get('side_effects', [])
                else:
                    side_effects = []
                if "Database operations" in side_effects:
                    critical_points.append(f"{file_path}: Data persistence point")
                    
        return critical_points[:5]  # Top 5 critical points
        
    def _extract_business_rules(self, file_contexts: Dict[str, Any], 
                               file_contents: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Extract business rules and constraints from code"""
        
        rules = []
        
        if not file_contents:
            return rules
            
        # Patterns that indicate business rules
        rule_patterns = [
            # Validation rules
            (r'(?:if|when|unless)\s+.*?(\w+).*?(?:<|>|<=|>=|==|!=)\s*(\d+)', 'numeric_constraint'),
            (r'(?:min|max)(?:_|\s)?(?:length|size|value|amount|price)\s*[=:]\s*(\d+)', 'limit_constraint'),
            (r'(?:validate|check|ensure|require|must)[\s_]+(\w+)', 'validation_rule'),
            (r'raise\s+\w*(?:Error|Exception).*?["\']([^"\']+)["\']', 'business_exception'),
            (r'assert\s+.*?,\s*["\']([^"\']+)["\']', 'assertion'),
            
            # Business logic patterns
            (r'(?:can|cannot|able to|allowed to|permission)[\s_]+(\w+)', 'permission_rule'),
            (r'(?:before|after|during)\s+(\w+)', 'temporal_rule'),
            (r'(?:daily|weekly|monthly|yearly)\s+(\w+)', 'recurring_rule'),
            (r'(?:expires?|expir\w+)\s+(?:in|after)\s+(\d+)', 'expiration_rule'),
            
            # Documented rules
            (r'#\s*(?:RULE|REQUIREMENT|CONSTRAINT|POLICY):\s*(.+)', 'documented_rule'),
            (r'""".*?(?:Rule|Requirement|Constraint):\s*(.+?)"""', 'docstring_rule'),
            
            # Configuration rules
            (r'(?:MAX|MIN)_(\w+)\s*=\s*(\d+)', 'configuration_limit'),
            (r'(?:ENABLE|DISABLE|ALLOW)_(\w+)\s*=\s*(True|False)', 'feature_flag'),
        ]
        
        # Extract rules from each file
        for file_path, content in file_contents.items():
            context = file_contexts.get(file_path, {})
            
            for pattern, rule_type in rule_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for match in matches:
                    rule_text = match.group(0).strip()
                    rule_detail = match.group(1) if match.lastindex >= 1 else rule_text
                    
                    # Extract context around the rule
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    code_context = content[start:end].strip()
                    
                    # Determine business impact
                    business_impact = self._assess_rule_impact(rule_text, rule_type, code_context)
                    
                    rules.append({
                        "rule": rule_detail,
                        "type": rule_type,
                        "file": file_path,
                        "full_text": rule_text,
                        "code_context": code_context,
                        "business_impact": business_impact,
                        "enforcement": self._determine_enforcement_level(rule_type, code_context)
                    })
                    
        # Deduplicate and categorize rules
        unique_rules = self._deduplicate_rules(rules)
        categorized_rules = self._categorize_rules(unique_rules)
        
        return categorized_rules
        
    def _assess_rule_impact(self, rule_text: str, rule_type: str, context: str) -> str:
        """Assess the business impact of a rule"""
        
        high_impact_keywords = ['payment', 'money', 'price', 'auth', 'security', 'password', 'admin', 'delete', 'critical']
        medium_impact_keywords = ['user', 'order', 'product', 'validate', 'limit', 'quota']
        
        rule_lower = rule_text.lower()
        context_lower = context.lower()
        combined = f"{rule_lower} {context_lower}"
        
        if any(keyword in combined for keyword in high_impact_keywords):
            return "HIGH - Affects critical business operations"
        elif any(keyword in combined for keyword in medium_impact_keywords):
            return "MEDIUM - Affects standard business processes"
        else:
            return "LOW - Basic validation or constraint"
            
    def _determine_enforcement_level(self, rule_type: str, context: str) -> str:
        """Determine how strictly a rule is enforced"""
        
        if 'raise' in context or 'throw' in context or 'assert' in context:
            return "STRICT - Throws exception on violation"
        elif 'return false' in context or 'return null' in context:
            return "SOFT - Returns error gracefully"
        elif 'log' in context or 'warn' in context:
            return "WARNING - Logs violation but continues"
        else:
            return "UNKNOWN - Enforcement unclear"
            
    def _deduplicate_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate rules"""
        seen = set()
        unique_rules = []
        
        for rule in rules:
            # Create a unique key
            key = f"{rule['rule']}:{rule['type']}:{rule['file']}"
            if key not in seen:
                seen.add(key)
                unique_rules.append(rule)
                
        return unique_rules
        
    def _categorize_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Categorize rules by business domain"""
        
        categories = {
            "authentication": ["auth", "login", "password", "token", "session"],
            "authorization": ["permission", "role", "admin", "access", "allowed"],
            "financial": ["payment", "price", "cost", "fee", "billing", "money"],
            "data_validation": ["validate", "check", "ensure", "format", "pattern"],
            "business_logic": ["order", "product", "user", "customer", "workflow"],
            "system_limits": ["max", "min", "limit", "quota", "threshold"],
            "temporal": ["expire", "deadline", "schedule", "time", "date"],
            "compliance": ["gdpr", "pci", "hipaa", "audit", "regulation"]
        }
        
        for rule in rules:
            rule_text = f"{rule['rule']} {rule['code_context']}".lower()
            
            # Find matching category
            rule["category"] = "general"
            for category, keywords in categories.items():
                if any(keyword in rule_text for keyword in keywords):
                    rule["category"] = category
                    break
                    
        return rules
        
    def _map_business_flows(self, file_contexts: Dict[str, Any], 
                           cross_file_intel: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map business processes and workflows"""
        
        flows = []
        execution_paths = cross_file_intel.get('execution_paths', [])
        
        # Analyze each execution path for business meaning
        for i, exec_path in enumerate(execution_paths):
            business_steps = []
            flow_type = "unknown"
            
            for step in exec_path:
                if ':' in step:
                    file_path, func = step.split(':', 1)
                    if file_path in file_contexts:
                        context = file_contexts[file_path]
                        
                        # Get context attributes
                        if hasattr(context, 'purpose'):
                            purpose = context.purpose
                        elif isinstance(context, dict):
                            purpose = context.get('purpose', '')
                        else:
                            purpose = ''
                            
                        if hasattr(context, 'business_logic'):
                            business_logic = context.business_logic
                        elif isinstance(context, dict):
                            business_logic = context.get('business_logic', '')
                        else:
                            business_logic = ''
                            
                        if hasattr(context, 'side_effects'):
                            side_effects = context.side_effects
                        elif isinstance(context, dict):
                            side_effects = context.get('side_effects', [])
                        else:
                            side_effects = []
                        
                        business_steps.append({
                            "technical_step": step,
                            "business_meaning": f"{purpose} - {func}",
                            "side_effects": side_effects,
                            "file": file_path
                        })
                        
                        # Infer flow type
                        if not flow_type or flow_type == "unknown":
                            flow_type = self._infer_flow_type(purpose, business_logic)
                            
            if business_steps:
                flows.append({
                    "flow_id": f"flow_{i+1}",
                    "flow_type": flow_type,
                    "steps": business_steps,
                    "description": self._describe_business_flow(business_steps, flow_type),
                    "data_flow": self._trace_data_flow(business_steps),
                    "integration_points": self._identify_integration_points(business_steps)
                })
                
        return flows
        
    def _infer_flow_type(self, purpose: str, business_logic: str) -> str:
        """Infer the type of business flow"""
        combined = f"{purpose} {business_logic}".lower()
        
        flow_types = [
            ("authentication", ["auth", "login", "session"]),
            ("transaction", ["order", "payment", "purchase"]),
            ("data_processing", ["process", "transform", "calculate"]),
            ("communication", ["email", "notify", "message"]),
            ("reporting", ["report", "analytics", "export"]),
            ("crud_operation", ["create", "read", "update", "delete"]),
            ("integration", ["api", "sync", "import", "export"]),
        ]
        
        for flow_name, keywords in flow_types:
            if any(keyword in combined for keyword in keywords):
                return flow_name
                
        return "general_workflow"
        
    def _describe_business_flow(self, steps: List[Dict[str, Any]], flow_type: str) -> str:
        """Generate a business description of a technical flow"""
        
        descriptions = {
            "authentication": "User authentication and session management flow",
            "transaction": "Business transaction processing workflow",
            "data_processing": "Data transformation and processing pipeline",
            "communication": "Communication and notification workflow",
            "reporting": "Report generation and analytics flow",
            "crud_operation": "Data management operations",
            "integration": "External system integration flow",
            "general_workflow": "Business process workflow"
        }
        
        base_description = descriptions.get(flow_type, "Business workflow")
        
        # Add specific details
        if len(steps) > 5:
            base_description += f" involving {len(steps)} steps across multiple components"
        
        # Check for specific patterns
        has_db = any("Database" in str(step.get("side_effects", [])) for step in steps)
        has_api = any("api" in step["business_meaning"].lower() for step in steps)
        
        if has_db and has_api:
            base_description += " with API endpoints and data persistence"
        elif has_db:
            base_description += " with database operations"
        elif has_api:
            base_description += " exposing API endpoints"
            
        return base_description
        
    def _trace_data_flow(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Trace how data flows through the business process"""
        data_flow = []
        
        for i, step in enumerate(steps):
            effects = step.get("side_effects", [])
            
            if "Network calls" in effects or "api" in step["business_meaning"].lower():
                data_flow.append(f"Step {i+1}: External data input/output")
            elif "Database operations" in effects:
                data_flow.append(f"Step {i+1}: Data persistence")
            elif "Cache operations" in effects:
                data_flow.append(f"Step {i+1}: Cached data access")
            elif "File I/O" in effects:
                data_flow.append(f"Step {i+1}: File system interaction")
                
        return data_flow
        
    def _identify_integration_points(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Identify where the flow integrates with external systems"""
        integrations = []
        
        for step in steps:
            effects = step.get("side_effects", [])
            
            for effect in effects:
                if effect not in ["Logging", "Environment access"]:  # Skip common ones
                    integrations.append(f"{step['file']}: {effect}")
                    
        return list(set(integrations))  # Deduplicate
        
    def _identify_key_features(self, file_contexts: Dict[str, Any], 
                              file_contents: Optional[Dict[str, str]] = None) -> Set[str]:
        """Identify the key features of the application"""
        
        features = set()
        
        # Feature detection patterns
        feature_indicators = {
            "User Authentication": ["login", "logout", "auth", "session", "jwt", "oauth"],
            "User Registration": ["register", "signup", "onboard", "create_account"],
            "Password Management": ["password", "reset_password", "forgot_password", "change_password"],
            "Payment Processing": ["payment", "charge", "stripe", "paypal", "billing", "invoice"],
            "Subscription Management": ["subscription", "plan", "tier", "upgrade", "downgrade"],
            "Search Functionality": ["search", "filter", "query", "find", "elasticsearch", "algolia"],
            "File Upload/Download": ["upload", "download", "file", "attachment", "media", "s3"],
            "Email Communication": ["email", "smtp", "sendgrid", "mailgun", "ses"],
            "Real-time Features": ["websocket", "socket.io", "realtime", "push", "live"],
            "API Integration": ["api", "webhook", "integration", "oauth", "rest", "graphql"],
            "Data Export/Import": ["export", "import", "csv", "excel", "pdf", "report"],
            "Multi-tenancy": ["tenant", "organization", "workspace", "multi-tenant"],
            "Internationalization": ["i18n", "locale", "translation", "language"],
            "Caching": ["cache", "redis", "memcache", "cdn"],
            "Queue Processing": ["queue", "job", "worker", "celery", "rabbitmq", "sqs"],
            "Analytics/Metrics": ["analytics", "metrics", "tracking", "statistics", "dashboard"],
            "Admin Panel": ["admin", "management", "backoffice", "control_panel"],
            "Notification System": ["notification", "alert", "push_notification", "notify"],
            "Two-Factor Auth": ["2fa", "two_factor", "totp", "mfa", "authenticator"],
            "Social Login": ["oauth", "google_login", "facebook_login", "github_login", "social_auth"],
            "Content Management": ["cms", "content", "article", "blog", "publish"],
            "E-commerce": ["cart", "checkout", "product", "order", "inventory"],
            "Scheduling": ["schedule", "calendar", "appointment", "booking", "cron"],
            "Audit Logging": ["audit", "activity_log", "history", "trail"],
            "Data Backup": ["backup", "restore", "snapshot", "archive"],
            "Rate Limiting": ["rate_limit", "throttle", "quota", "api_limit"],
            "Geolocation": ["location", "gps", "map", "geocode", "distance"],
            "Machine Learning": ["ml", "ai", "predict", "model", "tensorflow", "scikit"],
            "Blockchain": ["blockchain", "crypto", "smart_contract", "web3"],
            "Compliance": ["gdpr", "pci", "hipaa", "compliance", "regulation"]
        }
        
        # Check all contexts and content
        for file_path, context in file_contexts.items():
            # Combine all text to search
            text_to_search = file_path.lower()
            
            # Add context attributes
            if hasattr(context, 'purpose'):
                text_to_search += f" {context.purpose.lower()}"
            if hasattr(context, 'business_logic'):
                text_to_search += f" {context.business_logic.lower()}"
                
            # Add file content if available
            if file_contents and file_path in file_contents:
                # Just check first 1000 chars to avoid too much processing
                text_to_search += f" {file_contents[file_path][:1000].lower()}"
                
            # Check for feature indicators
            for feature, keywords in feature_indicators.items():
                if any(keyword in text_to_search for keyword in keywords):
                    features.add(feature)
                    
        return features
        
    def _identify_compliance_requirements(self, file_contexts: Dict[str, Any], 
                                        file_contents: Optional[Dict[str, str]] = None) -> Set[str]:
        """Identify potential compliance and regulatory requirements"""
        
        compliance_indicators = set()
        
        # Compliance patterns and their indicators
        compliance_patterns = {
            "GDPR Compliance": [
                "gdpr", "data_protection", "right_to_forget", "data_portability", 
                "consent", "privacy_policy", "data_deletion", "personal_data"
            ],
            "PCI DSS Compliance": [
                "pci", "credit_card", "card_number", "cvv", "payment_security",
                "cardholder", "pci_compliance", "tokenization"
            ],
            "HIPAA Compliance": [
                "hipaa", "health_data", "medical", "patient", "phi", 
                "health_information", "medical_record"
            ],
            "SOX Compliance": [
                "sox", "sarbanes", "financial_audit", "financial_control",
                "audit_trail", "financial_reporting"
            ],
            "CCPA Compliance": [
                "ccpa", "california_privacy", "data_sale", "opt_out",
                "consumer_privacy", "data_disclosure"
            ],
            "SOC 2 Compliance": [
                "soc2", "security_audit", "availability", "confidentiality",
                "processing_integrity", "privacy_principles"
            ],
            "ISO 27001": [
                "iso27001", "information_security", "isms", "security_controls",
                "risk_assessment", "security_policy"
            ],
            "Data Encryption": [
                "encrypt", "decrypt", "aes", "rsa", "tls", "ssl",
                "encryption_key", "crypto", "bcrypt", "hash"
            ],
            "Access Control": [
                "rbac", "access_control", "permission", "authorization",
                "role_based", "acl", "privilege"
            ],
            "Audit Requirements": [
                "audit_log", "audit_trail", "activity_log", "compliance_log",
                "event_logging", "security_log"
            ],
            "Data Retention": [
                "retention_policy", "data_retention", "archive", "purge",
                "retention_period", "data_lifecycle"
            ],
            "Cookie Compliance": [
                "cookie_policy", "cookie_consent", "tracking", "analytics_consent",
                "third_party_cookies"
            ]
        }
        
        # Check all files for compliance indicators
        all_text = ""
        
        for file_path, context in file_contexts.items():
            # Add context text
            if hasattr(context, 'security_concerns'):
                all_text += " ".join(context.security_concerns).lower() + " "
                
            # Add file content
            if file_contents and file_path in file_contents:
                all_text += file_contents[file_path][:2000].lower() + " "  # First 2000 chars
                
        # Check for compliance indicators
        for compliance, keywords in compliance_patterns.items():
            if any(keyword in all_text for keyword in keywords):
                compliance_indicators.add(compliance)
                
        # Add general security if encryption is found
        if any(enc in all_text for enc in ["encrypt", "hash", "bcrypt", "jwt"]):
            compliance_indicators.add("Security Best Practices")
            
        return compliance_indicators
        
    def _generate_executive_summary(self) -> str:
        """Generate a high-level executive summary for stakeholders"""
        
        summary_parts = [
            "## Executive Summary",
            "",
            "### Application Overview"
        ]
        
        # Summarize what the application does
        if self.key_features:
            top_features = list(self.key_features)[:5]
            summary_parts.append(
                f"This application provides {len(self.key_features)} key features including: "
                f"{', '.join(top_features)}"
            )
        else:
            summary_parts.append("This application provides business functionality.")
            
        # Domain model summary
        if self.domain_model and self.domain_model.get('entities'):
            entity_count = len(self.domain_model['entities'])
            top_entities = sorted(
                self.domain_model['entities'].items(),
                key=lambda x: x[1]['importance'],
                reverse=True
            )[:3]
            
            summary_parts.extend([
                "",
                f"### Core Business Model",
                f"The system manages {entity_count} business entities, primarily:"
            ])
            
            for name, data in top_entities:
                summary_parts.append(f"- **{name}**: {data['business_purpose']}")
                
        # User journey summary
        if self.user_journeys:
            summary_parts.extend([
                "",
                "### Key User Journeys"
            ])
            
            for journey in self.user_journeys[:3]:
                file_count = len(journey['involved_files'])
                summary_parts.append(
                    f"- **{journey['journey_type'].replace('_', ' ').title()}**: "
                    f"{journey['description']} ({file_count} components)"
                )
                
        # Business rules summary
        if self.business_rules:
            rule_categories = defaultdict(int)
            for rule in self.business_rules:
                rule_categories[rule.get('category', 'general')] += 1
                
            summary_parts.extend([
                "",
                f"### Business Rules & Constraints",
                f"The system enforces {len(self.business_rules)} business rules across:"
            ])
            
            for category, count in sorted(rule_categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                summary_parts.append(f"- {category.replace('_', ' ').title()}: {count} rules")
                
        # Compliance summary
        if self.compliance_indicators:
            summary_parts.extend([
                "",
                "### Compliance & Security",
                "The system appears to implement:"
            ])
            
            for compliance in sorted(self.compliance_indicators)[:5]:
                summary_parts.append(f"- {compliance}")
                
        # Architecture insights
        if self.business_flows:
            flow_types = [f['flow_type'] for f in self.business_flows]
            unique_flows = len(set(flow_types))
            summary_parts.extend([
                "",
                "### System Architecture",
                f"The application implements {unique_flows} distinct business workflows, "
                f"with {len(self.business_flows)} execution paths identified."
            ])
            
        return "\n".join(summary_parts)
        
    def _generate_business_glossary(self) -> Dict[str, str]:
        """Generate a glossary of business terms used in the codebase"""
        
        glossary = {}
        
        # From domain model
        if self.domain_model and self.domain_model.get('entities'):
            for entity_name, entity_data in self.domain_model['entities'].items():
                glossary[entity_name] = entity_data['business_purpose']
                
        # Common business terms
        standard_terms = {
            "Authentication": "Process of verifying user identity",
            "Authorization": "Process of determining user permissions",
            "Transaction": "A complete business operation",
            "Workflow": "Sequence of steps to complete a business process",
            "Entity": "A business object or concept (e.g., User, Order)",
            "Validation": "Ensuring data meets business requirements",
            "Integration": "Connection with external systems or services",
            "Audit": "Recording of system activities for compliance",
            "Cache": "Temporary storage for performance optimization",
            "API": "Interface for external system communication"
        }
        
        # Add standard terms that appear in the codebase
        for term, definition in standard_terms.items():
            if term not in glossary:
                # Check if term appears in features or rules
                term_found = False
                
                # Check in features
                for feature in self.key_features:
                    if term.lower() in feature.lower():
                        term_found = True
                        break
                        
                # Check in business rules
                if not term_found:
                    for rule in self.business_rules:
                        if term.lower() in str(rule).lower():
                            term_found = True
                            break
                            
                if term_found:
                    glossary[term] = definition
                
        return glossary
        
    def _create_ai_business_context(self) -> str:
        """Create an AI-friendly summary of the business context"""
        
        context_parts = [
            "🏢 BUSINESS CONTEXT FOR AI ASSISTANTS",
            "=" * 50,
            "",
            "This codebase implements the following business system:",
            ""
        ]
        
        # What does it do?
        if self.key_features:
            context_parts.append(f"**Primary Purpose**: A system that provides {', '.join(list(self.key_features)[:3])}")
        
        # Who uses it?
        if self.user_journeys:
            user_types = set()
            for journey in self.user_journeys:
                if "admin" in journey['journey_type']:
                    user_types.add("administrators")
                elif "customer" in journey['journey_type'] or "purchase" in journey['journey_type']:
                    user_types.add("customers")
                else:
                    user_types.add("users")
                    
            context_parts.append(f"**Target Users**: {', '.join(user_types)}")
            
        # Key business rules
        if self.business_rules:
            high_impact_rules = [r for r in self.business_rules if "HIGH" in r.get('business_impact', '')][:3]
            if high_impact_rules:
                context_parts.extend([
                    "",
                    "**Critical Business Rules**:"
                ])
                for rule in high_impact_rules:
                    context_parts.append(f"- {rule['rule']} ({rule['category']})")
                    
        # Compliance requirements
        if self.compliance_indicators:
            context_parts.extend([
                "",
                f"**Compliance Requirements**: {', '.join(list(self.compliance_indicators)[:3])}"
            ])
            
        # Integration points
        integration_count = sum(len(j.get('entry_points', [])) for j in self.user_journeys)
        if integration_count > 0:
            context_parts.append(f"**External Integrations**: {integration_count} API endpoints/integrations")
            
        context_parts.extend([
            "",
            "When modifying code, consider:",
            "1. Business rule enforcement and validation",
            "2. User journey continuity",
            "3. Compliance requirements",
            "4. Integration dependencies"
        ])
        
        return "\n".join(context_parts)
        
    def _generate_immediate_context(self) -> str:
        """Generate immediate context for AI to understand the business from the first message"""
        
        # Create a one-paragraph summary
        feature_summary = f"{len(self.key_features)} features" if self.key_features else "business functionality"
        entity_summary = f"{len(self.domain_model.get('entities', {}))} business entities" if self.domain_model else "data models"
        journey_summary = f"{len(self.user_journeys)} user workflows" if self.user_journeys else "user interactions"
        
        immediate_context = (
            f"This codebase implements a business application with {feature_summary}, "
            f"managing {entity_summary} through {journey_summary}. "
        )
        
        if self.compliance_indicators:
            immediate_context += f"The system requires {', '.join(list(self.compliance_indicators)[:2])} compliance. "
            
        if self.business_rules:
            rule_count = len(self.business_rules)
            immediate_context += f"It enforces {rule_count} business rules across various domains. "
            
        immediate_context += (
            "Understanding these business aspects is crucial before making any code modifications."
        )
        
        return immediate_context