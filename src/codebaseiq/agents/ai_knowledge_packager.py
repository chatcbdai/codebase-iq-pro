#!/usr/bin/env python3
"""
AI Knowledge Packager
Packages all analysis into a format optimized for AI consumption.
Ensures AI assistants have complete understanding from the first message.
"""

import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class AIKnowledgePackager:
    """
    Packages codebase knowledge into a format that AI assistants can immediately use.
    This is the final step that prevents AI from making uninformed changes.
    Provides 100% useful context at conversation startup.
    """
    
    def __init__(self):
        self.knowledge_package = {}
        self.danger_summary = []
        self.safe_zones = []
        self.critical_warnings = []
        
    def create_knowledge_package(self, 
                               deep_understanding: Dict[str, Any],
                               cross_file_intel: Dict[str, Any],
                               business_logic: Dict[str, Any],
                               original_file_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a comprehensive knowledge package for AI assistants.
        This gives AI complete understanding in one shot, preventing uninformed changes.
        """
        logger.info("📦 Creating AI Knowledge Package for immediate understanding...")
        
        # Extract key data from inputs
        self._extract_danger_zones(cross_file_intel)
        self._extract_safe_zones(deep_understanding, cross_file_intel)
        
        self.knowledge_package = {
            "metadata": self._create_metadata(original_file_count),
            "instant_context": self._create_instant_context(deep_understanding, business_logic),
            "danger_zones": self._package_danger_zones(cross_file_intel),
            "safe_modification_guide": self._create_safe_modification_guide(cross_file_intel),
            "system_understanding": self._create_system_understanding(deep_understanding, business_logic),
            "business_context": self._package_business_context(business_logic),
            "code_patterns": self._extract_code_patterns(deep_understanding),
            "testing_requirements": self._identify_testing_requirements(cross_file_intel, deep_understanding),
            "ai_instructions": self._generate_comprehensive_ai_instructions(
                deep_understanding, cross_file_intel, business_logic
            ),
            "quick_reference": self._create_quick_reference(deep_understanding, cross_file_intel, business_logic),
            "modification_checklist": self._create_modification_checklist(),
            "emergency_contacts": self._identify_emergency_contacts(deep_understanding)
        }
        
        return self.knowledge_package
        
    def _create_metadata(self, original_file_count: Optional[int] = None) -> Dict[str, Any]:
        """Create metadata about the analysis"""
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "version": "2.0-enhanced",
            "purpose": "Complete codebase understanding for AI assistants - prevents uninformed changes",
            "files_analyzed": original_file_count or "unknown",
            "warnings": [
                "⚠️ ALWAYS consult danger_zones before modifying any file",
                "⚠️ NEVER modify CRITICAL files without explicit permission",
                "⚠️ CHECK impact analysis for every change",
                "⚠️ RUN tests for all files in impact zones"
            ]
        }
        
    def _create_instant_context(self, deep_understanding: Dict[str, Any], 
                               business_logic: Dict[str, Any]) -> str:
        """Create instant context that AI can read in seconds"""
        
        # Quick stats
        total_files = deep_understanding.get('total_files_analyzed', 0)
        languages = deep_understanding.get('languages_found', [])
        critical_files = len(deep_understanding.get('critical_files', []))
        
        # Business summary
        business_summary = business_logic.get('immediate_context', '')
        key_features = business_logic.get('key_features', [])[:3]
        compliance = business_logic.get('compliance_requirements', [])[:2]
        
        instant_context = f"""
🚀 INSTANT CODEBASE CONTEXT (Read this first!)
=============================================

📊 **Quick Stats:**
- Files: {total_files} | Languages: {', '.join(languages)} | Critical files: {critical_files}

💼 **What This Does:**
{business_summary}

🌟 **Key Features:** {', '.join(key_features) if key_features else 'Multiple features'}

🔒 **Compliance:** {', '.join(compliance) if compliance else 'Standard requirements'}

⚡ **CRITICAL RULE:** Check danger_zones before ANY modification!

🎯 **Your Goal:** Make changes safely without breaking existing functionality.
"""
        
        return instant_context
        
    def _extract_danger_zones(self, cross_file_intel: Dict[str, Any]):
        """Extract and categorize danger zones"""
        impact_zones = cross_file_intel.get('impact_zones', {})
        
        for file_path, impact in impact_zones.items():
            if impact['risk_level'] == 'CRITICAL':
                self.danger_summary.append({
                    'file': file_path,
                    'level': 'CRITICAL',
                    'reason': impact['risk_factors'],
                    'impact_count': impact['total_impact']
                })
            elif impact['risk_level'] == 'HIGH':
                self.danger_summary.append({
                    'file': file_path,
                    'level': 'HIGH',
                    'reason': impact['risk_factors'],
                    'impact_count': impact['total_impact']
                })
                
    def _extract_safe_zones(self, deep_understanding: Dict[str, Any], 
                           cross_file_intel: Dict[str, Any]):
        """Identify files that are relatively safe to modify"""
        impact_zones = cross_file_intel.get('impact_zones', {})
        
        for file_path, impact in impact_zones.items():
            if impact['risk_level'] == 'LOW' and impact['total_impact'] < 2:
                self.safe_zones.append({
                    'file': file_path,
                    'reason': 'Low impact, few dependencies'
                })
                
    def _package_danger_zones(self, cross_file_intel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Package danger zones in a format that screams "BE CAREFUL!"
        This is critical for preventing breaking changes.
        """
        ai_guidance = cross_file_intel.get('ai_modification_guidance', {})
        
        danger_zones = {
            "summary": f"⛔ {len(self.danger_summary)} files require extreme caution",
            "do_not_modify": [],
            "extreme_caution": [],
            "careful_review": [],
            "quick_check": {
                "is_file_dangerous": "Check if file is in do_not_modify or extreme_caution lists",
                "how_to_check": "Search for filename in danger_zones before ANY modification"
            }
        }
        
        # Categorize files
        for item in ai_guidance.get('risk_categories', {}).get('do_not_modify', []):
            danger_zones["do_not_modify"].append({
                "file": item['file'],
                "reason": item['reason'],
                "alternatives": item.get('alternatives', ['Create new file instead']),
                "ai_directive": "⛔ DO NOT MODIFY - Create new functions or use alternatives"
            })
            
        for item in ai_guidance.get('risk_categories', {}).get('extreme_caution', []):
            danger_zones["extreme_caution"].append({
                "file": item['file'],
                "impact": item['impact_count'],
                "checklist": item.get('checklist', []),
                "ai_directive": "⚠️ EXTREME CAUTION - Follow checklist exactly"
            })
            
        # Add circular dependencies as danger zones
        circular_deps = cross_file_intel.get('circular_dependencies', [])
        if circular_deps:
            danger_zones["circular_dependencies"] = {
                "warning": "⚠️ These files have circular dependencies - changes require coordinated updates",
                "cycles": circular_deps[:5]
            }
            
        # Critical interfaces
        critical_interfaces = cross_file_intel.get('critical_interfaces', [])
        danger_zones["critical_interfaces"] = [
            {
                "file": intf['file'],
                "dependents": intf['dependent_count'],
                "type": intf['interface_type'],
                "ai_directive": intf['ai_guidance']
            }
            for intf in critical_interfaces[:10]
        ]
        
        return danger_zones
        
    def _create_safe_modification_guide(self, cross_file_intel: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive guide for safe code modifications"""
        
        return {
            "golden_rules": [
                "1. 🔍 ALWAYS check danger_zones BEFORE opening any file",
                "2. 📊 Review impact analysis to see what could break",
                "3. 🧪 Write tests BEFORE making changes",
                "4. 🎯 Prefer creating new functions over modifying existing ones",
                "5. 📝 Document WHY you're making changes",
                "6. 🔄 Make small, incremental changes",
                "7. ✅ Run ALL tests in the impact zone"
            ],
            "modification_workflow": [
                "Step 1: Check if file is in danger_zones",
                "Step 2: Run impact analysis for the file",
                "Step 3: Read current implementation thoroughly",
                "Step 4: Write tests for your changes",
                "Step 5: Make minimal necessary changes",
                "Step 6: Run tests for modified file AND all impacted files",
                "Step 7: Document changes and reasoning"
            ],
            "safe_patterns": {
                "adding_features": "Create new functions/classes rather than modifying existing",
                "fixing_bugs": "Understand root cause, check impact, minimal fix",
                "refactoring": "One small change at a time, maintain interface",
                "performance": "Measure first, optimize carefully, preserve behavior"
            },
            "danger_patterns": {
                "modifying_interfaces": "Changes signature of widely-used functions",
                "changing_config": "Alters configuration that affects entire system",
                "database_changes": "Modifies schema or data access patterns",
                "security_changes": "Touches authentication, authorization, or encryption"
            }
        }
        
    def _create_system_understanding(self, deep_understanding: Dict[str, Any], 
                                   business_logic: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive system understanding section"""
        
        # Determine system type
        system_type = self._determine_system_type(deep_understanding, business_logic)
        
        # Key architectural insights
        entry_points = deep_understanding.get('entry_points', [])
        external_integrations = deep_understanding.get('external_integrations', {})
        
        return {
            "system_type": system_type,
            "architecture_style": self._infer_architecture_style(deep_understanding),
            "main_components": self._identify_main_components(deep_understanding),
            "entry_points": entry_points[:5] if entry_points else ["No clear entry points found"],
            "data_flow_summary": self._summarize_data_flow(business_logic),
            "external_dependencies": self._summarize_external_deps(external_integrations),
            "key_patterns": deep_understanding.get('immediate_ai_guidance', '').split('\n')[0],
            "technology_stack": deep_understanding.get('languages_found', [])
        }
        
    def _determine_system_type(self, deep_understanding: Dict[str, Any], 
                              business_logic: Dict[str, Any]) -> str:
        """Determine what type of system this is"""
        
        features = set(business_logic.get('key_features', []))
        
        if 'API Integration' in features or 'REST' in str(features):
            if 'Frontend' not in str(features):
                return "Backend API Service"
            else:
                return "Full-Stack Web Application"
        elif 'User Interface' in features or 'React' in str(features):
            return "Frontend Application"
        elif 'CLI' in str(features) or 'Command Line' in str(features):
            return "Command-Line Tool"
        elif 'Machine Learning' in features:
            return "ML/AI Application"
        else:
            return "General Purpose Application"
            
    def _infer_architecture_style(self, deep_understanding: Dict[str, Any]) -> str:
        """Infer the architectural style"""
        
        file_purposes = deep_understanding.get('file_purposes', {})
        
        # Look for architectural indicators
        if any('controller' in f.lower() for f in file_purposes):
            return "MVC (Model-View-Controller)"
        elif any('service' in f.lower() for f in file_purposes):
            # Check if any file purpose mentions microservices
            purposes_text = ' '.join(str(v) for v in file_purposes.values() if v)
            if 'microservice' in purposes_text.lower():
                return "Microservices"
            else:
                return "Service-Oriented"
        elif any('layer' in f.lower() for f in file_purposes):
            return "Layered Architecture"
        else:
            return "Modular Architecture"
            
    def _identify_main_components(self, deep_understanding: Dict[str, Any]) -> List[str]:
        """Identify the main components of the system"""
        
        components = []
        file_purposes = deep_understanding.get('file_purposes', {})
        
        # Group files by component type
        component_keywords = {
            'Authentication': ['auth', 'login', 'session'],
            'Database': ['database', 'db', 'model', 'schema'],
            'API': ['api', 'endpoint', 'route', 'controller'],
            'Business Logic': ['service', 'business', 'logic'],
            'Frontend': ['component', 'view', 'ui', 'frontend'],
            'Configuration': ['config', 'settings', 'env'],
            'Utilities': ['util', 'helper', 'common'],
            'Testing': ['test', 'spec', 'mock']
        }
        
        for comp_name, keywords in component_keywords.items():
            matching_files = 0
            for file_path in file_purposes:
                if any(kw in file_path.lower() for kw in keywords):
                    matching_files += 1
                    
            if matching_files > 0:
                components.append(f"{comp_name} ({matching_files} files)")
                
        return components[:8]  # Top 8 components
        
    def _summarize_data_flow(self, business_logic: Dict[str, Any]) -> str:
        """Summarize how data flows through the system"""
        
        flows = business_logic.get('business_flows', [])
        if not flows:
            return "Data flow analysis not available"
            
        flow_types = [f['flow_type'] for f in flows]
        unique_flow_types = list(set(flow_types))
        unique_flows = len(unique_flow_types)
        
        return f"{unique_flows} distinct data flows identified, including: {', '.join(unique_flow_types[:3])}"
        
    def _summarize_external_deps(self, integrations: Dict[str, List[str]]) -> Dict[str, int]:
        """Summarize external dependencies"""
        
        summary = {}
        for integration_type, files in integrations.items():
            if files:
                summary[integration_type] = len(files)
                
        return summary
        
    def _package_business_context(self, business_logic: Dict[str, Any]) -> Dict[str, Any]:
        """Package business context in AI-friendly format"""
        
        return {
            "executive_summary": business_logic.get('executive_summary', 'Not available'),
            "domain_entities": self._summarize_domain_entities(business_logic.get('domain_model', {})),
            "key_user_journeys": self._summarize_user_journeys(business_logic.get('user_journeys', [])),
            "critical_business_rules": self._extract_critical_rules(business_logic.get('business_rules', [])),
            "compliance_requirements": business_logic.get('compliance_requirements', []),
            "business_glossary": business_logic.get('business_glossary', {})
        }
        
    def _summarize_domain_entities(self, domain_model: Dict[str, Any]) -> List[Dict[str, str]]:
        """Summarize key domain entities"""
        
        entities = domain_model.get('entities', {})
        if not entities:
            return []
            
        # Sort by importance
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: x[1].get('importance', 0),
            reverse=True
        )[:10]  # Top 10
        
        return [
            {
                "name": name,
                "purpose": data.get('business_purpose', ''),
                "operations": data.get('crud_operations', {})
            }
            for name, data in sorted_entities
        ]
        
    def _summarize_user_journeys(self, journeys: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Summarize key user journeys"""
        
        return [
            {
                "journey": j['journey_type'],
                "description": j['description'],
                "complexity": j['complexity'],
                "files_involved": len(j['involved_files'])
            }
            for j in journeys[:5]  # Top 5
        ]
        
    def _extract_critical_rules(self, rules: List[Dict[str, Any]]) -> List[str]:
        """Extract critical business rules"""
        
        critical_rules = []
        
        for rule in rules:
            if 'HIGH' in rule.get('business_impact', ''):
                rule_text = f"{rule['rule']} ({rule['category']}) - {rule['enforcement']}"
                critical_rules.append(rule_text)
                
        return critical_rules[:10]  # Top 10 critical rules
        
    def _extract_code_patterns(self, deep_understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Extract common code patterns and conventions"""
        
        # This would analyze the codebase for patterns
        # For now, returning sensible defaults based on languages
        languages = deep_understanding.get('languages_found', [])
        
        patterns = {
            "naming_conventions": {},
            "common_patterns": [],
            "anti_patterns_to_avoid": [],
            "style_guide": {}
        }
        
        if 'python' in languages:
            patterns["naming_conventions"]["python"] = {
                "files": "lowercase_with_underscores.py",
                "classes": "PascalCase",
                "functions": "lowercase_with_underscores",
                "constants": "UPPERCASE_WITH_UNDERSCORES"
            }
            patterns["common_patterns"].extend([
                "Dependency injection",
                "Repository pattern",
                "Service layer"
            ])
            
        if 'javascript' in languages or 'typescript' in languages:
            patterns["naming_conventions"]["javascript"] = {
                "files": "camelCase.js or PascalCase.jsx",
                "classes": "PascalCase",
                "functions": "camelCase",
                "constants": "UPPERCASE_WITH_UNDERSCORES"
            }
            patterns["common_patterns"].extend([
                "React components",
                "Async/await patterns",
                "Module exports"
            ])
            
        patterns["anti_patterns_to_avoid"] = [
            "God objects/functions (too much responsibility)",
            "Circular dependencies",
            "Hardcoded values (use config)",
            "Ignoring error handling",
            "Modifying shared state directly"
        ]
        
        return patterns
        
    def _identify_testing_requirements(self, cross_file_intel: Dict[str, Any], 
                                     deep_understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Identify testing requirements for safe modifications"""
        
        # Identify test patterns from the codebase
        test_indicators = ['test', 'spec', 'jest', 'pytest', 'unittest', 'mocha']
        file_purposes = deep_understanding.get('file_purposes', {})
        
        test_files = []
        for file_path, purpose in file_purposes.items():
            if any(indicator in file_path.lower() for indicator in test_indicators):
                test_files.append(file_path)
                
        # Determine test framework
        test_framework = "Unknown"
        if test_files:
            sample_file = test_files[0].lower()
            if 'jest' in sample_file or 'test.js' in sample_file:
                test_framework = "Jest (JavaScript)"
            elif 'pytest' in sample_file or 'test_' in sample_file:
                test_framework = "pytest (Python)"
            elif 'spec' in sample_file:
                test_framework = "RSpec or Jasmine"
                
        # Critical test areas based on danger zones
        critical_test_areas = []
        for file_path, impact in cross_file_intel.get('impact_zones', {}).items():
            if impact['risk_level'] in ['CRITICAL', 'HIGH']:
                critical_test_areas.append({
                    "file": file_path,
                    "reason": f"High risk - affects {impact['total_impact']} files"
                })
                
        return {
            "test_framework": test_framework,
            "test_file_pattern": "test_*.py or *.test.js based on language",
            "test_files_found": len(test_files),
            "critical_test_areas": critical_test_areas[:10],
            "test_commands": self._infer_test_commands(test_framework),
            "testing_checklist": [
                "1. Write tests for new functionality BEFORE implementing",
                "2. Run existing tests to ensure nothing breaks",
                "3. Test all files in the impact zone",
                "4. Add integration tests for cross-file changes",
                "5. Verify edge cases and error handling"
            ]
        }
        
    def _infer_test_commands(self, framework: str) -> List[str]:
        """Infer common test commands"""
        
        commands = {
            "Jest (JavaScript)": ["npm test", "npm run test:watch", "jest --coverage"],
            "pytest (Python)": ["pytest", "python -m pytest", "pytest -v"],
            "Unknown": ["npm test", "python -m pytest", "make test"]
        }
        
        return commands.get(framework, commands["Unknown"])
        
    def _generate_comprehensive_ai_instructions(self, 
                                              deep_understanding: Dict[str, Any],
                                              cross_file_intel: Dict[str, Any],
                                              business_logic: Dict[str, Any]) -> str:
        """
        Generate comprehensive instructions for AI assistants.
        This is the most important section - tells AI exactly how to behave.
        """
        
        total_files = deep_understanding.get('total_files_analyzed', 0)
        critical_files = len(deep_understanding.get('critical_files', []))
        high_risk_files = len([
            f for f, z in cross_file_intel.get('impact_zones', {}).items()
            if z['risk_level'] in ['CRITICAL', 'HIGH']
        ])
        
        instructions = f"""
🤖 AI ASSISTANT INSTRUCTIONS - READ CAREFULLY!
============================================

You have been given COMPLETE knowledge about this codebase with {total_files} files.
There are {critical_files} critical files and {high_risk_files} high-risk files.

⚠️ MANDATORY RULES - MUST FOLLOW:

1. **BEFORE ANY CODE MODIFICATION:**
   - ✓ Check danger_zones to see if file is restricted
   - ✓ Review impact analysis for the file
   - ✓ Read the current implementation completely
   - ✓ Check business rules that might be affected
   - ✓ Plan your changes to minimize impact

2. **MODIFICATION RULES BY RISK LEVEL:**
   
   🔴 **CRITICAL FILES (do_not_modify list):**
   - NEVER modify these files directly
   - Create new files/functions instead
   - Use adapter patterns or wrappers
   - Must have explicit permission to modify

   🟡 **HIGH RISK FILES (extreme_caution list):**
   - Follow the provided checklist EXACTLY
   - Write comprehensive tests FIRST
   - Make minimal changes
   - Document every change and why

   🟢 **LOW RISK FILES:**
   - Still check impact analysis
   - Follow standard practices
   - Run relevant tests

3. **SAFE MODIFICATION PATTERNS:**
   - ✅ Adding new functions/classes
   - ✅ Creating new files
   - ✅ Adding optional parameters (with defaults)
   - ✅ Improving error messages
   - ✅ Adding logging/monitoring

4. **DANGEROUS MODIFICATION PATTERNS:**
   - ❌ Changing function signatures
   - ❌ Modifying configuration defaults
   - ❌ Altering database queries
   - ❌ Changing authentication/authorization
   - ❌ Modifying critical business logic

5. **TESTING REQUIREMENTS:**
   - Run tests for the modified file
   - Run tests for ALL files in impact zone
   - Add new tests for new functionality
   - Verify no regression in critical paths

6. **COMMUNICATION RULES:**
   - Always explain the impact of changes
   - Warn about risks before making changes
   - Suggest safer alternatives when possible
   - Ask for clarification if requirements unclear

7. **BUSINESS CONTEXT AWARENESS:**
   - Respect business rules and constraints
   - Maintain compliance requirements
   - Preserve user journey integrity
   - Don't break existing features

Remember: You're modifying a production system. Every change has consequences.
When in doubt, ASK before making changes. It's better to clarify than to break.

🎯 Your success is measured by:
- Zero breaking changes
- Minimal system disruption  
- Clear communication about impacts
- Maintaining all existing functionality
"""
        
        return instructions
        
    def _create_quick_reference(self, deep_understanding: Dict[str, Any],
                               cross_file_intel: Dict[str, Any],
                               business_logic: Dict[str, Any]) -> Dict[str, Any]:
        """Create a quick reference card for AI"""
        
        # Most dangerous files
        danger_files = []
        for file, zone in cross_file_intel.get('impact_zones', {}).items():
            if zone['risk_level'] == 'CRITICAL':
                danger_files.append(file)
                
        # Key features
        features = business_logic.get('key_features', [])[:5]
        
        # Main entities
        entities = list(business_logic.get('domain_model', {}).get('entities', {}).keys())[:5]
        
        return {
            "codebase_type": self._determine_system_type(deep_understanding, business_logic),
            "key_features": features,
            "main_entities": entities,
            "danger_files_count": len(danger_files),
            "danger_files_sample": danger_files[:5],
            "safe_files_count": len(self.safe_zones),
            "primary_language": deep_understanding.get('languages_found', ['unknown'])[0],
            "compliance_required": bool(business_logic.get('compliance_requirements')),
            "quick_checks": {
                "is_dangerous": "file in danger_zones['do_not_modify'] or danger_zones['extreme_caution']",
                "has_tests": "check testing_requirements['test_files_found'] > 0",
                "needs_review": "impact_zones[file]['total_impact'] > 5"
            }
        }
        
    def _create_modification_checklist(self) -> List[str]:
        """Create a standard checklist for any modification"""
        
        return [
            "□ Checked if file is in danger_zones",
            "□ Reviewed impact analysis for the file",
            "□ Read and understood current implementation",
            "□ Identified affected business rules",
            "□ Written tests for the changes",
            "□ Made minimal necessary changes",
            "□ Run tests for modified file",
            "□ Run tests for all impacted files",
            "□ Documented what changed and why",
            "□ Verified no breaking changes"
        ]
        
    def _identify_emergency_contacts(self, deep_understanding: Dict[str, Any]) -> Dict[str, str]:
        """Identify who to contact for different issues (from code comments/docs)"""
        
        # This would parse comments for MAINTAINER, OWNER, etc.
        # For now, return guidance
        return {
            "general": "Check README.md or CONTRIBUTING.md for maintainer info",
            "security_issues": "CRITICAL - Notify security team immediately",
            "breaking_changes": "Require approval from tech lead/architect",
            "business_logic": "Consult product owner for requirement clarification",
            "emergency_process": "1. Stop changes, 2. Document issue, 3. Seek approval"
        }
        
    def export_for_ai(self, knowledge_package: Dict[str, Any]) -> str:
        """Export the knowledge package in a format optimized for AI consumption"""
        
        # Create a structured output that AI can parse quickly
        export = {
            "!PRIORITY_READ": {
                "instant_context": knowledge_package['instant_context'],
                "danger_zones_summary": knowledge_package['danger_zones']['summary'],
                "top_dangers": knowledge_package['danger_zones']['do_not_modify'][:5]
            },
            "MODIFICATION_GUIDE": knowledge_package['safe_modification_guide'],
            "BUSINESS_CONTEXT": {
                "what_this_does": knowledge_package['business_context']['executive_summary'],
                "key_rules": knowledge_package['business_context']['critical_business_rules'][:5]
            },
            "TECHNICAL_CONTEXT": knowledge_package['system_understanding'],
            "AI_INSTRUCTIONS": knowledge_package['ai_instructions'],
            "QUICK_REFERENCE": knowledge_package['quick_reference']
        }
        
        return json.dumps(export, indent=2)
        
    def generate_conversation_starter(self, knowledge_package: Dict[str, Any]) -> str:
        """Generate a message that can be used to start conversations with AI"""
        
        instant_context = knowledge_package['instant_context']
        danger_summary = knowledge_package['danger_zones']['summary']
        quick_ref = knowledge_package['quick_reference']
        
        starter = f"""
I'm working with a {quick_ref['codebase_type']} codebase. Here's what you need to know:

{instant_context}

CRITICAL: {danger_summary}
- {quick_ref['danger_files_count']} files are high-risk
- {len(quick_ref['main_entities'])} main business entities: {', '.join(quick_ref['main_entities'][:3])}
- Key features: {', '.join(quick_ref['key_features'][:3])}

Before ANY modifications, you MUST check the danger_zones I've provided.
"""
        
        return starter