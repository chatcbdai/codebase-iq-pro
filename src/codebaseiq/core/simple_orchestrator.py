#!/usr/bin/env python3
"""
Simplified Orchestrator for CodebaseIQ Pro
No external dependencies required - pure Python implementation
"""

import asyncio
import logging
from typing import Dict, List, Any, Set, Optional
from datetime import datetime
from collections import defaultdict
import networkx as nx

from .analysis_base import BaseAgent, AgentRole

logger = logging.getLogger(__name__)

class SimpleOrchestrator:
    """
    Simplified orchestration engine without LangGraph dependency.
    Uses asyncio for parallel execution and networkx for execution planning.
    """
    
    def __init__(self):
        self.agents: Dict[AgentRole, BaseAgent] = {}
        self.execution_plans = {
            'full': self._build_full_plan,
            'security_focus': self._build_security_plan,
            'performance_focus': self._build_performance_plan,
            'quick': self._build_quick_plan,
        }
        self.results = {}
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.role] = agent
        logger.info(f"Registered agent: {agent.role.value}")
        
    def _build_full_plan(self) -> Dict[str, List[str]]:
        """Build execution plan for full analysis"""
        return {
            'parallel_groups': [
                # Group 1: Initial analysis (can run in parallel)
                [AgentRole.DEPENDENCY, AgentRole.SECURITY, AgentRole.ARCHITECTURE],
                # Group 2: Secondary analysis (depends on group 1)
                [AgentRole.VERSION, AgentRole.PATTERN, AgentRole.PERFORMANCE],
                # Group 3: Documentation and testing
                [AgentRole.DOCUMENTATION, AgentRole.TEST_COVERAGE],
                # Group 4: Embeddings (needs all previous data)
                [AgentRole.EMBEDDING]
            ]
        }
        
    def _build_security_plan(self) -> Dict[str, List[str]]:
        """Build execution plan for security-focused analysis"""
        return {
            'parallel_groups': [
                [AgentRole.SECURITY],
                [AgentRole.DEPENDENCY, AgentRole.VERSION],
                [AgentRole.PATTERN]  # Look for security anti-patterns
            ]
        }
        
    def _build_performance_plan(self) -> Dict[str, List[str]]:
        """Build execution plan for performance-focused analysis"""
        return {
            'parallel_groups': [
                [AgentRole.PERFORMANCE, AgentRole.ARCHITECTURE],
                [AgentRole.PATTERN],  # Look for performance anti-patterns
                [AgentRole.DEPENDENCY]  # Check for heavy dependencies
            ]
        }
        
    def _build_quick_plan(self) -> Dict[str, List[str]]:
        """Build execution plan for quick analysis"""
        return {
            'parallel_groups': [
                [AgentRole.DEPENDENCY, AgentRole.ARCHITECTURE],
                [AgentRole.DOCUMENTATION]
            ]
        }
        
    async def execute(self, context: Dict[str, Any], analysis_type: str = "full") -> Dict[str, Any]:
        """Execute the orchestrated analysis"""
        start_time = datetime.utcnow()
        
        # Get execution plan
        if analysis_type not in self.execution_plans:
            logger.warning(f"Unknown analysis type: {analysis_type}, falling back to 'full'")
            analysis_type = 'full'
            
        plan = self.execution_plans[analysis_type]()
        parallel_groups = plan['parallel_groups']
        
        # Initialize results
        results = {
            'analysis_type': analysis_type,
            'agents_executed': [],
            'execution_time': {},
            'agent_results': {},
            'errors': []
        }
        
        # Execute groups in sequence, agents within groups in parallel
        for group_idx, agent_group in enumerate(parallel_groups):
            logger.info(f"Executing group {group_idx + 1}/{len(parallel_groups)}: {[a.value for a in agent_group]}")
            
            # Create tasks for parallel execution
            tasks = []
            for agent_role in agent_group:
                if agent_role not in self.agents:
                    logger.warning(f"Agent {agent_role.value} not registered, skipping")
                    continue
                    
                agent = self.agents[agent_role]
                
                # Pass context with previous results
                agent_context = {
                    **context,
                    'previous_results': results['agent_results'].copy(),
                    'execution_group': group_idx
                }
                
                # Create task
                task = asyncio.create_task(
                    self._execute_agent(agent, agent_context)
                )
                tasks.append((agent_role, task))
                
            # Wait for all tasks in group to complete
            for agent_role, task in tasks:
                try:
                    agent_start = datetime.utcnow()
                    result = await task
                    agent_end = datetime.utcnow()
                    
                    # Store results
                    results['agent_results'][agent_role.value] = result
                    results['agents_executed'].append(agent_role.value)
                    results['execution_time'][agent_role.value] = (agent_end - agent_start).total_seconds()
                    
                    logger.info(f"Agent {agent_role.value} completed in {results['execution_time'][agent_role.value]:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Agent {agent_role.value} failed: {e}")
                    results['errors'].append({
                        'agent': agent_role.value,
                        'error': str(e),
                        'group': group_idx
                    })
                    results['agent_results'][agent_role.value] = {'error': str(e)}
                    
        # Calculate total execution time
        total_time = (datetime.utcnow() - start_time).total_seconds()
        results['total_execution_time'] = total_time
        
        # Add summary
        results['summary'] = self._generate_summary(results)
        
        logger.info(f"Orchestration completed in {total_time:.2f}s")
        return results
        
    async def _execute_agent(self, agent: BaseAgent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent with timeout protection"""
        timeout = 60  # 60 second timeout per agent
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.analyze(context),
                timeout=timeout
            )
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Agent {agent.role.value} timed out after {timeout}s")
            raise TimeoutError(f"Agent execution timed out after {timeout}s")
            
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the orchestration results"""
        agent_results = results.get('agent_results', {})
        
        summary = {
            'total_agents': len(results['agents_executed']),
            'successful_agents': len([a for a in results['agents_executed'] if a not in [e['agent'] for e in results['errors']]]),
            'failed_agents': len(results['errors']),
            'total_time': results.get('total_execution_time', 0)
        }
        
        # Add key metrics from agents
        if 'security' in agent_results:
            summary['security_score'] = agent_results['security'].get('security_score', 'N/A')
            
        if 'dependency' in agent_results:
            deps = agent_results['dependency']
            summary['external_dependencies'] = len(deps.get('external_dependencies', {}))
            
        if 'architecture' in agent_results:
            arch = agent_results['architecture']
            summary['architecture_style'] = arch.get('architecture_style', 'unknown')
            
        if 'pattern' in agent_results:
            patterns = agent_results['pattern']
            summary['code_quality_score'] = patterns.get('code_quality_score', 'N/A')
            
        if 'performance' in agent_results:
            perf = agent_results['performance']
            summary['performance_score'] = perf.get('performance_score', 'N/A')
            
        if 'documentation' in agent_results:
            docs = agent_results['documentation']
            summary['documentation_score'] = docs.get('average_doc_score', 'N/A')
            
        if 'test_coverage' in agent_results:
            tests = agent_results['test_coverage']
            summary['test_coverage'] = tests.get('coverage_percentage', 'N/A')
            
        return summary
        
    def get_agent_dependencies(self, agent_role: AgentRole) -> List[AgentRole]:
        """Get dependencies for a specific agent (which agents must run before it)"""
        # Define agent dependencies
        dependencies = {
            AgentRole.EMBEDDING: [AgentRole.DEPENDENCY, AgentRole.ARCHITECTURE, AgentRole.DOCUMENTATION],
            AgentRole.VERSION: [AgentRole.DEPENDENCY],
            AgentRole.PATTERN: [AgentRole.ARCHITECTURE],
            # Most agents can run independently
        }
        
        return dependencies.get(agent_role, [])