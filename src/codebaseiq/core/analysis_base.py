#!/usr/bin/env python3
"""
Base classes and common structures for CodebaseIQ Pro
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# Re-export from main file to avoid circular imports
class AgentRole(Enum):
    """Specialized agent roles"""
    DEPENDENCY = "dependency"
    SECURITY = "security"
    PATTERN = "pattern"
    VERSION = "version"
    ARCHITECTURE = "architecture"
    PERFORMANCE = "performance"
    EMBEDDING = "embedding"
    DOCUMENTATION = "documentation"
    TEST_COVERAGE = "test_coverage"
    REFACTORING = "refactoring"
    COMPLIANCE = "compliance"

@dataclass
class EnhancedCodeEntity:
    """Enhanced code entity with embedding support"""
    path: str
    type: str
    name: str
    importance_score: float = 0.0
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    ast_signature: Optional[str] = None
    semantic_hash: Optional[str] = None
    danger_level: int = 0
    version_constraints: Dict[str, str] = field(default_factory=dict)
    
    # Enhanced fields
    embedding: Optional[np.ndarray] = None
    embedding_id: Optional[str] = None
    symbols: List[Dict[str, Any]] = field(default_factory=list)
    test_coverage: Optional[float] = None
    documentation_score: float = 0.0
    last_modified: Optional[datetime] = None
    semantic_neighbors: List[str] = field(default_factory=list)
    code_snippet: Optional[str] = None
    docstring: Optional[str] = None
    signature: Optional[str] = None

@dataclass
class AgentMessage:
    """Message format for inter-agent communication"""
    sender: str
    receiver: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None

class BaseAgent(ABC):
    """Base class for all analysis agents"""
    
    def __init__(self, role: AgentRole):
        self.role = role
        self.message_queue = asyncio.Queue()
        self.results = {}
        self.logger = logging.getLogger(f"agent.{role.value}")
        
    @abstractmethod
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform agent-specific analysis"""
        pass
        
    async def send_message(self, receiver: str, message_type: str, payload: Dict[str, Any]):
        """Send message to another agent"""
        message = AgentMessage(
            sender=self.role.value,
            receiver=receiver,
            message_type=message_type,
            payload=payload
        )
        self.logger.debug(f"Sending message to {receiver}: {message_type}")
        # In production, use proper message broker
        
    async def receive_messages(self):
        """Process incoming messages"""
        while not self.message_queue.empty():
            message = await self.message_queue.get()
            await self._handle_message(message)
            
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming message"""
        self.logger.debug(f"Received message: {message.message_type} from {message.sender}")
        
    async def health_check(self) -> 'HealthStatus':
        """Check agent health"""
        return HealthStatus(
            is_healthy=True,
            reason="OK",
            timestamp=datetime.utcnow()
        )

@dataclass
class HealthStatus:
    """Health status for agents"""
    is_healthy: bool
    reason: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)