#!/usr/bin/env python3
"""
Token Manager for CodebaseIQ Pro
Handles accurate token counting and output size management to stay within MCP limits.
Uses tiktoken for OpenAI-compatible token counting.
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class TokenBudget:
    """Token budget allocation for different sections"""
    dependency: int = 7000
    security: int = 1000
    architecture: int = 3000
    business_logic: int = 5000
    technical_stack: int = 4000
    code_intelligence: int = 5000
    
    @property
    def total(self) -> int:
        """Total token budget"""
        return (self.dependency + self.security + self.architecture + 
                self.business_logic + self.technical_stack + self.code_intelligence)

class TokenManager:
    """Manages token counting and content truncation to stay within limits"""
    
    # MCP server token limit
    MCP_TOKEN_LIMIT = 25000
    
    # Default encoding for modern models
    DEFAULT_ENCODING = "cl100k_base"
    
    def __init__(self, encoding_name: str = None):
        """Initialize token manager with specified encoding"""
        self.encoding_name = encoding_name or self.DEFAULT_ENCODING
        try:
            self.encoding = tiktoken.get_encoding(self.encoding_name)
            logger.info(f"Initialized TokenManager with {self.encoding_name} encoding")
        except Exception as e:
            logger.error(f"Failed to initialize tiktoken encoding: {e}")
            # Fallback to character-based estimation
            self.encoding = None
            
    def count_tokens(self, text: Union[str, Dict, List]) -> int:
        """Count tokens in text, dict, or list"""
        if isinstance(text, (dict, list)):
            text = json.dumps(text, separators=(',', ':'))
            
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed, using fallback: {e}")
                
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4
        
    def truncate_to_tokens(self, content: Union[str, Dict, List], max_tokens: int) -> Union[str, Dict, List]:
        """Truncate content to fit within token limit"""
        original_type = type(content)
        
        # Convert to string for processing
        if isinstance(content, (dict, list)):
            text = json.dumps(content, indent=2)
        else:
            text = str(content)
            
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return content
            
        # Need to truncate
        logger.warning(f"Truncating content from {current_tokens} to {max_tokens} tokens")
        
        if isinstance(content, dict):
            return self._truncate_dict(content, max_tokens)
        elif isinstance(content, list):
            return self._truncate_list(content, max_tokens)
        else:
            return self._truncate_string(text, max_tokens)
            
    def _truncate_dict(self, data: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Intelligently truncate dictionary to fit token limit"""
        # Priority order for keeping data
        priority_keys = [
            'summary', 'instant_context', 'danger_zones', 'critical_files',
            'risk_level', 'total_files', 'languages', 'key_features',
            'entry_points', 'dependencies', 'vulnerabilities'
        ]
        
        result = {}
        current_tokens = 0
        
        # First pass: add priority keys
        for key in priority_keys:
            if key in data:
                key_tokens = self.count_tokens({key: data[key]})
                if current_tokens + key_tokens <= max_tokens:
                    result[key] = data[key]
                    current_tokens += key_tokens
                    
        # Second pass: add remaining keys if space allows
        for key, value in data.items():
            if key not in result:
                key_tokens = self.count_tokens({key: value})
                if current_tokens + key_tokens <= max_tokens * 0.9:  # Leave 10% buffer
                    result[key] = value
                    current_tokens += key_tokens
                    
        # If still over, truncate values
        if current_tokens > max_tokens:
            for key in list(result.keys()):
                if isinstance(result[key], (list, dict, str)) and len(str(result[key])) > 100:
                    if isinstance(result[key], list):
                        result[key] = result[key][:5]  # Keep first 5 items
                    elif isinstance(result[key], dict):
                        result[key] = dict(list(result[key].items())[:5])
                    elif isinstance(result[key], str) and len(result[key]) > 200:
                        result[key] = result[key][:200] + "..."
                        
                if self.count_tokens(result) <= max_tokens:
                    break
                    
        return result
        
    def _truncate_list(self, data: List[Any], max_tokens: int) -> List[Any]:
        """Truncate list to fit token limit"""
        result = []
        current_tokens = 2  # For brackets
        
        for item in data:
            item_tokens = self.count_tokens(item)
            if current_tokens + item_tokens <= max_tokens * 0.9:
                result.append(item)
                current_tokens += item_tokens
            else:
                # Add truncation indicator
                if len(result) < len(data):
                    result.append(f"... and {len(data) - len(result)} more items")
                break
                
        return result
        
    def _truncate_string(self, text: str, max_tokens: int) -> str:
        """Truncate string to fit token limit"""
        if self.encoding:
            tokens = self.encoding.encode(text)
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens - 10]  # Leave room for ellipsis
                return self.encoding.decode(truncated_tokens) + "\n... (truncated)"
        else:
            # Fallback: character-based truncation
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                return text[:max_chars - 20] + "\n... (truncated)"
                
        return text
        
    def distribute_tokens(self, data: Dict[str, Any], budget: TokenBudget) -> Dict[str, Any]:
        """Distribute content across sections according to token budget"""
        result = {}
        
        # Map section names to budget allocations
        section_budgets = {
            'dependency_analysis': budget.dependency,
            'security_analysis': budget.security,
            'architecture_analysis': budget.architecture,
            'business_logic': budget.business_logic,
            'technical_stack': budget.technical_stack,
            'code_intelligence': budget.code_intelligence
        }
        
        for section, max_tokens in section_budgets.items():
            if section in data:
                result[section] = self.truncate_to_tokens(data[section], max_tokens)
                actual_tokens = self.count_tokens(result[section])
                logger.info(f"{section}: {actual_tokens}/{max_tokens} tokens")
                
        return result
        
    def validate_output_size(self, content: Union[str, Dict, List]) -> Tuple[bool, int]:
        """Validate if content fits within MCP token limit"""
        tokens = self.count_tokens(content)
        is_valid = tokens <= self.MCP_TOKEN_LIMIT
        
        if not is_valid:
            logger.warning(f"Content exceeds MCP limit: {tokens}/{self.MCP_TOKEN_LIMIT} tokens")
            
        return is_valid, tokens
        
    def get_token_summary(self, content: Union[str, Dict, List]) -> Dict[str, Any]:
        """Get detailed token usage summary"""
        tokens = self.count_tokens(content)
        
        return {
            'total_tokens': tokens,
            'mcp_limit': self.MCP_TOKEN_LIMIT,
            'percentage_used': round((tokens / self.MCP_TOKEN_LIMIT) * 100, 2),
            'tokens_remaining': self.MCP_TOKEN_LIMIT - tokens,
            'is_within_limit': tokens <= self.MCP_TOKEN_LIMIT,
            'encoding': self.encoding_name
        }