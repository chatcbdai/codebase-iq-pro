#!/usr/bin/env python3
"""
Embedding Service Implementations for CodebaseIQ Pro
Supports OpenAI (default/required) and Voyage AI (premium)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import hashlib
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class EmbeddingService(ABC):
    """Abstract base class for embedding services"""
    
    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        pass
    
    @abstractmethod
    async def embed_code_entity(self, entity: Any) -> np.ndarray:
        """Generate embedding for a code entity with rich context"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        pass

class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embedding service (DEFAULT - Required)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config['api_key']
        self.model = config.get('model', 'text-embedding-3-small')
        self.dimension = config.get('dimension', 1536)
        self.batch_size = config.get('batch_size', 100)
        self.max_retries = config.get('max_retries', 3)
        self._cache = {}  # Simple in-memory cache
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI embedding service with model: {self.model}")
        except ImportError:
            logger.error("OpenAI client not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
            
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        # Check cache
        cache_key = hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            # Make API call
            response = await self._make_embedding_request([text])
            embedding = np.array(response['data'][0]['embedding'])
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            # Cache result
            self._cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embedding: {e}")
            raise
            
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = await self._make_embedding_request(batch)
                
                for item in response['data']:
                    embedding = np.array(item['embedding'])
                    # Normalize
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                    
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                # Return partial results or raise
                raise
                
        return embeddings
        
    async def _make_embedding_request(self, texts: List[str], retry_count: int = 0) -> Dict:
        """Make embedding request with retry logic"""
        try:
            # For OpenAI v1.0+
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return response.model_dump()
            
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Embedding request failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                return await self._make_embedding_request(texts, retry_count + 1)
            else:
                raise
                
    async def embed_code_entity(self, entity: Any) -> np.ndarray:
        """Generate embedding for a code entity with rich context"""
        # Create comprehensive context for code
        context_parts = [
            f"Type: {entity.type}",
            f"Name: {entity.name}",
            f"Path: {entity.path}",
        ]
        
        # Add code-specific context
        if hasattr(entity, 'symbols') and entity.symbols:
            symbol_names = [s.get('name', '') for s in entity.symbols[:10]]
            context_parts.append(f"Symbols: {', '.join(symbol_names)}")
            
        if hasattr(entity, 'dependencies') and entity.dependencies:
            dep_list = list(entity.dependencies)[:5]
            context_parts.append(f"Dependencies: {', '.join(dep_list)}")
            
        if hasattr(entity, 'docstring') and entity.docstring:
            # Include first 200 chars of docstring
            doc_preview = entity.docstring[:200].replace('\n', ' ')
            context_parts.append(f"Documentation: {doc_preview}")
            
        # Add code snippet if available
        if hasattr(entity, 'code_snippet') and entity.code_snippet:
            # Include first 300 chars of code
            code_preview = entity.code_snippet[:300].replace('\n', ' ')
            context_parts.append(f"Code: {code_preview}")
            
        context = "\n".join(context_parts)
        return await self.embed_text(context)
        
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimension

class VoyageEmbeddingService(EmbeddingService):
    """Voyage AI embedding service (PREMIUM - Optimized for code)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config['api_key']
        self.model = config.get('model', 'voyage-code-2')  # Best for code
        self.dimension = config.get('dimension', 1536)
        self.batch_size = config.get('batch_size', 128)
        self.max_retries = config.get('max_retries', 3)
        self._cache = {}
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Voyage client"""
        try:
            import voyageai
            self.client = voyageai.Client(api_key=self.api_key)
            logger.info(f"Initialized Voyage AI embedding service with model: {self.model}")
        except ImportError:
            logger.error("Voyage AI client not installed. Run: pip install voyageai")
            raise
            
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using Voyage AI"""
        # Check cache
        cache_key = hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            # Voyage AI expects list input
            response = await self._make_embedding_request([text])
            embedding = np.array(response.embeddings[0])
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            # Cache result
            self._cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate Voyage embedding: {e}")
            raise
            
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = await self._make_embedding_request(batch)
                
                for embedding_data in response.embeddings:
                    embedding = np.array(embedding_data)
                    # Normalize
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                    
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                raise
                
        return embeddings
        
    async def _make_embedding_request(self, texts: List[str], retry_count: int = 0):
        """Make embedding request to Voyage AI with retry logic"""
        try:
            # Voyage AI specific request
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="document"  # Optimized for code documentation
            )
            return response
            
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                logger.warning(f"Voyage request failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                return await self._make_embedding_request(texts, retry_count + 1)
            else:
                raise
                
    async def embed_code_entity(self, entity: Any) -> np.ndarray:
        """Generate code-optimized embedding using Voyage AI"""
        # Voyage AI has special handling for code
        context_parts = []
        
        # Code structure information
        context_parts.append(f"[{entity.type.upper()}] {entity.name}")
        
        # File path provides important context
        context_parts.append(f"File: {entity.path}")
        
        # Function/class signatures are crucial
        if hasattr(entity, 'signature') and entity.signature:
            context_parts.append(f"Signature: {entity.signature}")
            
        # Include symbols with type information
        if hasattr(entity, 'symbols') and entity.symbols:
            for symbol in entity.symbols[:10]:
                sym_type = symbol.get('type', 'unknown')
                sym_name = symbol.get('name', '')
                context_parts.append(f"{sym_type}: {sym_name}")
                
        # Dependencies are important for code understanding
        if hasattr(entity, 'dependencies') and entity.dependencies:
            context_parts.append(f"Imports: {', '.join(list(entity.dependencies)[:10])}")
            
        # Documentation
        if hasattr(entity, 'docstring') and entity.docstring:
            doc_lines = entity.docstring.split('\n')[:5]  # First 5 lines
            context_parts.extend(doc_lines)
            
        # Code snippet - Voyage handles code well
        if hasattr(entity, 'code_snippet') and entity.code_snippet:
            # Include more code for Voyage (optimized for code)
            code_preview = entity.code_snippet[:500]
            context_parts.append("```")
            context_parts.append(code_preview)
            context_parts.append("```")
            
        context = "\n".join(context_parts)
        
        # Use special code embedding
        try:
            response = self.client.embed(
                texts=[context],
                model=self.model,
                input_type="code"  # Special handling for code
            )
            embedding = np.array(response.embeddings[0])
            return embedding / np.linalg.norm(embedding)
            
        except:
            # Fallback to regular embedding
            return await self.embed_text(context)
            
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimension

# Factory function
def create_embedding_service(config: Dict[str, Any]) -> EmbeddingService:
    """Create the appropriate embedding service based on configuration"""
    service_type = config.get('service', 'openai')
    
    if service_type == 'voyage':
        return VoyageEmbeddingService(config)
    else:
        return OpenAIEmbeddingService(config)