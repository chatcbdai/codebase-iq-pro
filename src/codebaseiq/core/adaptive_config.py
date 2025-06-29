#!/usr/bin/env python3
"""
Adaptive Configuration System for CodebaseIQ Pro
Automatically detects available services and adapts to use free or premium options
"""

import os
import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class ServiceTier(Enum):
    """Service tier levels"""
    FREE = "free"
    PREMIUM = "premium"

class AdaptiveConfig:
    """
    Smart configuration that adapts based on available API keys and services.
    Falls back to free/local options when premium services are not available.
    """
    
    def __init__(self):
        self._detect_available_services()
        
    def _detect_available_services(self):
        """Detect which services are available based on environment variables"""
        # Essential API keys (OpenAI is mandatory)
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for CodebaseIQ Pro to function")
            
        # Premium API keys (optional)
        self.voyage_api_key = os.getenv('VOYAGE_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        
        # Vector Database Configuration
        self.vector_db_config = self._configure_vector_db()
        
        # Embedding Service Configuration
        self.embedding_config = self._configure_embedding_service()
        
        # Caching Configuration
        self.cache_config = self._configure_cache()
        
        # Security Configuration
        self.security_config = self._configure_security()
        
        # Performance Configuration
        self.performance_config = self._configure_performance()
        
        # Log configuration summary
        self._log_configuration_summary()
        
    def _configure_vector_db(self) -> Dict[str, Any]:
        """Configure vector database based on available services"""
        if self.pinecone_api_key:
            logger.info("✨ Premium: Using Pinecone vector database")
            return {
                'type': 'pinecone',
                'tier': ServiceTier.PREMIUM,
                'api_key': self.pinecone_api_key,
                'environment': self.pinecone_environment,
                'index_name': 'codebase-iq-pro',
                'dimension': 1536,  # OpenAI embedding dimension
                'metric': 'cosine'
            }
        else:
            logger.info("🆓 Free: Using Qdrant local vector database")
            return {
                'type': 'qdrant',
                'tier': ServiceTier.FREE,
                'path': './qdrant_storage',  # Local storage
                'collection_name': 'codebase_embeddings',
                'dimension': 1536,  # OpenAI embedding dimension
                'on_disk': True,  # Store on disk for persistence
                'force_disable_check': True  # Disable version check for local
            }
            
    def _configure_embedding_service(self) -> Dict[str, Any]:
        """Configure embedding service based on available API keys"""
        if self.voyage_api_key:
            logger.info("✨ Premium: Using Voyage AI embeddings (optimized for code)")
            return {
                'service': 'voyage',
                'tier': ServiceTier.PREMIUM,
                'api_key': self.voyage_api_key,
                'model': 'voyage-code-2',  # Best model for code
                'dimension': 1536,
                'batch_size': 128,
                'max_retries': 3
            }
        else:
            logger.info("🆓 Standard: Using OpenAI embeddings")
            return {
                'service': 'openai',
                'tier': ServiceTier.FREE,  # Included with OpenAI API key
                'api_key': self.openai_api_key,
                'model': 'text-embedding-3-small',  # Cost-effective model
                'dimension': 1536,
                'batch_size': 100,
                'max_retries': 3
            }
            
    def _configure_cache(self) -> Dict[str, Any]:
        """Configure caching based on available services"""
        redis_url = os.getenv('REDIS_URL')
        
        if redis_url:
            # Try to import redis to check if it's installed
            try:
                import redis
                logger.info("🚀 Using Redis for distributed caching")
                return {
                    'type': 'redis',
                    'url': redis_url,
                    'ttl': int(os.getenv('CACHE_TTL', '3600')),
                    'max_connections': 10
                }
            except ImportError:
                logger.warning("Redis URL provided but redis package not installed")
                
        logger.info("🆓 Using in-memory dictionary cache")
        return {
            'type': 'memory',
            'max_size': 10000,  # Maximum number of cached items
            'ttl': 3600  # 1 hour TTL
        }
        
    def _configure_security(self) -> Dict[str, Any]:
        """Configure security settings"""
        vault_url = os.getenv('VAULT_URL')
        
        if vault_url and os.getenv('VAULT_TOKEN'):
            logger.info("🔒 Premium: Using HashiCorp Vault for secrets")
            return {
                'credential_storage': 'vault',
                'vault_url': vault_url,
                'vault_token': os.getenv('VAULT_TOKEN'),
                'enable_audit': True
            }
        else:
            logger.info("🔒 Standard: Using environment variables for secrets")
            return {
                'credential_storage': 'env',
                'enable_audit': True,
                'audit_file': './audit.log'
            }
            
    def _configure_performance(self) -> Dict[str, Any]:
        """Configure performance settings based on system capabilities"""
        import multiprocessing
        
        cpu_count = multiprocessing.cpu_count()
        
        # Adaptive worker configuration
        if cpu_count >= 8:
            max_workers = min(cpu_count - 2, 16)  # Leave 2 cores free
            batch_size = 200
            logger.info(f"🚀 High performance mode: {max_workers} workers")
        elif cpu_count >= 4:
            max_workers = cpu_count - 1
            batch_size = 100
            logger.info(f"⚡ Standard performance mode: {max_workers} workers")
        else:
            max_workers = 2
            batch_size = 50
            logger.info(f"🔋 Low resource mode: {max_workers} workers")
            
        return {
            'max_workers': max_workers,
            'batch_size': batch_size,
            'max_file_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', '10')),
            'analysis_timeout': int(os.getenv('ANALYSIS_TIMEOUT', '300')),
            'enable_profiling': os.getenv('ENABLE_PROFILING', 'false').lower() == 'true'
        }
        
    def _log_configuration_summary(self):
        """Log a summary of the configuration"""
        logger.info("=" * 60)
        logger.info("CodebaseIQ Pro Configuration Summary:")
        logger.info("=" * 60)
        
        # Vector DB
        vector_tier = "Premium (Pinecone)" if self.vector_db_config['tier'] == ServiceTier.PREMIUM else "Free (Qdrant)"
        logger.info(f"Vector Database: {vector_tier}")
        
        # Embeddings
        embed_tier = "Premium (Voyage AI)" if self.embedding_config['tier'] == ServiceTier.PREMIUM else "Standard (OpenAI)"
        logger.info(f"Embeddings: {embed_tier}")
        
        # Cache
        cache_type = self.cache_config['type'].capitalize()
        logger.info(f"Caching: {cache_type}")
        
        # Performance
        logger.info(f"Workers: {self.performance_config['max_workers']}")
        logger.info(f"Batch Size: {self.performance_config['batch_size']}")
        
        logger.info("=" * 60)
        
    def get_vector_db_client(self):
        """Get the appropriate vector database client"""
        if self.vector_db_config['type'] == 'pinecone':
            from vector_db import PineconeVectorDB
            return PineconeVectorDB(self.vector_db_config)
        else:
            from vector_db import QdrantVectorDB
            return QdrantVectorDB(self.vector_db_config)
            
    def get_embedding_service(self):
        """Get the appropriate embedding service"""
        if self.embedding_config['service'] == 'voyage':
            from embedding_service import VoyageEmbeddingService
            return VoyageEmbeddingService(self.embedding_config)
        else:
            from embedding_service import OpenAIEmbeddingService
            return OpenAIEmbeddingService(self.embedding_config)
            
    def get_cache_service(self):
        """Get the appropriate cache service"""
        if self.cache_config['type'] == 'redis':
            from cache_service import RedisCache
            return RedisCache(self.cache_config)
        else:
            from cache_service import MemoryCache
            return MemoryCache(self.cache_config)
            
    def is_premium_feature_available(self, feature: str) -> bool:
        """Check if a premium feature is available"""
        premium_features = {
            'voyage_embeddings': self.voyage_api_key is not None,
            'pinecone_vector_db': self.pinecone_api_key is not None,
            'advanced_security': os.getenv('VAULT_URL') is not None,
            'distributed_cache': self.cache_config['type'] == 'redis'
        }
        return premium_features.get(feature, False)
        
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        return {
            'vector_db': {
                'type': self.vector_db_config['type'],
                'tier': self.vector_db_config['tier'].value
            },
            'embeddings': {
                'service': self.embedding_config['service'],
                'model': self.embedding_config['model'],
                'tier': self.embedding_config['tier'].value
            },
            'cache': {
                'type': self.cache_config['type']
            },
            'performance': {
                'max_workers': self.performance_config['max_workers'],
                'batch_size': self.performance_config['batch_size']
            },
            'premium_features': {
                'voyage_embeddings': self.is_premium_feature_available('voyage_embeddings'),
                'pinecone_vector_db': self.is_premium_feature_available('pinecone_vector_db'),
                'distributed_cache': self.is_premium_feature_available('distributed_cache')
            }
        }

# Singleton instance
_config_instance: Optional[AdaptiveConfig] = None

def get_config() -> AdaptiveConfig:
    """Get the singleton configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AdaptiveConfig()
    return _config_instance