#!/usr/bin/env python3
"""
Cache Service Implementations for CodebaseIQ Pro
Supports in-memory cache (default) and Redis (optional)
"""

import asyncio
import logging
import json
import pickle
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib

logger = logging.getLogger(__name__)

class CacheService(ABC):
    """Abstract base class for cache services"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL in seconds"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass

class MemoryCache(CacheService):
    """In-memory cache implementation (DEFAULT - Free)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_size = config.get('max_size', 10000)
        self.default_ttl = config.get('ttl', 3600)
        self._cache = OrderedDict()
        self._expiry_times = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        logger.info(f"Initialized in-memory cache with max_size: {self.max_size}")
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Check if key exists and not expired
        if key in self._cache:
            # Check expiry
            if key in self._expiry_times:
                if datetime.utcnow() > self._expiry_times[key]:
                    # Expired, remove it
                    await self.delete(key)
                    self._misses += 1
                    return None
                    
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
            
        self._misses += 1
        return None
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL"""
        # Check if we need to evict
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Evict least recently used
            oldest_key = next(iter(self._cache))
            await self.delete(oldest_key)
            self._evictions += 1
            
        # Set value
        self._cache[key] = value
        self._cache.move_to_end(key)
        
        # Set expiry
        if ttl is None:
            ttl = self.default_ttl
        if ttl > 0:
            self._expiry_times[key] = datetime.utcnow() + timedelta(seconds=ttl)
            
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        if key in self._cache:
            del self._cache[key]
        if key in self._expiry_times:
            del self._expiry_times[key]
            
    async def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        self._expiry_times.clear()
        logger.info("Cleared in-memory cache")
        
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if key not in self._cache:
            return False
            
        # Check expiry
        if key in self._expiry_times:
            if datetime.utcnow() > self._expiry_times[key]:
                await self.delete(key)
                return False
                
        return True
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        return {
            'type': 'memory',
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': round(hit_rate, 3),
            'evictions': self._evictions,
            'expired_keys': len([k for k, v in self._expiry_times.items() 
                               if datetime.utcnow() > v])
        }

class RedisCache(CacheService):
    """Redis cache implementation (OPTIONAL - Distributed)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url = config['url']
        self.default_ttl = config.get('ttl', 3600)
        self.max_connections = config.get('max_connections', 10)
        self.client = None
        self._hits = 0
        self._misses = 0
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Redis client"""
        try:
            import redis.asyncio as redis
            
            # Parse Redis URL and create connection pool
            self.client = redis.from_url(
                self.url,
                max_connections=self.max_connections,
                decode_responses=False  # We'll handle encoding/decoding
            )
            logger.info(f"Initialized Redis cache at: {self.url}")
            
        except ImportError:
            logger.error("Redis client not installed. Run: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            # Get raw bytes from Redis
            data = await self.client.get(key)
            
            if data is None:
                self._misses += 1
                return None
                
            # Deserialize
            try:
                value = pickle.loads(data)
                self._hits += 1
                return value
            except:
                # Try JSON fallback
                try:
                    value = json.loads(data.decode('utf-8'))
                    self._hits += 1
                    return value
                except:
                    logger.error(f"Failed to deserialize cache value for key: {key}")
                    self._misses += 1
                    return None
                    
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._misses += 1
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis with optional TTL"""
        try:
            # Serialize value
            try:
                # Try pickle for complex objects
                data = pickle.dumps(value)
            except:
                # Fallback to JSON for simple objects
                data = json.dumps(value).encode('utf-8')
                
            # Set with TTL
            if ttl is None:
                ttl = self.default_ttl
                
            if ttl > 0:
                await self.client.setex(key, ttl, data)
            else:
                await self.client.set(key, data)
                
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            
    async def delete(self, key: str) -> None:
        """Delete value from Redis"""
        try:
            await self.client.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            
    async def clear(self) -> None:
        """Clear all cache entries (use pattern matching)"""
        try:
            # Use SCAN to find all keys (safer than KEYS for production)
            cursor = 0
            while True:
                cursor, keys = await self.client.scan(cursor, count=100)
                if keys:
                    await self.client.delete(*keys)
                if cursor == 0:
                    break
                    
            logger.info("Cleared Redis cache")
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        try:
            # Get Redis info
            info = await self.client.info()
            
            # Get database size
            db_size = await self.client.dbsize()
            
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'type': 'redis',
                'size': db_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 3),
                'used_memory': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'redis_version': info.get('redis_version', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {
                'type': 'redis',
                'error': str(e)
            }

# Cache key generation utilities
def generate_cache_key(*args) -> str:
    """Generate a cache key from arguments"""
    key_data = ':'.join(str(arg) for arg in args)
    return hashlib.md5(key_data.encode()).hexdigest()

def cache_decorator(ttl: Optional[int] = None):
    """Decorator for caching function results"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Get cache service from self if available
            cache = getattr(self, 'cache', None)
            if not cache:
                return await func(self, *args, **kwargs)
                
            # Generate cache key
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            result = await cache.get(cache_key)
            if result is not None:
                return result
                
            # Call function
            result = await func(self, *args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl)
            
            return result
            
        return wrapper
    return decorator

# Factory function
def create_cache_service(config: Dict[str, Any]) -> CacheService:
    """Create the appropriate cache service based on configuration"""
    cache_type = config.get('type', 'memory')
    
    if cache_type == 'redis':
        try:
            return RedisCache(config)
        except Exception as e:
            logger.warning(f"Failed to create Redis cache, falling back to memory: {e}")
            return MemoryCache(config)
    else:
        return MemoryCache(config)