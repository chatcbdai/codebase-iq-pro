#!/usr/bin/env python3
"""
Vector Database Implementations for CodebaseIQ Pro
Supports both Qdrant (free/local) and Pinecone (premium)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class VectorDatabase(ABC):
    """Abstract base class for vector database implementations"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector database connection"""
        pass
    
    @abstractmethod
    async def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """Insert or update vectors in the database"""
        pass
    
    @abstractmethod
    async def search(self, query_vector: np.ndarray, top_k: int = 10, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass

class QdrantVectorDB(VectorDatabase):
    """Qdrant vector database implementation (FREE option)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.path = Path(config['path'])
        self.collection_name = config['collection_name']
        self.dimension = config['dimension']
        self.client = None
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize Qdrant client and create collection if needed"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            # Create local storage directory
            self.path.mkdir(parents=True, exist_ok=True)
            
            # Initialize client with local storage
            self.client = QdrantClient(path=str(self.path))
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
            self.is_initialized = True
            
        except ImportError:
            logger.error("Qdrant client not installed. Run: pip install qdrant-client")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
            
    async def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """Insert or update vectors in Qdrant"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            from qdrant_client.models import PointStruct
            
            # Convert to Qdrant points
            points = []
            for vector_data in vectors:
                point = PointStruct(
                    id=vector_data['id'],
                    vector=vector_data['values'],
                    payload=vector_data.get('metadata', {})
                )
                points.append(point)
                
            # Batch upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.debug(f"Upserted {len(points)} vectors to Qdrant")
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors to Qdrant: {e}")
            raise
            
    async def search(self, query_vector: np.ndarray, top_k: int = 10, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                query_filter=filter  # Qdrant supports filtering
            )
            
            # Convert results to standard format
            results = []
            for hit in search_result:
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'metadata': hit.payload
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to search in Qdrant: {e}")
            return []
            
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors from Qdrant"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={"points": ids}
            )
            logger.debug(f"Deleted {len(ids)} vectors from Qdrant")
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Qdrant: {e}")
            raise
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get Qdrant collection statistics"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'vector_count': collection_info.vectors_count,
                'indexed_vectors': collection_info.indexed_vectors_count,
                'status': collection_info.status,
                'storage_path': str(self.path),
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get Qdrant stats: {e}")
            return {'error': str(e)}

class PineconeVectorDB(VectorDatabase):
    """Pinecone vector database implementation (PREMIUM option)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config['api_key']
        self.environment = config['environment']
        self.index_name = config['index_name']
        self.dimension = config['dimension']
        self.metric = config['metric']
        self.index = None
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize Pinecone client and create index if needed"""
        try:
            import pinecone
            
            # Initialize Pinecone
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            # Check if index exists
            existing_indexes = pinecone.list_indexes()
            
            if self.index_name not in existing_indexes:
                # Create index
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    pods=1,  # Start with 1 pod
                    pod_type='s1.x1'  # Starter pod type
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
                
                # Wait for index to be ready
                await self._wait_for_index_ready()
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
                
            # Connect to index
            self.index = pinecone.Index(self.index_name)
            self.is_initialized = True
            
        except ImportError:
            logger.error("Pinecone client not installed. Run: pip install pinecone-client")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
            
    async def _wait_for_index_ready(self, timeout: int = 60):
        """Wait for Pinecone index to be ready"""
        import pinecone
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                desc = pinecone.describe_index(self.index_name)
                if desc['status']['ready']:
                    return
            except:
                pass
            await asyncio.sleep(2)
            
        raise TimeoutError(f"Pinecone index {self.index_name} not ready after {timeout} seconds")
        
    async def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """Insert or update vectors in Pinecone"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # Convert to Pinecone format
            upsert_data = []
            for vector_data in vectors:
                upsert_data.append((
                    vector_data['id'],
                    vector_data['values'],
                    vector_data.get('metadata', {})
                ))
                
            # Batch upsert (Pinecone recommends batches of 100)
            batch_size = 100
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i:i + batch_size]
                self.index.upsert(batch)
                
            logger.debug(f"Upserted {len(upsert_data)} vectors to Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors to Pinecone: {e}")
            raise
            
    async def search(self, query_vector: np.ndarray, top_k: int = 10, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # Perform search
            search_result = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter  # Pinecone supports metadata filtering
            )
            
            # Convert results to standard format
            results = []
            for match in search_result['matches']:
                results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match.get('metadata', {})
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to search in Pinecone: {e}")
            return []
            
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors from Pinecone"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            self.index.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} vectors from Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {e}")
            raise
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            stats = self.index.describe_index_stats()
            
            return {
                'vector_count': stats['total_vector_count'],
                'dimension': stats['dimension'],
                'index_fullness': stats['index_fullness'],
                'namespaces': stats.get('namespaces', {}),
                'index_name': self.index_name,
                'environment': self.environment
            }
            
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {'error': str(e)}

# Factory function
def create_vector_db(config: Dict[str, Any]) -> VectorDatabase:
    """Create the appropriate vector database based on configuration"""
    db_type = config.get('type', 'qdrant')
    
    if db_type == 'pinecone':
        return PineconeVectorDB(config)
    else:
        return QdrantVectorDB(config)