#!/usr/bin/env python3
"""
Embedding Agent for CodebaseIQ Pro
Handles creation and management of code embeddings
"""

import logging
import hashlib
from typing import Dict, Any

from ..core import BaseAgent, AgentRole

logger = logging.getLogger(__name__)

class EmbeddingAgent(BaseAgent):
    """Agent responsible for creating and managing code embeddings"""
    
    def __init__(self, vector_db, embedding_service):
        super().__init__(AgentRole.EMBEDDING)
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create embeddings for all code entities"""
        entities = context.get('entities', {})
        
        if not context.get('enable_embeddings', True):
            return {
                'embeddings_created': 0,
                'reason': 'Embeddings disabled'
            }
            
        embeddings_created = 0
        batch = []
        failed_entities = []
        
        for path, entity in entities.items():
            try:
                # Generate embedding
                embedding = await self.embedding_service.embed_code_entity(entity)
                entity.embedding = embedding
                entity.embedding_id = hashlib.md5(path.encode()).hexdigest()
                
                # Prepare for vector DB
                batch.append({
                    'id': entity.embedding_id,
                    'values': embedding.tolist(),
                    'metadata': {
                        'path': path,
                        'type': entity.type,
                        'name': entity.name,
                        'importance': entity.importance_score,
                        'danger_level': entity.danger_level,
                        'doc_score': getattr(entity, 'documentation_score', 0)
                    }
                })
                
                if len(batch) >= 100:  # Batch size
                    await self.vector_db.upsert(batch)
                    embeddings_created += len(batch)
                    batch = []
                    
            except Exception as e:
                logger.warning(f"Failed to create embedding for {path}: {e}")
                failed_entities.append(path)
                
        # Upload remaining batch
        if batch:
            try:
                await self.vector_db.upsert(batch)
                embeddings_created += len(batch)
            except Exception as e:
                logger.error(f"Failed to upload final batch: {e}")
                
        # Find semantic neighbors for each entity
        if embeddings_created > 0:
            await self._find_semantic_neighbors(entities)
            
        return {
            'embeddings_created': embeddings_created,
            'total_entities': len(entities),
            'failed_entities': len(failed_entities),
            'vector_db_status': 'synchronized' if embeddings_created > 0 else 'empty',
            'embedding_model': self.embedding_service.get_dimension()
        }
        
    async def _find_semantic_neighbors(self, entities: Dict[str, Any]):
        """Find semantically similar code for each entity"""
        for path, entity in entities.items():
            if hasattr(entity, 'embedding') and entity.embedding is not None:
                try:
                    # Search for similar code
                    results = await self.vector_db.search(
                        query_vector=entity.embedding,
                        top_k=6  # Get top 6 to exclude self
                    )
                    
                    # Filter out self and store neighbors
                    neighbors = []
                    for result in results:
                        neighbor_path = result.get('metadata', {}).get('path')
                        if neighbor_path and neighbor_path != path:
                            neighbors.append(neighbor_path)
                            
                    entity.semantic_neighbors = neighbors[:5]  # Top 5 neighbors
                    
                except Exception as e:
                    logger.debug(f"Failed to find neighbors for {path}: {e}")