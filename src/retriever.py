import faiss
import logging
import numpy as np
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class FAISSRetriever:
    def __init__(self, dimension: int):
        self.dimension = dimension
        # Use IndexFlatIP for Inner Product search (which yields Cosine Similarity if inputs are L2-normalized)
        self.index = faiss.IndexFlatIP(dimension)
        self.candidate_ids = []
        logger.info(f"Initialized FAISS IndexFlatIP with dimension {dimension}")

    def add_candidates(self, candidate_embeddings: np.ndarray, ids: List[str]):
        """
        Adds candidate embeddings to the FAISS index.
        Embeddings are L2 normalized to ensure inner product corresponds to cosine similarity.
        """
        assert len(candidate_embeddings) == len(ids), "Embeddings and IDs count mismatch"
        if len(ids) == 0:
            return

        # Normalize embeddings
        faiss.normalize_L2(candidate_embeddings)
        self.index.add(candidate_embeddings)
        self.candidate_ids.extend(ids)
        logger.info(f"Added {len(ids)} candidate embeddings to the FAISS index (Total: {len(self.candidate_ids)})")

    def retrieve(self, query_embedding: np.ndarray, top_n: int = 15) -> List[Tuple[str, float]]:
        """
        Retrieves the top N candidates closest to the query.
        Returns a list of tuples (candidate_id, similarity_score).
        Similarity scores are between 0 and 1.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty. Retrieval aborted.")
            return []

        # Ensure query is 2D and normalized
        query_vector = query_embedding.copy()
        if len(query_vector.shape) == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
            
        faiss.normalize_L2(query_vector)

        # Cap top_n to total available candidates
        actual_top_n = min(top_n, self.index.ntotal)
        
        # Search in FAISS
        scores, indices = self.index.search(query_vector, actual_top_n)

        # Parse results
        results = []
        for i in range(actual_top_n):
            idx = indices[0][i]
            score = float(scores[0][i])
            # Clip score to [0.0, 1.0] range
            score = max(0.0, min(1.0, score))
            
            if idx != -1:
                results.append((self.candidate_ids[idx], score))
                
        logger.info(f"Retrieved {len(results)} candidates using FAISS semantic search.")
        return results
