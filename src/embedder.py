import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info("SentenceTransformer model loaded successfully.")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates embedding vector for a single text.
        """
        # clean input text
        cleaned_text = text.replace("\n", " ").strip()
        embedding = self.model.encode(cleaned_text, convert_to_numpy=True)
        return embedding

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embedding vectors for a list of texts.
        """
        cleaned_texts = [t.replace("\n", " ").strip() for t in texts]
        embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True, show_progress_bar=False)
        return np.array(embeddings)
