from typing import List, Optional
import logging
import asyncio

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# OpenAI embedding model - text-embedding-3-small is fast and cheap
# 1536 dimensions matches our Vector column
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


class EmbeddingService:
    """Service for generating embeddings using OpenAI."""

    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        if settings.OPENAI_API_KEY:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    def _build_card_text(
        self,
        name: str,
        type_line: Optional[str] = None,
        oracle_text: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> str:
        """Build text representation of a card for embedding."""
        parts = [name]

        if type_line:
            parts.append(type_line)

        if oracle_text:
            parts.append(oracle_text)

        if keywords:
            parts.append(f"Keywords: {', '.join(keywords)}")

        return " | ".join(parts)

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        if not self.client:
            logger.warning("OpenAI client not configured, skipping embedding")
            return None

        try:
            response = await self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
                dimensions=EMBEDDING_DIMENSIONS,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    async def get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts in batches."""
        if not self.client:
            logger.warning("OpenAI client not configured, skipping embeddings")
            return [None] * len(texts)

        results: List[Optional[List[float]]] = [None] * len(texts)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = await self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    dimensions=EMBEDDING_DIMENSIONS,
                )
                for j, embedding_data in enumerate(response.data):
                    results[i + j] = embedding_data.embedding

                # Rate limiting - be nice to the API
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                # Continue with next batch

        return results

    async def get_card_embedding(
        self,
        name: str,
        type_line: Optional[str] = None,
        oracle_text: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> Optional[List[float]]:
        """Generate embedding for a card."""
        text = self._build_card_text(name, type_line, oracle_text, keywords)
        return await self.get_embedding(text)

    async def get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for a search query."""
        return await self.get_embedding(query)


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
