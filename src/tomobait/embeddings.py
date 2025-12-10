"""
Embedding model utilities for TomoBait.

Provides a unified interface for different embedding providers (HuggingFace, ANL Argo).
"""

from typing import List

import requests
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from .config import get_config


class ArgoEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings class for ANL Argo embedding API.

    This class wraps the ANL Argo embedding API to be compatible with
    LangChain's Embeddings interface, allowing it to be used with ChromaDB
    and other LangChain components.
    """

    def __init__(
        self,
        user: str,
        model: str = "ada002",
        base_url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/",
        batch_size: int = 100,
    ):
        """
        Initialize the Argo embeddings client.

        Args:
            user: ANL username for authentication
            model: Argo embedding model (default: "ada002")
            base_url: Argo API endpoint URL
            batch_size: Maximum number of texts to embed in one request
        """
        self.user = user
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts using the Argo API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        payload = {
            "user": self.user,
            "model": self.model,
            "prompt": texts,
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()
            embeddings = result.get("embedding", [])

            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                )

            return embeddings

        except requests.exceptions.RequestException as e:
            print(f"Error calling Argo embedding API: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            print(
                f"  Embedding batch {i // self.batch_size + 1}/"
                f"{(len(texts) + self.batch_size - 1) // self.batch_size}..."
            )
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embeddings = self._embed_batch([text])
        return embeddings[0]


def get_embeddings() -> Embeddings:
    """
    Get the configured embeddings instance based on config.yaml settings.

    Returns:
        LangChain Embeddings instance (either HuggingFaceEmbeddings or ArgoEmbeddings)
    """
    config = get_config()
    embedding_config = config.embedding

    if embedding_config.provider == "argo":
        if not embedding_config.argo_user:
            # Fall back to LLM api_key if argo_user not set
            argo_user = config.llm.api_key
            if not argo_user:
                raise ValueError(
                    "Argo embedding requires either 'embedding.argo_user' or "
                    "'llm.api_key' to be set in config.yaml"
                )
        else:
            argo_user = embedding_config.argo_user

        print(
            f"Using Argo embeddings (model: {embedding_config.model}, "
            f"user: {argo_user})"
        )
        return ArgoEmbeddings(
            user=argo_user,
            model=embedding_config.model,
            base_url=embedding_config.argo_base_url,
        )
    else:
        # Default to HuggingFace local embeddings
        model_name = embedding_config.model
        device = embedding_config.device

        # Build model_kwargs for device selection
        model_kwargs = {}
        if device != "auto":
            model_kwargs["device"] = device

        print(f"Using HuggingFace embeddings (model: {model_name}, device: {device})")
        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
