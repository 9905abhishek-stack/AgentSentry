"""Index and retrieve source chunks using dense and sparse Qdrant vectors."""

import json
from collections.abc import Iterable
from itertools import islice
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.config import settings
from app.schemas.agent_state import EvidenceChunk

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


def connect_corpus() -> QdrantClient:
    if not settings.qdrant_host:
        raise ValueError("Set QDRANT_HOST to use a Qdrant corpus.")
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        timeout=3,
    )


def chunk_id(chunk: EvidenceChunk) -> str:
    identity = json.dumps([chunk.source_doc, chunk.chunk_id], ensure_ascii=False)
    return str(uuid5(NAMESPACE_URL, identity))


def index_chunks(
    client: QdrantClient, collection: str, chunks: Iterable[EvidenceChunk]
) -> int:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=384, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    indexed = 0
    iterator = iter(chunks)
    while batch := list(islice(iterator, 32)):
        points = [
            models.PointStruct(
                id=chunk_id(chunk),
                vector={
                    DENSE_VECTOR: models.Document(text=chunk.text, model=DENSE_MODEL),
                    SPARSE_VECTOR: models.Document(text=chunk.text, model=SPARSE_MODEL),
                },
                payload=chunk.model_dump(),
            )
            for chunk in batch
        ]
        client.upsert(collection_name=collection, points=points, wait=True, timeout=30)
        indexed += len(points)
    return indexed
