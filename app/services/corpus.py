"""Index and retrieve source chunks using dense and sparse Qdrant vectors."""

import json
from collections.abc import Iterable
from itertools import islice
from tempfile import TemporaryFile
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.config import settings
from app.schemas.agent_state import EvidenceChunk

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
DENSE_SIZE = 384


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


def compatible_collection(params: models.CollectionParams) -> bool:
    dense = (
        params.vectors.get(DENSE_VECTOR) if isinstance(params.vectors, dict) else None
    )
    sparse = (params.sparse_vectors or {}).get(SPARSE_VECTOR)
    return (
        dense is not None
        and dense.size == DENSE_SIZE
        and dense.distance == models.Distance.COSINE
        and sparse is not None
        and sparse.modifier == models.Modifier.IDF
    )


def index_chunks(
    client: QdrantClient, collection: str, chunks: Iterable[EvidenceChunk]
) -> int:
    # Validate and snapshot the entire input before mutating the collection.
    with TemporaryFile(mode="w+t", encoding="utf-8") as validated:
        identities = set()
        for chunk in chunks:
            identity = (chunk.source_doc, chunk.chunk_id)
            if identity in identities:
                raise ValueError(f"Duplicate source and chunk ID: {identity!r}.")
            identities.add(identity)
            validated.write(chunk.model_dump_json() + "\n")
        if not identities:
            return 0
        validated.seek(0)

        if not client.collection_exists(collection):
            client.create_collection(
                collection_name=collection,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=DENSE_SIZE, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
            )
        if not compatible_collection(client.get_collection(collection).config.params):
            raise ValueError(
                "The collection does not use the expected dense and BM25 vectors."
            )

        indexed = 0
        iterator = (EvidenceChunk.model_validate_json(line) for line in validated)
        while batch := list(islice(iterator, 32)):
            points = [
                models.PointStruct(
                    id=chunk_id(chunk),
                    vector={
                        DENSE_VECTOR: models.Document(
                            text=chunk.text, model=DENSE_MODEL
                        ),
                        SPARSE_VECTOR: models.Document(
                            text=chunk.text, model=SPARSE_MODEL
                        ),
                    },
                    payload=chunk.model_dump(),
                )
                for chunk in batch
            ]
            result = client.upsert(
                collection_name=collection, points=points, wait=True, timeout=30
            )
            if result.status != models.UpdateStatus.COMPLETED:
                raise RuntimeError(
                    f"Qdrant returned {result.status} after {indexed} confirmed chunk(s). "
                    "Retry the same input to complete indexing."
                )
            indexed += len(points)
        return indexed
