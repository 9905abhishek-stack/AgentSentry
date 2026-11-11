"""Index a JSON Lines file of document chunks into Qdrant."""

import argparse
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from app.config import settings
from app.schemas.agent_state import EvidenceChunk
from app.services.corpus import connect_corpus, index_chunks


def read_chunks(path: Path) -> Iterator[EvidenceChunk]:
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield EvidenceChunk.model_validate_json(line)
                except ValidationError as exc:
                    raise ValueError(
                        f"Invalid evidence chunk on line {number}."
                    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, help="JSONL file with text, source_doc, and chunk_id fields"
    )
    args = parser.parse_args()
    client = connect_corpus()
    try:
        count = index_chunks(client, settings.qdrant_collection, read_chunks(args.path))
    finally:
        client.close()
    print(f"Indexed {count} chunk(s) in {settings.qdrant_collection}.")


if __name__ == "__main__":
    main()
