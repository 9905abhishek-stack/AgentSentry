# AgentSentry

AgentSentry audits factual, compliance, and security assertions against document evidence. The FastAPI service returns a verdict, per-claim reasoning, source citations, and a human-review flag. It accepts evidence with a request or retrieves it from an indexed Qdrant corpus.

## Audit workflow

1. A vLLM model decomposes an assertion into up to ten claims. LangGraph verifies them in dependency order. A claim whose prerequisite is uncertain also requires review.
2. The service uses the supplied evidence chunks or queries Qdrant for dense BGE-small and sparse BM25 matches. It combines the corpus rankings with reciprocal rank fusion using `k=60` by default.
3. A first model call extracts factual tuples with source IDs and verbatim quotes. AgentSentry checks each quote against its source, sorts and deduplicates the tuples, and hashes their canonical JSON using SHA-256. Each tuple includes a hash of its source text.
4. A separate model call decides the verdict from the claim and frozen tuples. In strict mode this call receives only the claim and frozen facts. Setting `strict_evidence_lock` to `false` also provides the original evidence to the decision call. Both modes validate the cited IDs.
5. The audit reports `SUPPORTED`, `REFUTED`, `CONFLICTING_EVIDENCE`, or `INSUFFICIENT_EVIDENCE`. Any uncertain claim or per-claim confidence below the configured threshold flags the audit for human review. The flag is part of the response for the caller to act on.

The model supplies the confidence scores and interprets the text. The hashes let a reviewer check the frozen representation and source contents. Semantic correctness, model calibration, and resistance to instructions embedded in documents remain model-dependent. Use corpora and request evidence you are authorized to send to the configured inference service.

## Run locally

Requires Python 3.11 or later. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Swagger UI is available at `http://127.0.0.1:8000/docs`. By default the service runs without external connections and abstains with `INSUFFICIENT_EVIDENCE`. To enable inference, set `VLLM_API_BASE` to the OpenAI-compatible API base including `/v1`, and set `VLLM_MODEL` to the served model ID. For example, for a model already listening on port 8001:

```powershell
$env:VLLM_API_BASE = 'http://127.0.0.1:8001/v1'
$env:VLLM_MODEL = 'my-served-model'
uvicorn app.main:app --port 8000
```

Set `VLLM_API_KEY` if the endpoint requires a key. The model must support JSON-object chat responses. The service also reads settings from a local `.env` file.

## Index a corpus

Set `QDRANT_HOST` to a running Qdrant instance and `QDRANT_PORT` if it differs from 6333. `QDRANT_COLLECTION` defaults to `audit_evidence`. The indexer takes UTF-8 JSON Lines with a source name, a unique chunk number within that source, and up to 2,000 characters of text per line.

```json
{"source_doc":"retention-policy.txt","chunk_id":1,"text":"Production access logs are retained for 30 days."}
```

```powershell
$env:QDRANT_HOST = 'localhost'
python -m app.index_corpus evidence.jsonl
```

The indexer creates a compatible collection if needed and upserts by source and chunk number. FastEmbed downloads the dense and sparse model assets on first use. For a replacement snapshot, index into a fresh collection and point the API at it. When updating a collection in place, change `CORPUS_REVISION` on the API after indexing to invalidate cached verdicts.

`docker compose up --build -d` runs the API, Qdrant, and Redis. Index from the host against `localhost:6333`. Compose stores Qdrant data in the `qdrant_data` volume. If you used the earlier Compose file with `qdrant_storage/`, reindex your source chunks into the new volume. When the inference service runs on the Docker Desktop host, set `VLLM_API_BASE=http://host.docker.internal:8001/v1` for Compose and provide the served model ID. The API and Qdrant host ports bind to loopback. For a browser frontend on another origin, set `CORS_ORIGINS` to a JSON list of allowed origins. Deployments serving other users need an authenticated gateway and appropriate corpus access controls.

## API

| Route | Result |
| --- | --- |
| `POST /api/v1/audit` | Complete audit response with evidence and per-claim citations |
| `POST /api/v1/audit/stream` | Server-sent events with `start`, `node_complete`, and `done` with the full audit result, or `error` |
| `GET /api/v1/metrics` | Process-local counts, cache hits, human-review flags, and average completed-audit latency |
| `GET /health` | Status of the configured Redis, Qdrant collection, and inference model |

Both audit routes accept the same JSON:

```json
{
  "query": "Production access logs are retained for 30 days.",
  "context_documents": ["Production access logs are retained for 30 days."],
  "strict_evidence_lock": true
}
```

Omit `context_documents` to search the configured corpus. Provide up to eight chunks of 2,000 characters each to use request-local evidence instead. An empty list explicitly supplies no evidence. Omit `strict_evidence_lock` to use `EVIDENCE_LOCKED_STRICT`, which defaults to true. `CONFIDENCE_THRESHOLD` defaults to 0.85.

The service caches completed corpus audits that clear the review gate. Cache keys distinguish the query text, verification mode, model, and corpus revision. Cache hits retain source evidence and citations. Set `REDIS_URL` to share the cache across workers. Each worker has a bounded, expiring local fallback. Request-local evidence bypasses the cache.

## Evaluation and tests

```powershell
python -m pip install pytest
python -m pytest
python -m evaluation.run_ragas_eval
```

The evaluation command requires a configured vLLM model. It reports verdict accuracy on four labeled fixtures, the fraction of paired verdicts that change when the same evidence is reversed, and mean verification latency. This is a small verifier benchmark. The fixture results do not measure corpus retrieval precision, RAGAS faithfulness, or production latency.

## License

MIT. Created by [Abhishek Raghuwanshi](https://github.com/9905abhishek-stack).
