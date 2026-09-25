# Document QA — Zania coding challenge

One FastAPI service answers a JSON list of questions from **one uploaded PDF or JSON document**. PDF processing includes native text and bounded `gpt-4o-mini` image analysis. A small browser UI is served by the same app.

No agent loop, database, queue, frontend build system, or persistent document store.

## Run locally

Use Python 3.12 (3.14 is not supported by this dependency set).

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
python scripts/prepare_model.py
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then:

```sh
uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Open <http://127.0.0.1:8000>. API documentation: `/docs`. `/health` checks liveness; `/ready` verifies that the service loaded its local model and has a configured key. Readiness does **not** make an OpenAI call or validate the key remotely. Without a key/model, the UI and health endpoint still work; QA returns 503.

The preparation step downloads a public, revision-pinned ~63 MiB ONNX model plus tokenizer files. Runtime embeddings are local and use no OpenAI embedding API. The base BGE model is MIT licensed; see [BAAI's model card](https://huggingface.co/BAAI/bge-small-en-v1.5) and the [ONNX distribution](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q).

## API and examples

`POST /qa` expects exactly two multipart **files**:

- `questions`: UTF-8 `.json` array of nonempty strings.
- `document`: one `.pdf` or `.json` file. A JSON knowledge base's own `question` fields are source content, not additional request questions.

```json
["Which cloud providers do you rely on?", "Is personal information disclosed to third parties?"]
```

The repository includes independently authored questions and synthetic JSON/PDF documents. These are not reproductions of the private challenge samples.

For a broader set of paired examples and failure scenarios, see the [example catalog](examples/README.md). It includes expected review outcomes, invalid uploads, scanned/encrypted/blank PDFs, and offline operational-failure checks.

```sh
# JSON document request
curl --fail-with-body http://127.0.0.1:8000/qa \
  -F 'questions=@examples/questions.json;type=application/json' \
  -F 'document=@examples/document.json;type=application/json'

# Separate PDF document request; includes image-analysis calls
curl --fail-with-body http://127.0.0.1:8000/qa \
  -F 'questions=@examples/questions.json;type=application/json' \
  -F 'document=@examples/document.pdf;type=application/pdf'
```

Every submitted question gets one ordered result after successful ingestion. Duplicate questions are processed once, then restored to their original positions and strings.

```json
{
  "request_id": "server-generated-id",
  "results": [{
    "question": "Who hosts the service, and in which region?",
    "answer": "The service is hosted on GCP.\n\nNot found in document: the hosting region.",
    "status": "partial",
    "citations": [{"source_type": "text", "source_path": "/0", "excerpt": "Google Cloud Platform (GCP)"}]
  }],
  "warnings": []
}
```

Statuses:

- `answered`: supported answer with citations.
- `partial`: supported facts **and missing details in the same answer string**.
- `not_found`: exactly `Not found in document`, with empty citations.
- `error`: an operational failure, with a stable error code. Never presented as missing evidence.

PDF citations use physical, one-based `page` numbers; JSON citations use RFC 6901 `source_path` pointers (`""` means the root). Text excerpts match whitespace-normalized extracted text or serialized JSON. An `image` citation quotes a **model-extracted observation**, not a verified verbatim PDF quotation. The UI labels this distinction.

The server resolves source locations and checks excerpt membership. These checks verify provenance, not whether every answer claim logically follows from its citations.

Failures use `{"request_id":"...","error":{"code":"...","message":"..."}}`. Invalid syntax/multipart → 400; upload deadline → 408; size/work limits → 413; unsupported format → 415; invalid content/schema → 422; invalid model output → 502; unavailable/configuration/busy → 503; processing timeout → 504. Unexpected/too many multipart parts are rejected by the parser with 400; missing/duplicate expected fields produce 422.

After ingestion, mixed success/failure returns 200 with per-question statuses. All-question failure returns 502/503/504 and includes the result entries. A failed required vision call fails ingestion; the service does not silently claim to have searched the image content.

## How it works

1. Validate uploads and the entire question list before expensive work.
2. Parse JSON records or extract PDF text in a disposable subprocess. For PDFs, select substantial raster images and captioned vector diagrams, then render one bounded crop per candidate page.
3. Analyze selected images once with `gpt-4o-mini`. Reuse identical image/context pairs within the request. Add source-labeled observations alongside native text.
4. Split source units with LangChain; embed with FastEmbed/BGE; build one request-local FAISS index. Retrieve diverse evidence for each unique question.
5. Generate structured answers through LangChain `ChatOpenAI`, validate citations, and restore input order.

The PDF and JSON branches never share evidence. Source URLs are not followed. Source questions, confidence labels, and `Data-Not-Found` are not themselves factual answers. Prompts treat uploaded content and user questions as untrusted data; no model tools are enabled.

Files are deliberately small and concrete:

| File | Responsibility |
| --- | --- |
| `app/main.py` | HTTP validation, request limits, lifecycle, UI |
| `app/ingestion.py` | JSON/PDF parsing, rendering, worker cleanup |
| `app/retrieval.py` | Local model, splitting, FAISS and evidence selection |
| `app/provider.py` | Two structured `gpt-4o-mini` calls: vision and answers |
| `app/service.py` | Orchestration, citations, partial failures |
| `app/models.py`, `config.py`, `logging.py` | Schemas, settings, JSON logs |
| `app/static/` | HTML/CSS/JavaScript, no build step |

## Limits and operational behavior

Configure settings with `ZANIA_` environment variables; `.env.example` lists common overrides. Full definitions are in `app/config.py`.

| Limit | Default |
| --- | --- |
| Request / document / questions upload | 21 MiB / 20 MiB / 256 KiB |
| Questions / question length | 30 / 2,000 characters |
| PDF pages / selected visual pages | 200 / 10 |
| Rendered image | ≤1,536 px longest side; ≤4 MiB PNG |
| Visual input budget | 350,000 estimated tokens, including retries |
| Text / JSON depth / chunks | 1,000,000 characters / 32 / 2,000 |
| Chunk target / overlap | 250 / 40 local-model tokens, also byte-bounded |
| Retrieved context | Up to 6 hits; expand to the same page/record when ≤5,000 bytes; 12,000 total UTF-8 bytes including labels (roughly 3K English tokens, **not** an exact GPT-token count) |
| Upload / parse-and-render / vision / processing deadline | 60 / 40 / 120 / 180 seconds |
| Provider attempt / retry count | 30 seconds / at most one transient retry |
| Active requests / global model calls | 2 / 6 per process |
| Answer calls / vision calls per request | 3 / 2 |
| Answer / image output | 800 / 1,200 tokens |

CPU work runs outside the event loop, with one embedding job at a time and two ONNX inference threads. Cancelled embedding work retains its slot until the thread exits; the processing deadline is therefore not a hard OS-level wall-clock kill for that stage. PDF/JSON workers are killed and reaped on cancellation. Linux workers have a 512 MiB address-space limit; macOS development relies on the other bounds. Upload limits count actual received bytes, not just `Content-Length`.

Temporary documents/rasters are removed after workers stop. Uploads, answers and indexes are not persisted. JSON logs contain request IDs, stage timing, counts, retries and returned model token usage, not document text or credentials. External tracing and ONNX telemetry are disabled. Use one Uvicorn worker to retain the documented process-wide limits.

Vision can be expensive relative to text; the per-request budget is **not** an account-wide spending cap. Set a provider project budget separately. Over-limit scanned PDFs are rejected, not silently truncated.

## Tests and evaluation

```sh
pytest
ruff check app scripts tests main.py
ruff format --check app scripts tests main.py
node --check app/static/app.js
```

Tests disable network sockets and use deterministic embeddings plus mocked provider responses. They still exercise actual multipart handling, JSON parsing, PDF extraction/rendering, FAISS, service orchestration, and OpenAI SDK request/response handling. No key or model download is needed for tests.

Additional opt-in checks:

```sh
# Real local embeddings and retrieval; zero OpenAI calls
python scripts/evaluate.py examples/document.json
python scripts/evaluate.py examples/document.pdf

# Paid model calls: explicitly opt in after configuring your key/budget
python scripts/evaluate.py examples/document.pdf --live
```

Local-only PDF evaluation reports visual candidate pages but does not analyze them. It cannot establish vision or final answer quality. Evaluation output intentionally contains source evidence; keep private sample runs out of shared logs. See [verification notes](docs/verification.md) for actual results and remaining checks.

The supplied CSV is a **JSON document export**, not the questions input. If the original local files are available:

```sh
python scripts/convert_sample.py \
  'codingRequirements/Sample JSON.xlsx - Sheet1.csv' \
  'codingRequirements/sample-document.json'
python scripts/evaluate.py codingRequirements/sample-document.json
python scripts/evaluate.py codingRequirements/Nave-SOC2-Type-2-Report.pdf
```

Conversion preserves all 19 rows and all named field values; only the unnamed index column is dropped. It refuses to overwrite an existing output. Original challenge files are excluded from Git and Docker because they include private material and an API credential. The included examples work without them.

## Docker

```sh
docker build -t zania-qa .
docker run --rm --name zania-qa -p 127.0.0.1:8000:8000 --env-file .env zania-qa
```

The image installs version-pinned runtime dependencies and downloads revision-pinned public model artifacts during the build. It runs as a non-root user with one worker. No credentials or original challenge files enter the build context. The Docker health check is liveness; use `/ready` for configuration readiness.

## Known limitations

- Visual selection is a heuristic, not complete page OCR. Tiny or uncaptioned vector graphics can be missed; small labels/relationships may be misread. Scan-heavy documents can exceed the visual-page budget.
- Local BGE embeddings are English-focused. Retrieval and the LLM can still miss evidence or misinterpret contradictions. Evaluate multilingual or specialist documents before relying on them.
- Citation validation cannot prove semantic entailment or defeat every prompt injection. Human review remains necessary for security questionnaires.
- No authentication, persistent jobs, shared multi-process rate limiter, or audit store. Intended for local interview evaluation, not unauthenticated public deployment.

The original [design plan](docs/design-plan.md) and [interviewer questions/decisions](docs/interviewer-questions.md) record the agreed scope and implementation adjustments.
