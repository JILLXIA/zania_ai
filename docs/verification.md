# Implementation verification

Checked on macOS / Python 3.12.14, 2026-09-24. These notes separate local evidence from checks that still require external services.

## Completed

- **76 offline tests pass**, with network sockets disabled. No API key or downloaded embedding model is required for that suite.
- Ruff lint and formatting checks, Python compilation, JavaScript syntax check, and `pip check` pass.
- Started the actual Uvicorn server on loopback and verified `/health` over HTTP returned `{"status":"ok"}`. No API key was configured for this smoke check.
- Tests exercise real JSON/PDF parsing, real PDFium image-region rendering, real FAISS indexing/MMR, multipart validation, response schemas, and service orchestration. Deterministic embeddings and fake answers are test doubles, not claims of answer quality.
- The real LangChain/OpenAI SDK is exercised against an in-memory HTTP transport: structured schemas, the fixed `gpt-4o-mini` model, image payloads, refusal/invalid output, retry bounds, token retry budget, and shared call concurrency.
- Additional checks cover question order/deduplication, partial-answer text, unknown/fabricated citations, PDF versus JSON isolation, visual failure handling, byte/field limits, parser/vision deadlines, disconnect cleanup, CPU cancellation accounting, and content-free JSON logs.
- Public BGE ONNX artifacts were downloaded at revision `aa8f8b060edb00e03bfdd08813a2949946c8ba55`. Actual local inference produces 384-dimensional vectors.
- Converted the supplied CSV locally and verified 19 records. The conversion utility preserves all named field values and refuses overwrites. The generated private JSON stays under the ignored requirements directory.
- Synthetic JSON, independent questions, and a one-page PDF with an embedded diagram are included for fresh-checkout requests.

## Real sample retrieval (not live generation)

The complete 84-page PDF parsed within the configured bounds. Candidate image pages were **1, 2, 5, 8, 13, 15, 16, 28**; the infrastructure and organization diagrams were cropped for pages 15 and 16.

Real local retrieval was run using the independent questions in `examples/questions.json`. The final PDF run took about **28.5 seconds** after model initialization on this machine; this is an observation, not a latency guarantee.

| PDF question | Retrieved physical pages | Checked evidence |
| --- | --- | --- |
| Which cloud providers do you rely on? | 58, 9, 17, 16, 81 | Context includes the explicit GCP application-hosting statement on page 16. |
| Is personal information disclosed to third parties? | 59, 42, 63, 66 | Retrieval only; answerability has not been established by a live answer review. |
| How are incidents reported and what is the notification SLA? | 75, 72, 42, 21 | Context includes page 21's “without undue delay” wording. A numeric SLA must not be invented. |

Initial 400-token chunks and a 3,500-byte context cap missed some useful evidence. The implementation now uses 250-token chunks / 40 overlap, bounded same-source expansion, and a 12,000-byte context budget. This improved coverage of the checked GCP and notification passages without adding another model or retrieval service. It is a small sample check, not a benchmark or proof of generalized retrieval quality.

The converted JSON document was also queried separately using real local embeddings. In the final run (about 2.1 seconds after model initialization), hosting, personal-information disclosure and incident-notification questions ranked records `/0`, `/3` and `/10` first, respectively. No PDF content was used in the JSON request.

## Not yet verified

- **Live OpenAI generation/vision:** no paid API calls were made. Actual diagram-reading accuracy, answer entailment, prompt-injection resistance, provider latency/rate limits, and total usage remain to be reviewed with `scripts/evaluate.py --live` after configuring a key and budget. A passing mocked test does not establish these properties.
- **Docker build and Linux runtime:** Docker CLI is present, but the local Docker daemon is not running. The Dockerfile and version lock are supplied; image build, Linux dependency compatibility, worker memory limits, non-root rendering and container health still need a real container smoke test.
- **Browser visual/interaction QA:** the Browser tool reported no connected browser. HTTP/static-route and JavaScript syntax checks pass; real file-selection, card rendering, responsive layout and download interactions still need browser review.

## Suggested final interview checks

1. Configure a private `.env` key; start the service and check `/ready`.
2. Run one synthetic JSON request and one synthetic PDF request with `--live`; inspect the image observation against the source diagram.
3. Run a small independent question set against each original sample **separately**. Check citations, partial answers, unsupported questions, and the distinction between a missing fact and a provider failure.
4. Start Docker and run the documented image build/request checks; verify `/health` and `/ready` independently.
5. Review the page in a browser and perform a submission-time secret scan. Do not publish the original requirements directory or private evaluation output.
