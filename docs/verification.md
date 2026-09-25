# Implementation verification

Checked on macOS / Python 3.12.14, most recently 2026-09-25. These notes separate local evidence from checks that still require external services.

## Completed

- **150 offline tests pass**, with network sockets disabled. No API key or downloaded embedding model is required for that suite. The original suite had 76 tests; the expanded example catalog adds 36 checks, optional LangSmith tracing adds 14, JSON citation regression coverage adds 9, and hybrid retrieval adds 15.
- Ruff lint and formatting checks, Python compilation, JavaScript syntax check, and `pip check` pass.
- Started the actual Uvicorn server on loopback and verified `/health` over HTTP returned `{"status":"ok"}`. No API key was configured for this smoke check.
- Tests exercise real JSON/PDF parsing, real PDFium image-region rendering, real FAISS indexing/MMR, BM25/RRF, multipart validation, response schemas, and service orchestration. Deterministic embeddings and fake answers are test doubles, not claims of answer quality.
- The real LangChain/OpenAI SDK is exercised against an in-memory HTTP transport: structured schemas, the fixed `gpt-4o-mini` model, image payloads, refusal/invalid output, retry bounds, token retry budget, and shared call concurrency.
- A user-supplied live trace exposed a JSON citation rewritten by joining the `answer` and `comments` fields. The prompt now explicitly requires continuous source spans, unchanged capitalization/punctuation, and separate citations for separate field values, with valid/invalid examples. Regression tests accept literal quotes and reject the reported merged quote; a mocked SDK test verifies that the new rules are actually sent. The exact-match validator is unchanged. These checks do not prove live model compliance with the revised prompt; restart and rerun the sanctions question to verify it.
- The real LangSmith SDK is exercised against a recording HTTP session: explicit client/project routing, request/child trace relationships, vector-index events, input/output masking (including vision payloads), visible retries, concurrent-request isolation, nonfatal tracing authentication failures, and parser-worker credential exclusion. A regression guard rejects accidental use of a default tracing client. These tests do not validate a real LangSmith account or dashboard.
- Additional checks cover question order/deduplication, partial-answer text, unknown/fabricated citations, PDF versus JSON isolation, visual failure handling, byte/field limits, parser/vision deadlines, disconnect cleanup, CPU cancellation accounting, and content-free JSON logs.
- Public BGE ONNX artifacts were downloaded at revision `aa8f8b060edb00e03bfdd08813a2949946c8ba55`. Actual local inference produces 384-dimensional vectors.
- Converted the supplied CSV locally and verified 19 records. The conversion utility preserves all named field values and refuses overwrites. The generated private JSON stays under the ignored requirements directory.
- Synthetic JSON, independent questions, and a one-page PDF with an embedded diagram are included for fresh-checkout requests.
- The [expanded example catalog](../examples/README.md) adds paired JSON cases for partial/missing answers, record semantics, contradictions, nested data, wording/ambiguity, prompt injection, and request isolation. A reproducible local generator supplies native-text, scanned, encrypted and blank PDFs plus malformed/limit-boundary fixtures. Offline checks assert parsing, source locations, exact rejection codes and simulated provider failures, not live semantic answer quality.

## Earlier dense-only sample retrieval (not live generation)

The complete 84-page PDF parsed within the configured bounds. Candidate image pages were **1, 2, 5, 8, 13, 15, 16, 28**; the infrastructure and organization diagrams were cropped for pages 15 and 16.

Real local retrieval was run using the independent questions in `examples/questions.json`. The final PDF run took about **28.5 seconds** after model initialization on this machine; this is an observation, not a latency guarantee.

| PDF question | Retrieved physical pages | Checked evidence |
| --- | --- | --- |
| Which cloud providers do you rely on? | 58, 9, 17, 16, 81 | Context includes the explicit GCP application-hosting statement on page 16. |
| Is personal information disclosed to third parties? | 59, 42, 63, 66 | Retrieval only; answerability has not been established by a live answer review. |
| How are incidents reported and what is the notification SLA? | 75, 72, 42, 21 | Context includes page 21's “without undue delay” wording. A numeric SLA must not be invented. |

Initial 400-token chunks and a 3,500-byte context cap missed some useful evidence. The dense-only version adopted 250-token chunks / 40 overlap, bounded same-source expansion, and a 12,000-byte context budget. This improved coverage of the checked GCP and notification passages without adding another model or retrieval service. It was a small sample check, not a benchmark or proof of generalized retrieval quality. The current hybrid version's larger context cap and checks are described below.

The converted JSON document was also queried separately using real local embeddings. In the final run (about 2.1 seconds after model initialization), hosting, personal-information disclosure and incident-notification questions ranked records `/0`, `/3` and `/10` first, respectively. No PDF content was used in the JSON request.

## Hybrid retrieval checks (2026-09-25)

Added request-local `rank-bm25==0.2.2`, keeping the local embedding model and FAISS.
The dense branch still applies MMR to its top 12 candidates, selecting up to 6.
RRF merges those hits with up to 12 positive BM25 hits, recognizing different
chunks of the same expandable source as one result. Final context is bounded by
8 evidence chunks and 16,000 UTF-8 bytes, up from 6 chunks / 12,000 bytes. This can
increase live prompt size, but adds no model calls or external retrieval service.

The first naive fusion version dropped useful PDF passages. Common-word filtering
alone did not fix that. The checked implementation retains dense diversity,
deduplicates source-level votes, gives both branches more context space, and
retains a matching chunk when its whole source does not fit. These are small-sample
development adjustments, not held-out benchmark results or proof that every query
improves. Some supporting passages now rank lower, even though they still fit.

Fifteen new offline cases cover exact identifier recovery outside the dense top
12, RRF agreement/ties/duplicate votes, no-match dense fallback, small/tokenless
corpora, expandable versus large-source fusion, original source metadata, UTF-8
context limits, expansion fallback, deduplication, and request isolation. The
LangSmith test also checks the new content-free `hybrid_retrieval` event.

Both private demo documents were rerun locally with all seven questions from
`demo/questions.json`, without vision or answer-model calls. Final observed times
after model initialization: JSON **2.02 s**, PDF **28.11 s**; not latency guarantees.

| Demo question | JSON evidence retained | PDF native-text evidence retained |
| --- | --- | --- |
| Cloud providers | `/0`, ranked first | Page 16's Google Cloud Platform statement |
| TLS versions | `/5`, ranked first | Page 16 |
| Dedicated sanctions officer | `/11`, ranked first | No answerability claim; related hits are not evidence of a dedicated officer |
| Office wireless network | `/14`, `/13`, ranked first and second | No answerability claim |
| Hosting plus exact monthly cost | `/0`, ranked first | Page 16 hosting statement; missing cost must not be invented |
| Incident reporting plus notification deadline | `/10`, ranked first | Pages 75 and 21; returned text includes `without undue delay` |

The revenue question still retrieves loosely related content. No similarity or
BM25 score is used as proof of answerability; the answer model must still choose
`not_found` or `partial` based on evidence. Live answer/citation quality has not
been evaluated for this retrieval change.

## Not yet verified

- **Live LangSmith delivery:** the user supplied model input/output trace information from their live setup. Automated verification remains mocked; account authentication, workspace/region configuration, and the complete dashboard trace tree have not been independently verified. See [setup and privacy instructions](langsmith.md).
- **Live OpenAI generation/vision:** no paid API calls were made by the coding agent. The user's reported citation failure is covered above; revised-prompt behavior, diagram-reading accuracy, answer entailment, prompt-injection resistance, provider latency/rate limits, and total usage still need live review. A passing mocked test does not establish these properties.
- **Docker build and Linux runtime:** Docker CLI is present, but the local Docker daemon is not running. The Dockerfile and version lock are supplied; image build, Linux dependency compatibility, worker memory limits, non-root rendering and container health still need a real container smoke test.
- **Browser visual/interaction QA:** the Browser tool reported no connected browser. HTTP/static-route and JavaScript syntax checks pass; real file-selection, card rendering, responsive layout and download interactions still need browser review.

## Suggested final interview checks

1. Configure a private `.env` key; start the service and check `/ready`.
2. Run one synthetic JSON request and one synthetic PDF request with `--live`; inspect the image observation against the source diagram.
3. Run a small independent question set against each original sample **separately**. Check citations, partial answers, unsupported questions, and the distinction between a missing fact and a provider failure.
4. Start Docker and run the documented image build/request checks; verify `/health` and `/ready` independently.
5. Review the page in a browser and perform a submission-time secret scan. Do not publish the original requirements directory or private evaluation output.
