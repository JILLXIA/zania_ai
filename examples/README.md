# Example catalog

Use these to learn the request flow, manually evaluate answers, and reproduce failures. All content is fictional. Every request still uploads **one questions file and one document**; the `expected*.json` files are review notes, never uploads.

There are three different outcomes to keep separate:

- **Unsupported question:** valid request, HTTP 200, `status: not_found`, exact answer `Not found in document`, empty citations.
- **Invalid input:** HTTP 400/413/415/422 with a request error; no model call should be needed.
- **Operational failure:** provider/configuration/deadline failures, not missing document evidence. These are simulated offline because a file cannot reliably force a provider outage.

## Quick start

From the repository root, using the project's Python 3.12 environment:

```sh
# Generate four PDFs and boundary fixtures. No model calls or network access.
.venv/bin/python scripts/create_example_cases.py --large

# Run the new examples' offline checks. No running server or API key needed.
.venv/bin/pytest tests/test_examples.py -v
```

JSON case pairs below are already in the repository. Generated PDFs go in `output/pdf/`; byte/size boundary fixtures go in `output/example-fixtures/`. These generated files are ignored by Git, so run the command after a fresh clone. `--large` adds a >20 MiB document; omit it if you do not need the upload-size fixture. Existing files are never overwritten; `--output output/another-run` creates a fresh set elsewhere.

For real API requests, start the app using the main README and check `/ready` first. **Without a configured key and local model, `/qa` returns 503 before validating uploads**, even when the file is intentionally invalid. Successful answer requests use paid model calls; the offline tests do not.

## JSON question/document pairs

Each ordinary case directory contains `document.json`, `questions.json`, and `expected.json`. Expected results correspond to question positions, not exact generated wording. They are quality targets for manual review, not a claim that live model behavior has been verified.

| Case | What to inspect |
| --- | --- |
| [supported](cases/supported/questions.json) | Four supported questions: AWS/Frankfurt, encryption at rest versus in transit, backup schedule/retention, and facts found only in comments. |
| [partial-missing](cases/partial-missing/questions.json) | Known provider but unknown region; qualitative incident timing but no numeric SLA; unrelated 24-hour offboarding trap; entirely unsupported certification identifier. |
| [record-semantics](cases/record-semantics/questions.json) | Short `Yes`/`No` answers need their source question. `Data-Not-Found`, high confidence, and a question-only record must not invent evidence. |
| [conflicting-sources](cases/conflicting-sources/questions.json) | Undated AWS and Azure claims conflict. Report both with citations; do not invent a migration or assume the later array entry wins. |
| [nested-and-wording](cases/nested-and-wording/questions.json) | Nested JSON, paraphrases, a whitespace-only duplicate, multiple timing scopes, and a deliberately vague question. |
| [prompt-injection](cases/prompt-injection/questions.json) | Ignore instructions embedded in source data or a question; cite the actual Azure hosting record. This is an adversarial evaluation, not proof of injection immunity. |
| [request-isolation](cases/request-isolation/questions.json) | Upload `document-a.json` and `document-b.json` in two separate requests using the same questions. A says AWS/Frankfurt; B says Azure with no region. Never carry Frankfurt into B's answer. Review `expected-a.json` and `expected-b.json`. |

For the small nested document, the parser keeps the entire object as one source. Its JSON Pointer is `""` (the root), not an invented leaf path.

The last nested/wording question has `status: null` in the review notes: that means **no predetermined test assertion**, not a valid API status. Query rewriting and clarification are not implemented. Observe how the model handles missing intent, and do not confuse a fluent answer with a justified interpretation.

Example request; change the directory to try another case:

```sh
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/qa \
  -F 'questions=@examples/cases/partial-missing/questions.json;type=application/json' \
  -F 'document=@examples/cases/partial-missing/document.json;type=application/json'
```

Or use the browser's two upload fields. To inspect real retrieval without a paid model call:

```sh
.venv/bin/python scripts/evaluate.py \
  examples/cases/partial-missing/document.json \
  --questions examples/cases/partial-missing/questions.json
```

Only add `--live` when you want paid answer/vision evaluation. A retrieval-only run does not prove that the final answers are correct.

## PDF examples

Generate these with the quick-start command. The two positive examples have separate questions and review notes in `examples/pdf/`.

| Document | Questions | Expected behavior |
| --- | --- | --- |
| `output/pdf/text-only.pdf` | `examples/pdf/questions-text.json` | Two pages, native text only. Azure/West Europe on page 1; incident notification and offboarding on page 2. Answers should distinguish partial support from an unsupported certificate number. See `expected-text.json`. |
| `output/pdf/scanned-policy.pdf` | `examples/pdf/questions-scanned.json` | Image-only page: AWS/Ireland, backup retention 45 days, administrator MFA required. These answers need `source_type: image`, page 1. No incident-notification SLA is supplied. See `expected-scanned.json`. |
| `output/pdf/encrypted.pdf` | `examples/questions.json` | HTTP 422, `encrypted_pdf`; no vision call. The viewer password is `sample-password`, but the API intentionally has no password input. |
| `output/pdf/blank.pdf` | `examples/questions.json` | Intentionally blank page; HTTP 422, `empty_document`, not a successful answer of `not_found`. |

```sh
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/qa \
  -F 'questions=@examples/pdf/questions-scanned.json;type=application/json' \
  -F 'document=@output/pdf/scanned-policy.pdf;type=application/pdf'
```

The original `examples/document.pdf` remains a mixed native-text/diagram example. To specifically exercise its image-only facts, submit a questions array such as `["Which database and cache components appear in the diagram?"]`.

## Invalid upload fixtures

Replace **only the indicated field** in a normal request. For an invalid questions file, keep `examples/document.json` as the document. For an invalid document, keep `examples/questions.json` as the questions. This makes the reason for rejection unambiguous.

### Files already included under `examples/failures/`

| File | Field | HTTP | Error code |
| --- | --- | --- | --- |
| `questions-empty.json` | questions | 422 | `invalid_questions` |
| `questions-object.json` | questions | 422 | `invalid_questions` |
| `questions-records.json` | questions | 422 | `invalid_questions` |
| `questions-mixed.json` | questions | 422 | `invalid_questions` |
| `questions-blank.json` | questions | 422 | `invalid_questions` |
| `questions-malformed.json` | questions | 400 | `invalid_json` |
| `document-malformed.json` | document | 400 | `invalid_json` |
| `document-empty.json` | document | 422 | `empty_document` |
| `document-scalar.json` | document | 422 | `invalid_document` |
| `document-no-text.json` | document | 422 | `empty_document` |
| `document-unsupported.txt` | document | 415 | `unsupported_file` |

The malformed JSON files are deliberately invalid; do not auto-fix their trailing commas. The numeric/boolean-only document also demonstrates a current limitation: the parser requires a nonempty string value somewhere, even when object keys are meaningful.

### Generated files under `output/example-fixtures/`

These expectations use the application's default limits. Overrides can change the result.

| File | Field | HTTP | Error code / behavior |
| --- | --- | --- | --- |
| `questions-invalid-utf8.json` | questions | 400 | `invalid_json` |
| `questions-limit-30.json` | questions | 200 | 30 ordered results; one unique question, so generation is reused |
| `questions-too-many-31.json` | questions | 422 | `invalid_questions` |
| `questions-too-long.json` | questions | 422 | `invalid_questions` (2,001 characters) |
| `questions-too-large.json` | questions | 413 | `file_too_large` (>256 KiB) |
| `document-empty-bytes.json` | document | 422 | `empty_file` (zero bytes) |
| `document-too-deep.json` | document | 413 | `json_too_deep` |
| `document-too-large.json` | document | 413 | `file_too_large` (>20 MiB; generated with `--large`) |
| `corrupt.pdf` | document | 422 | `invalid_document` (PDF header, corrupt body) |
| `not-a-pdf.pdf` | document | 415 | `invalid_pdf_signature` (plain text renamed `.pdf`) |

```sh
# Wrong questions schema: HTTP 422 / invalid_questions
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/qa \
  -F 'questions=@examples/failures/questions-records.json;type=application/json' \
  -F 'document=@examples/document.json;type=application/json'

# Corrupt PDF: HTTP 422 / invalid_document
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/qa \
  -F 'questions=@examples/questions.json;type=application/json' \
  -F 'document=@output/example-fixtures/corrupt.pdf;type=application/pdf'
```

Additional request-format failures do not need new files:

```sh
# Missing document: HTTP 422 / invalid_fields
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/qa \
  -F 'questions=@examples/questions.json;type=application/json'

# Extension says JSON, declared MIME says PDF: HTTP 415 / file_type_mismatch
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/qa \
  -F 'questions=@examples/questions.json;type=application/json' \
  -F 'document=@examples/document.json;type=application/pdf'

# Two documents are not supported: HTTP 400 / invalid_request (too many file parts)
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/qa \
  -F 'questions=@examples/questions.json;type=application/json' \
  -F 'document=@examples/document.json;type=application/json' \
  -F 'document=@examples/document.pdf;type=application/pdf'
```

## Operational failures: reproduce offline

No special question text in the real API triggers a timeout or outage. These cases require simulated provider behavior or configuration, so keep them in tests instead of paying for deliberately broken live calls.

| Scenario | Reproduction | Expected result |
| --- | --- | --- |
| All answer calls unavailable | `tests/test_examples.py::test_operational_failure_examples[unavailable-503-provider_unavailable]` | HTTP 503; `all_questions_failed`; each result has `status: error` / `provider_unavailable` |
| All answer calls time out | Same parametrized test, `timeout` case | HTTP 504; per-question `provider_timeout`, never `not_found` |
| Model fabricates a quotation | Same parametrized test, `bad_quote` case | HTTP 502; per-question `invalid_citation` |
| Some answers succeed, one fails | `tests/test_api.py::test_mixed_and_all_upstream_failures` | HTTP 200 with both usable answers and explicit per-question errors |
| Processing deadline after one completed answer | `tests/test_api.py::test_deadline_preserves_completed_answers` | Keep the completed answer; unfinished question becomes an error |
| Model refuses or returns invalid schema | `tests/test_provider.py::test_invalid_or_refused_output_is_not_not_found` | `invalid_model_output`, not missing document evidence |
| Provider 429 / 5xx | `tests/test_provider.py::test_transient_error_retried_once` | Retry at most once, then `provider_unavailable` |
| Substantive PDF image is unreadable | `tests/test_api.py::test_unreadable_visual_is_not_missing_evidence` | HTTP 422 / `unreadable_image`; no answer generation |
| Missing API key | `tests/test_api.py::test_health_readiness_openapi_and_static` | Health 200, readiness and QA 503 / `not_ready` |
| Concurrent-request capacity exhausted | `tests/test_api.py::test_concurrent_requests_share_capacity` | Excess requests get 503 / `busy` |

```sh
.venv/bin/pytest tests/test_examples.py -k operational -v
.venv/bin/pytest tests/test_provider.py tests/test_security.py -v
```

## What has and has not been verified

The new offline tests check shipped input formats, source locations, native/scanned PDF extraction, HTTP rejection codes, duplicate handling, isolation and simulated operational failures. The PDF pages are also rendered for visual inspection.

The expected JSON answer notes are **not canned API responses**. No live model calls were made to establish answer quality for these cases. In particular, conflicting evidence, prompt injection, ambiguous wording, and image interpretation still need manual live review. Do not mistake a passing mocked-provider test for a measured model capability.
