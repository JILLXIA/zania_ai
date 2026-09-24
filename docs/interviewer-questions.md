# Design Decisions and Optional Submission Questions

Status: The earlier technical questions have been resolved by the user's clarifications, request to include PDF image analysis, and instruction to adopt the remaining proposed defaults. This document replaces the earlier question list and draft email, which contained an incorrect interpretation of the sample JSON document. No interviewer response is required to proceed with the [revised design](design-plan.md).

## Confirmed input and answer behavior

- Each request uploads exactly two files: one questions JSON file, and one document file that is either PDF or JSON. It never uploads both document formats together.
- The questions file is independently supplied as an array of strings, for example:

```json
[
  "Which cloud providers do you rely on?",
  "Is personal information disclosed to third parties?"
]
```

- The supplied CSV represents the JSON document's knowledge-base records. Convert its complete records to JSON, preserving `id`, `question`, `answer`, `comments`, and `confidence`. The records' question fields are source context, not the request's question list.
- The PDF and JSON sample are separate evidence sources. Each response uses only its request's uploaded document. Source answers/comments are evidence in JSON requests, and cannot be reused for PDF requests.
- For partial support, the same `answer` string states supported facts and the details not found. Keep citations for the supported portion. An additional `status: "partial"` is allowed but does not replace that explanation.
- When nothing is supported, return exactly `Not found in document` with empty citations. Provider failures remain operational errors, not missing-evidence answers.
- PDF processing combines native text extraction with `gpt-4o-mini` image analysis for selected diagrams, images, and scanned pages. This supersedes the earlier text-only choice. Read visible text and relationships through that same model; do not add a separate OCR or multimodal model.
- Visual observations are indexed once per request and reused across questions. Their citations include the original PDF page and `source_type: "image"`; the UI distinguishes model-extracted visual descriptions from native-text quotes.

## Accepted defaults

| Topic | Decision |
| --- | --- |
| Embeddings and generation | Local FastEmbed `BAAI/bge-small-en-v1.5` embeddings, in-memory FAISS, and `gpt-4o-mini` for every OpenAI call, including visual analysis. No hosted embedding API. |
| JSON support | Support the sample's full records plus nested objects/arrays containing meaningful text; preserve record context and cite JSON Pointer paths with excerpts. |
| Response schema | Preserve `results`, `question`, `answer`, and `citations`; add request ID, result status, citation source type, optional errors, and document warnings as documented extensions. |
| Input limits | 20 MiB document, 200 PDF pages, 30 questions, two active requests per process, and a 180-second processing deadline; retain the detailed safeguards in the design. |
| Visual-analysis limits | At most 10 selected visual pages, bounded rendering, a 120-second visual stage within the overall deadline, and a 350,000 estimated input-token budget across visual attempts. Reject overflow explicitly before visual processing. |
| Visual-analysis failures | Required visual-ingestion failure is a clear request error, not a silent fallback to native text or `Not found in document`. |
| Partial operational failures | Preserve usable results and explicit errors for failed questions. Return an appropriate non-2xx response if all processing fails operationally. |
| Model provisioning | Prepare the local embedding model during Docker build or the documented local setup step, and include a tested pypdfium2 renderer. Offline tests mock visual/generation calls and need no model download. |
| Delivery | One FastAPI service with a minimal same-origin upload page, local Docker instructions, and no hosted deployment requirement in this design. |
| Referenced external material | Do not fetch URLs or additional files mentioned inside source records. Their absence does not block processing of the uploaded document. |

These decisions record the user's chosen implementation approach; they are not attributed to an interviewer reply. A later explicit requirement change can be incorporated without treating the previous question list as an outstanding approval gate.

## Optional submission logistics

Only confirm these if they have not already been communicated; they do not block local design or implementation:

1. What is the submission deadline or intended timebox?
2. If the repository is private, which reviewer account should receive access?

No technical clarification email needs to be sent based on this plan. Implementation has now been completed locally; see [verification notes](verification.md) for actual checks and remaining external validation. No message has been sent and no repository has been published.
