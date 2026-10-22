# OnDeviceRAG — Design

_Status: milestone 1 implemented and Mac-tested (2026-10-22); milestones 2–5 pending. Design reviewed and approved 2026-10-22 (with the milestone-1 deviations listed in `Docs/milestone-1-plan.md`). Author: David Ortega, with AI assistance._

## 1. Goal

A reusable Swift package that gives any iOS/macOS app **on-device retrieval-augmented
generation context**: the app hands it PDFs and a question, and gets back the most relevant
text, packed to fit a language model's prompt. Everything runs locally — no network calls.

First users:
- this research app (Prototype On-Device) — normal chat, and optionally a later research stage;
- a whiteboard AI tutor (algebra, speech + whiteboard answers);
- a coding-learning app.

Developed inside this repo as a local package (`Packages/OnDeviceRAG/`), then moved to its own
GitHub repo once it works (see §11).

### Non-goals (for now)
- Answer generation. The package returns context text; the app chooses and runs the model.
- Formats other than PDF (Markdown/LaTeX for exact math is a later decision — §10).
- Languages other than English (§10).
- Cloud anything.

## 2. Evidence behind the key decisions (spike, 2026-10-22)

Throwaway spike on branch `spike/embedding-iphone` (not merged), scored on the research
project's 40 tasks (140 chunks; correct spec known per question):

| Retriever | Correct spec in top-5 chunks (prompt as written / names removed) | In top-3 sections |
|---|---|---|
| Existing hybrid (BM25 + Apple `NLEmbedding`), Stage 1 | 34/40 / 29/40 | 31/40 |
| BM25 only (Python re-implementation) | 40/40 / 39/40 | 38/40 / 38/40 |
| `bge-small-en-v1.5` dense only, on iPhone via Core AI | **40/40 / 40/40** | **40/40 / 40/40** |

- `NLEmbedding` (Apple's sentence embedding) is **missing on iOS 27.0** and was the weak half of the
  old hybrid. `NLContextualEmbedding` is present on the iPhone (dim 512) but Apple recommends
  `NLEmbedding` for similarity, so it's only a fallback option.
- `bge-small-en-v1.5` (33M parameters, MIT license) exported with `coreai-models`' TorchConverter
  recipe runs on iPhone 16e (iOS 27.0) and Mac (macOS 27): cosine vs. PyTorch 0.9995–0.9999,
  64 MB at float16.
- **Flexible input length is too slow on iPhone** (~2.5 s per new length — Core AI re-prepares the
  model). Fixed-length export with 3 functions (`embed_128/256/512`) + attention mask: one-time
  load 11.9 s on iPhone (Core AI caches it; later launches expected faster — to confirm),
  then ~8 ms per chunk (140 chunks in 1.1 s), thermal Nominal.
- Caveat: these 40 tasks are now too easy to tell good retrievers apart — hence the harder
  benchmark in §9.

## 3. Architecture

Five components, each with one job, behind small protocols so they can be tested and swapped:

```
PDF ──► Ingest ──► [sections + chunks] ──► Embedder ──► Index (disk)
                                                          │
question (+ optional history) ──► Retriever ◄─────────────┘
                                     │ top ~20 chunks
                                     ▼
                       (optional) Reranker — Apple Foundation Model
                                     │
                                     ▼
                              Context packer ──► RAGContext (text for the prompt)
```

| Component | Job |
|---|---|
| **Ingest** | PDF → page text → sections → ~400-character chunks, with metadata |
| **Embedder** | text → 384-dim normalized vector (bundled bge-small via Core AI) |
| **Index** | chunks, vectors, BM25 statistics, persisted on disk; shipped + user-added documents |
| **Retriever** | dense (cosine) + BM25, fused with Reciprocal Rank Fusion → candidates |
| **Reranker** | optional Foundation Models re-ordering of the top candidates |
| **Context packer** | candidates → whole sections (or chunk windows) within a token budget |

## 4. Public API (sketch)

```swift
public actor RAGEngine {
    // Milestone 1: in memory. `storageDirectory` and `addPrebuiltIndex(at:)` arrive with
    // persistence in milestone 2.
    public init(configuration: RAGConfiguration)

    // Documents
    public func addDocument(pdfAt url: URL, title: String? = nil) async throws -> DocumentInfo
    public func addDocument(_ document: IngestedDocument) async throws -> DocumentInfo  // e.g. from DocumentIngestor.ingestText
    public func addPrebuiltIndex(at url: URL) async throws          // milestone 2: documents shipped with the app
    public func removeDocument(id: DocumentID) async throws
    public var documents: [DocumentInfo] { get }

    // Retrieval
    public func search(_ question: String, history: [String] = [], limit: Int = 20)
        async throws -> [RetrievedChunk]
    public func context(for question: String, history: [String] = [], tokenBudget: Int = 2000)
        async throws -> RAGContext
}

public struct RAGContext: Sendable {
    public let text: String                 // ready to insert under "Reference material:"
    public let sections: [ContextSection]   // title, document, pages, score — for citations
    public let tokenCount: Int
    public let tokenCountIsEstimate: Bool
}

public protocol Embedder: Sendable {
    var dimension: Int { get }
    func embed(_ texts: [String], as kind: EmbeddingKind) async throws -> [[Float]]  // .query / .passage
}
public protocol TokenCounter: Sendable { func count(_ text: String) async -> TokenCount }
public protocol Reranker: Sendable {
    func rerank(_ question: String, _ candidates: [RetrievedChunk]) async throws -> [RetrievedChunk]
}
```

`RAGConfiguration` holds every tunable constant (§8) plus the pluggable embedder, token counter and
reranker (`nil` = off). `history` is **optional and empty by default** — the research runs leave it out.

## 5. Ingest (PDF only)

1. **Page text:** `PDFPage.attributedString` (keeps fonts). If a page yields < 50 characters
   (scanned page), render it and run Vision `RecognizeDocumentsRequest` (paragraphs, lists, tables)
   — **milestone 2**; milestone 1 lists these pages in `DocumentInfo.pagesWithoutText`.
   Plain text is also accepted (`DocumentIngestor.ingestText`: sections at `## ` lines, code inside
   ``` fences) — for apps whose sources are text, and for tests.
2. **Sections**, first that applies: the PDF outline (`PDFDocument.outlineRoot`); else headings
   detected from font size (≥ 1.2 × median body size) or bold short lines (< 100 characters);
   else ~1,500-character windows.
3. **Chunks:** ≤ 400 characters, built from whole sentences within a section.
   - **Code is never split and keeps its indentation** — lines in a monospaced font form one block
     (a block longer than 400 characters stays one chunk).
   - Each chunk records document ID, section index/title, page range and its position in the section.
4. **Math:** extracted text may garble equations. Ingest flags pages with a high share of math
   symbols (`metadata.mathHeavy`) so the benchmark can measure the damage. No fix yet (§10).

## 6. Embedder, Index and Retriever

**Embedder (default `BGESmallEmbedder`):**
- Model: `bge-small-en-v1.5`, float16, CLS pooling + L2 normalization baked into the export;
  functions `embed_128`, `embed_256`, `embed_512`; inputs `input_ids` + `attention_mask` (int32).
  Texts are padded to the nearest length; longer texts are truncated at 512 tokens.
- Query texts get bge's instruction prefix (`Represent this sentence for searching relevant passages: `);
  passages don't.
- Tokenizer: `swift-transformers` BERT (WordPiece) tokenizer from the model's `tokenizer.json`.
  **Must produce byte-identical token IDs to Python** (unit test).
- Export script lives in `Packages/OnDeviceRAG/Tools/export_embedder.py` (from the spike), pinned to
  the same `coreai-models` toolchain versions.

**Index:**
- On disk per document: `manifest.json` (document info, settings, embedder ID + revision),
  `pages.json` (extracted page text), `chunks.json`, `vectors.f16` (contiguous float16, chunk order).
- An index built with a different embedder ID/revision or chunk settings is rebuilt from the stored
  page text (no need for the original PDF).
- Search is brute-force cosine with Accelerate (`vDSP`): fine up to ~50k chunks
  (≈ 20M multiply-adds per query, milliseconds).
- **Prebuilt indexes:** a macOS command-line tool in the package (`rag-index`) turns PDFs into an
  index folder the app ships as a resource — same code path as on-device, so results are identical
  and the user waits for nothing on first launch.

**Retriever:**
- Query text = the question, or, when `history` is given, the question followed by the last ≤ 2
  history messages, most recent first — so the embedder's 512-token truncation only ever cuts history,
  never the question.
- Dense ranking (cosine) and BM25 ranking (the research app's `BM25Index` logic, **copied** into the
  package — the app's file stays unchanged because batch Stages 2/3 depend on it: k1 1.2, b 0.75,
  tokens `[a-z0-9_]+`), fused with RRF (k = 60) → top 20 candidates.

**Reranker (optional, off by default):**
- `FoundationModelsReranker`: one `LanguageModelSession` call with the question and the top 10
  candidates (numbered, each truncated to ~400 characters), guided generation into
  `@Generable struct Ranking { var order: [Int] }`. Invalid or partial output → keep the original order.
- Skipped automatically when `SystemLanguageModel.default` is unavailable (Apple Intelligence off).
- Default stays **off** unless the benchmark passes the rule in §9.

## 7. Context packer

1. Walk candidates in rank order; map each to its section.
2. Add the **whole section** if it fits the remaining budget.
3. If the section alone is larger than the remaining budget, add a **window**: the matched chunk plus
   neighbors on both sides (alternating) until the budget is used.
4. Skip sections already included; stop when the budget is full or candidates run out.
5. The top-ranked chunk is always included (truncated if the budget is tiny).
6. Output format, sections in rank order, blank line between them:
   ```
   ## <section title> (<document title>, p. <first>–<last>)
   <text>
   ```
7. **Token counting:** default `AppleTokenCounter` uses `SystemLanguageModel.default.tokenCount(for:)`
   when available; otherwise `EstimatingTokenCounter` (characters ÷ 3, deliberately conservative) and
   `tokenCountIsEstimate = true`. Apps using other models can supply their own counter.

## 8. Configuration defaults

| Setting | Default |
|---|---|
| chunk size | 400 characters |
| fallback section window | 1,500 characters |
| embedding lengths | 128 / 256 / 512 tokens |
| BM25 k1 / b | 1.2 / 0.75 |
| RRF k | 60 |
| candidates | 20 |
| reranker | off (top 10 when on) |
| history messages used | ≤ 2 |
| token budget | 2,000 |

## 9. Testing and evaluation

**Unit tests (Swift Testing, macOS):** chunker (code blocks never split, page ranges, sentence
boundaries), section detection on fixture PDFs, BM25 and RRF rankings, packer never exceeds the budget
and always includes the top chunk, tokenizer IDs identical to Python, embedder parity
(cosine ≥ 0.999 vs. reference vectors), index save/load round trip and embedder-mismatch rejection.

**Retrieval benchmark (macOS, `Benchmarks/`):**
- Documents: openly licensed textbooks covering both domains — e.g. OpenStax *College Algebra*
  (CC BY) and an openly licensed Python textbook. A script downloads them; PDFs aren't committed
  unless the license allows it.
- 60–100 questions with their correct section and page; a subset of follow-up questions with a
  preceding message (tests `history`). Drafted with AI assistance, reviewed by the researcher,
  frozen before any results.
- Metrics: section Recall@1 / Recall@3, MRR; also chunk Hit@5. Conditions: dense only, BM25 only,
  hybrid, hybrid + reranker; with / without history on the follow-up subset.
- **Reranker rule (fixed before results):** on by default only if it raises section Recall@3 by
  ≥ 5 percentage points over hybrid **and** adds ≤ 3 s per question (p95) on the iPhone.

**On-device performance (iPhone 16e and Mac):** first and later model load, time to index a
~100-page PDF, time per question (search, rerank, pack), peak memory, thermal state.

**Research (separate decision, later):** a possible exploratory "Stage 4" pass@1 run on the 40 tasks
with this pipeline, with its own pre-specified plan. The frozen oracle protocol and existing
results are not affected.

## 10. Known limitations and open questions

- **iOS 27 / macOS 27 only** (Core AI). The `Embedder` protocol allows another backend.
- **English-only embedder.** Spanish content needs a multilingual model (e.g. `multilingual-e5-small`,
  ~240 MB at float16; its 250k-token vocabulary needs a slightly different export). Test before promising.
- **Adds ~64 MB** to an app.
- **PDFs set in Apple's system font** (e.g. exported from Pages/Notes) lose bold/monospace when read
  back: PDFKit reports the private `.SF…` font as Times New Roman (found 2026-10-22 in milestone 1).
  Headings are then detected by size only and code isn't recognized as code. Named fonts
  (Helvetica, Menlo, Courier…) are read correctly. Check with real PDFs in the milestone-3 benchmark.
- **Math in PDFs** may extract badly. Decide later with the whiteboard team whether to also accept
  Markdown/LaTeX source.
- **Neural Engine:** the int32 inputs keep the model on the GPU (Core AI warns
  "Incompatible element type for ANE"). int16 IDs + float16 mask may allow the ANE — optional
  optimization; speed is already fine.
- **Model distribution in the public repo:** the 63 MB weights file is under GitHub's 100 MB limit,
  but this repo ignores `*.aimodel`. Options for the standalone repo: commit it directly, Git LFS, or a
  release asset fetched by a script. To decide at extraction time.
- Cached Core AI specialization across launches on iPhone — to confirm.

## 11. Milestones

1. **Core package:** Ingest, Embedder, Index (in memory), Retriever, Context packer, `Reranker`
   protocol (the Foundation Models implementation comes with the benchmark in milestone 3) + unit
   tests (Mac). OCR for scanned pages moves to milestone 2. Plan: `Docs/milestone-1-plan.md`.
2. **Persistence and shipping:** on-disk index, `rag-index` CLI, prebuilt indexes; integrate into this
   app's normal chat (replacing `KnowledgeRetriever`'s internals without changing its call sites —
   Architecture Rule #4). Batch oracle mode untouched.
3. **Benchmark:** build the harder test set; evaluate conditions; decide the reranker default.
4. **On-device verification** on iPhone and Mac.
5. **Extraction:** move `Packages/OnDeviceRAG/` to its own GitHub repo (with history), tag `0.1.0`,
   switch this app to the URL dependency; README with the model-agnostic usage and limitations.

Branching: milestones 1–4 on `feature/ondevice-rag` (from `main` after `feature/rag` is merged), merged
via PR. The spike branch is discarded.
