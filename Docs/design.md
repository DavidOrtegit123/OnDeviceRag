# OnDeviceRAG — Design

_Status: milestones 1, 2a (persistence, prebuilt indexes, `rag-index`) and 2b (OCR for scanned pages) implemented and Mac-tested (2026-10-22 / 2026-10-25); extracted to its own repository and released as `0.1.0` (2026-10-25); milestones 3–4 pending. Design reviewed and approved 2026-10-22 (with the milestone-1 deviations listed in `Docs/milestone-1-plan.md`). Author: David Ortega, with AI assistance._

## 1. Goal

A reusable Swift package that gives any iOS/macOS app **on-device retrieval-augmented
generation context**: the app hands it PDFs and a question, and gets back the most relevant
text, packed to fit a language model's prompt. Everything runs locally — no network calls.

First users:
- a research app comparing on-device language models (its normal chat);
- a whiteboard AI tutor (algebra, speech + whiteboard answers);
- a coding-learning app.

Developed first as a local package inside the research app's repository, then extracted to this
repository with its history (§11). The milestone plans in `Docs/` were written during that phase, so
their file paths refer to that repository.

### Non-goals (for now)
- Answer generation. The package returns context text; the app chooses and runs the model.
- Formats other than PDF (Markdown/LaTeX for exact math is a later decision — §10).
- Languages other than English (§10).
- Cloud anything.

## 2. Evidence behind the key decisions (spike, 2026-10-22)

Throwaway spike, scored on a 40-document test set from the author's research project: 40 short
fictional API/format specifications in 4 manuals (140 chunks) and 80 questions (as written / with
names removed), the correct spec known for each:

| Retriever | Correct spec in top-5 chunks (prompt as written / names removed) | In top-3 sections |
|---|---|---|
| Earlier hybrid (BM25 + Apple `NLEmbedding`) | 34/40 / 29/40 | 31/40 |
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
    public init(configuration: RAGConfiguration)                                    // in memory
    public init(configuration: RAGConfiguration, storageDirectory: URL) async throws // saved to disk (2a)

    // Documents
    public func addDocument(pdfAt url: URL, title: String? = nil) async throws -> DocumentInfo
    public func addDocument(_ document: IngestedDocument) async throws -> DocumentInfo  // e.g. from DocumentIngestor.ingestText
    public func removeDocument(id: DocumentID) throws
    public var documents: [DocumentInfo] { get }

    // Shipped documents (2a): a read-only `.ragindex` file built with `rag-index`
    @discardableResult public func addPrebuiltIndex(at url: URL) async throws -> [DocumentInfo]
    public func exportIndex(to url: URL) throws        // the user's (non-prebuilt) documents
    public func isPrebuilt(_ id: DocumentID) -> Bool
    public var loadIssues: [String] { get }            // unreadable files, skipped ID clashes

    // Retrieval — history (0.2.0): oldest first; used only when switched on in `configuration.history`
    public func search(_ question: String, history: [ChatTurn] = [], limit: Int = 20)
        async throws -> [RetrievedChunk]
    public func context(for question: String, history: [ChatTurn] = [], tokenBudget: Int = 2000)
        async throws -> RAGContext

    // 0.2.0: new title, re-chunked from the stored source
    @discardableResult public func renameDocument(id: DocumentID, to title: String) async throws -> DocumentInfo
}

public struct RAGContext: Sendable {
    public let text: String                 // ready to insert under "Reference material:"
    public let sections: [ContextSection]   // title, document, pages, score — for citations
    public let tokenCount: Int
    public let tokenCountIsEstimate: Bool
    public var history: [ChatTurn]          // 0.2.0: recent turns that fit history.tokenBudget
    public var historyTokenCount: Int
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
reranker (`nil` = off). **History (0.2.0):** `configuration.history` has two switches, both off by
default — `useForSearch` and `includeInContext` — so passing history changes nothing until an app
turns them on. Details: `Docs/0.2.0-history-design.md`.

## 5. Ingest (PDF only)

1. **Page text:** `PDFPage.attributedString` (keeps fonts). **Scanned pages (2b, implemented):** a
   page with < 50 visible characters is rendered at 2× (≤ 4,000 px, rotation applied) and read with
   Vision `RecognizeDocumentsRequest`. Its paragraphs (lists arrive inside them) become `TextLine`s
   marked `isOCR`, with a blank line between paragraphs. Sizes are snapped: a line under 1.3 × the
   page's lower-median line height is body text (11 pt); taller lines keep their ratio (heading
   candidates); the recognized document title is marked bold. The OCR text replaces the text layer
   only if it has **more than twice** the layer's characters, so short typed pages (a chapter title,
   a slide) keep their fonts. OCR lines are stored in the document's source, so re-chunking never
   re-runs OCR. `DocumentInfo.ocrPages` lists OCR'd pages; `pagesWithoutText` lists pages still
   under 50 characters after OCR. A failing page never fails the document; only cancellation throws.
   Switch: `RAGConfiguration.ocrScannedPages` (default on) — not in `IngestSettings`, because a
   settings change re-chunks from the source, which can't re-run OCR. Plan: `Docs/milestone-2b-plan.md`.
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
- On disk: **one JSON file per document** (`<id>.ragdoc`, written atomically) holding its info,
  sections, chunks, embedder ID + dimension, chunk settings, the date added, its **source** (the
  extracted lines with font facts + outline for PDFs, or the original text) and its vectors as Float16
  (base64; ~1 MB for a 1,000-chunk textbook). Format version 1. *(Changed in 2a from the first draft's
  folder of four files per document: Xcode flattens resource folders into the app bundle (seen with a
  Core AI model folder), so a folder-based shipped index would collide; a single file can't be
  flattened.)*
- Rebuild rules on load: chunk settings changed → re-chunk from the stored source and re-embed (no need
  for the original PDF); embedder changed or vectors damaged → re-embed; otherwise use as stored.
  Rebuilt user documents are saved back; prebuilt ones stay in memory only.
- Unreadable or newer-format files are listed in `loadIssues` and skipped — they never block opening.
- Search is brute-force cosine with Accelerate (`vDSP`): fine up to ~50k chunks
  (≈ 20M multiply-adds per query, milliseconds).
- **Prebuilt indexes:** a macOS command-line tool in the package (`rag-index build`) turns PDFs (or
  `.md`/`.txt`) into **one `.ragindex` file** (all its documents, same record format) the app ships as
  a resource — same code path as on-device, so results are identical and the user waits for nothing
  on first launch. Document IDs = input file names, so rebuilds keep IDs. Prebuilt documents are
  read-only (can't be removed or replaced) and never written to the storage directory; a shipped
  document whose ID is already loaded is skipped. `rag-index query` prints the packed context.
  Measured on the 4 test manuals of §2 (Mac): model load 0.15–1.1 s, 0.15–0.22 s per manual, 202 KB index.

**Retriever:**
- Query text = the question, or, when `history.useForSearch` is on, the question followed by the last
  ≤ `searchTurns` (2) **user** turns, most recent first — so the embedder's 512-token truncation only
  ever cuts history, never the question. Assistant turns are never searched.
- Dense ranking (cosine) and BM25 ranking (k1 1.2, b 0.75, tokens `[a-z0-9_]+`), fused with RRF
  (k = 60) → top 20 candidates.

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
| history (both off by default) | search: ≤ 2 user turns; prompt: ≤ 6 turns within 600 tokens |
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
  preceding message (tests `history`). Drafted with AI assistance, reviewed by the author,
  frozen before any results.
- Metrics: section Recall@1 / Recall@3, MRR; also chunk Hit@5. Conditions: dense only, BM25 only,
  hybrid, hybrid + reranker; with / without history on the follow-up subset.
- **Reranker rule (fixed before results):** on by default only if it raises section Recall@3 by
  ≥ 5 percentage points over hybrid **and** adds ≤ 3 s per question (p95) on the iPhone.

**On-device performance (iPhone 16e and Mac):** first and later model load, time to index a
~100-page PDF, time per question (search, rerank, pack), peak memory, thermal state.

## 10. Known limitations and open questions

- **iOS 27 / macOS 27 only** (Core AI). The `Embedder` protocol allows another backend.
- **English-only embedder.** Spanish content needs a multilingual model (e.g. `multilingual-e5-small`,
  ~240 MB at float16; its 250k-token vocabulary needs a slightly different export). Test before promising.
- **Adds ~64 MB** to an app.
- **OCR (2b):** ~0.2 s per page on a Mac; the first OCR in a newly built or installed app took ~47 s
  (system model preparation, seen on the Mac — iPhone unmeasured). Misreads lone letters (`l` → `I`)
  and math spacing (`x = y + 2` → `x=y +2`) — relevant for the algebra tutor. Tables aren't handled
  separately; whether table text arrives in `paragraphs` is unverified (check with real scans in
  milestone 3). Tested only on synthetic scans so far; no real scanned PDF has been tried.
- **All-monospace PDFs** (e.g. a plain-text file printed with `cupsfilter`, Monaco font; PDFKit also
  drops their blank lines) were one unsplit code block per section: a 7,000-character manual became
  4 chunks of 500–1,950 characters (found 2026-10-25). **Fixed (2026-10-25):** when ≥ 80% of a
  document's visible characters are monospaced (`DocumentIngestor.mostlyMonospacedShare`), every
  line is treated as ordinary text (decided per document, so a full-page listing in a code book stays
  code); the stored source keeps the original lines and the rule re-runs on every re-chunk. Same
  manual: 20 chunks, max 398 characters. Remaining: code listings in such PDFs are split like prose,
  and their headings aren't found (uniform size/weight) → ~1,500-character window sections.
- **PDFs set in Apple's system font** (e.g. exported from Pages/Notes) lose bold/monospace when read
  back: PDFKit reports the private `.SF…` font as Times New Roman (found 2026-10-22 in milestone 1).
  Headings are then detected by size only and code isn't recognized as code. Named fonts
  (Helvetica, Menlo, Courier…) are read correctly. Check with real PDFs in the milestone-3 benchmark.
- **Math in PDFs** may extract badly. Decide later with the whiteboard team whether to also accept
  Markdown/LaTeX source.
- **Neural Engine:** the int32 inputs keep the model on the GPU (Core AI warns
  "Incompatible element type for ANE"). int16 IDs + float16 mask may allow the ANE — optional
  optimization; speed is already fine.
- **Model distribution (decided 2026-10-25):** the 64 MB model isn't committed (`*.aimodel` is
  ignored); it's attached to each GitHub release and fetched with `Tools/download_model.sh` (`gh` if
  installed, else `curl`). The repository is public, so no login is needed.
- Cached Core AI specialization across launches on iPhone — to confirm.
- **Exact token counting is slow per call** (found 2026-10-25 with `rag-index query`, Mac): Foundation
  Models' `tokenCount(for:)` costs ~48 ms per call regardless of length, and packing made ~50 calls,
  so `context(for:)` took ~2.2–2.8 s while search took 7–36 ms. **Fixed (2026-10-25):** the engine
  wraps the counter in `CachingTokenCounter` (exact counts only — estimates are never cached), counts
  the block separator once per pack, and pre-counts each document's whole-section blocks in the
  background when it's added or loaded. Mac: repeated question 0.014 s, new question after pre-counting
  0.43 s, packed text unchanged; a question during pre-counting is still ~2.5 s. Counts are not saved
  to disk (a `TokenCounter` has no identity, so a stored count could be reused with another model's
  counter) — revisit if the iPhone pre-count turns out too slow. iPhone not yet measured.

## 11. Milestones

1. **Core package:** Ingest, Embedder, Index (in memory), Retriever, Context packer, `Reranker`
   protocol (the Foundation Models implementation comes with the benchmark in milestone 3) + unit
   tests (Mac). OCR for scanned pages moves to milestone 2. Plan: `Docs/milestone-1-plan.md`.
2. **Persistence, shipping and app integration**, in three plans:
   - **2a — done (2026-10-25):** on-disk index with rebuild rules, prebuilt read-only indexes,
     `rag-index` CLI. Plan: `Docs/milestone-2a-plan.md`.
   - **2b — done (2026-10-25):** Vision OCR for scanned pages (package only). Plan:
     `Docs/milestone-2b-plan.md`.
   - **2c:** integration into the research app's normal chat — planned and done in that app's
     repository, using this package as a dependency.
3. **Benchmark:** build the harder test set; evaluate conditions; decide the reranker default.
4. **On-device verification** on iPhone and Mac.
5. **Extraction — done (2026-10-25, moved ahead of 3–4):** moved to its own GitHub repository with its
   history, tagged `0.1.0` (0.x: the API may still change). Apps depend on tagged releases.
6. **0.2.0 — conversation history (switchable) and `renameDocument`.** Design:
   `Docs/0.2.0-history-design.md`; plan: `Docs/0.2.0-history-plan.md`. Changes: `CHANGELOG.md`.
