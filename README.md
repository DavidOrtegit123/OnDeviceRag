# OnDeviceRAG

On-device retrieval-augmented generation context for iOS and macOS apps. Give it PDFs (or text)
and a question; it returns the most relevant text, packed to fit your language model's prompt.
Everything runs locally — no network calls.

It does **not** generate answers. The returned text goes into the prompt of whatever model you
use: Apple's Foundation Model, an open model through Core AI, MLX, llama.cpp, or a server.

## How it works

1. **Ingest** — PDF text is read with its fonts, split into sections (the PDF's table of contents,
   else large/bold headings, else ~1,500-character windows), then into chunks of ≤ 400 characters
   made of whole sentences. Code blocks are never split.
2. **Search** — every chunk is ranked two ways and the rankings are merged (Reciprocal Rank Fusion):
   by meaning, with the bundled `bge-small-en-v1.5` embedding model (via Core AI), and by exact
   words, with BM25 (good at names like `encode_zorp`).
3. **Pack** — the best matches' **whole sections** are added until a token budget (default 2,000)
   is full; a section too big for the budget contributes a window around the matching chunk.

## Requirements

- iOS 27 / macOS 27 (Core AI), Swift 6.4 tools.
- The embedding model `bge_small_buckets_float16.aimodel` (64 MB), exported with
  `Tools/export_embedder.py` (`uv run export_embedder.py float16`). It is not in the repository;
  add it to your app's bundle. The tokenizer files are bundled with the package.

## Usage

```swift
import OnDeviceRAG

let modelURL = Bundle.main.url(forResource: "bge_small_buckets_float16", withExtension: "aimodel")!
let engine = RAGEngine(configuration: RAGConfiguration(embedder: try await BGESmallEmbedder(modelURL: modelURL)))
_ = try await engine.addDocument(pdfAt: pdfURL)

let question = "How do I encode a list?"
let context = try await engine.context(for: question)   // optionally: history: [earlier messages]
let prompt = "Reference material:\n\(context.text)\n\nQuestion: \(question)"
```

`context.sections` lists the sections used (title, document, pages) for citations.
`DocumentIngestor.ingestText(_:title:)` accepts plain text (`## ` headings, ``` code fences).
All settings live in `RAGConfiguration`; the embedder, token counter and re-ranker are protocols,
so you can plug in your own.

## Saving and shipping indexes

```swift
// Saved between launches (re-embedding only when settings or the embedder change):
let engine = try await RAGEngine(configuration: configuration, storageDirectory: appSupportURL.appending(path: "RAG"))
// Documents shipped with the app, built on a Mac with `rag-index build`:
try await engine.addPrebuiltIndex(at: Bundle.main.url(forResource: "textbook", withExtension: "ragindex")!)
```

- With a storage directory, each document is saved as one file (`<id>.ragdoc`). On the next launch
  it's loaded without re-embedding. If the chunk settings changed, it's re-chunked from its saved
  source (the original PDF isn't needed) and re-embedded; if the embedder changed, it's re-embedded.
  Use one engine per storage directory.
- A prebuilt index (`.ragindex`, one file) is **read-only**: its documents are searchable but can't
  be removed or replaced (`RAGError.prebuiltDocument`) and are never written to the storage
  directory. If it was built with other settings or another embedder, it's rebuilt in memory on
  every launch — rebuild the file instead. A shipped document whose ID is already loaded is skipped.
- Files that can't be read (damaged, newer format version) and skipped documents are listed in
  `engine.loadIssues`; they never stop the engine from opening.
- `isPrebuilt(_:)` tells shipped documents from the user's; `exportIndex(to:)` writes the user's
  documents into a `.ragindex`.

Build and check an index on a Mac with the same code the apps run:

```bash
M=~/Developer/coreai-models/exports/bge_small_buckets_float16.aimodel
swift run -c release --package-path Packages/OnDeviceRAG rag-index build --model $M --output textbook.ragindex chapter1.pdf notes.md
swift run -c release --package-path Packages/OnDeviceRAG rag-index query --model $M --index textbook.ragindex [--budget 2000] "How do I encode a list?"
```

Document IDs are the input file names without extension, so a rebuild keeps the same IDs.
`.md`/`.txt` inputs are read as text (`## ` headings). The index format is one file on purpose:
Xcode flattens resource folders into the app bundle.

## Limitations

- **English only** (the embedding model). Other languages need a multilingual model.
- **PDF and plain text only.** Scanned pages are reported (`DocumentInfo.pagesWithoutText`) but not
  yet read (OCR is planned). Math may extract badly; pages heavy in math symbols are flagged
  (`DocumentInfo.mathHeavyPages`).
- PDFs set in Apple's **system font** lose bold/monospace information when read back, so headings
  are found by size only and code isn't recognized as code.
- Token counts are exact for Apple's model when Apple Intelligence is available; otherwise a
  conservative estimate (characters ÷ 3). Supply a `TokenCounter` for other models.
- **Exact token counting is slow:** with Apple's counter, `context(for:)` took ~2.2–2.8 s on a Mac
  (search itself: 7–36 ms), because each candidate section is counted with Foundation Models.
  `EstimatingTokenCounter` avoids it. A fix is planned.
- Adds ~64 MB (the embedding model) to your app.

## Tests

```bash
swift test --package-path Packages/OnDeviceRAG
```

Tests that need the embedding model look for it at `ONDEVICE_RAG_MODEL`, defaulting to
`~/Developer/coreai-models/exports/bge_small_buckets_float16.aimodel`, and are skipped if it's
missing. `Tools/make_package_fixtures.py` regenerates the tokenizer files and the Python reference
values the tests compare against.
