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

## Limitations

- **English only** (the embedding model). Other languages need a multilingual model.
- **PDF and plain text only.** Scanned pages are reported (`DocumentInfo.pagesWithoutText`) but not
  yet read (OCR is planned). Math may extract badly; pages heavy in math symbols are flagged
  (`DocumentInfo.mathHeavyPages`).
- PDFs set in Apple's **system font** lose bold/monospace information when read back, so headings
  are found by size only and code isn't recognized as code.
- The index lives **in memory**: documents are re-embedded on each launch (saving to disk is planned).
- Token counts are exact for Apple's model when Apple Intelligence is available; otherwise a
  conservative estimate (characters ÷ 3). Supply a `TokenCounter` for other models.
- Adds ~64 MB (the embedding model) to your app.

## Tests

```bash
swift test --package-path Packages/OnDeviceRAG
```

Tests that need the embedding model look for it at `ONDEVICE_RAG_MODEL`, defaulting to
`~/Developer/coreai-models/exports/bge_small_buckets_float16.aimodel`, and are skipped if it's
missing. `Tools/make_package_fixtures.py` regenerates the tokenizer files and the Python reference
values the tests compare against.
