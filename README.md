# OnDeviceRAG

On-device retrieval-augmented generation context for iOS and macOS apps. Give it PDFs (or text)
and a question; it returns the most relevant text, packed to fit your language model's prompt.
Everything runs locally — no network calls.

It does **not** generate answers. The returned text goes into the prompt of whatever model you
use: Apple's Foundation Model, an open model through Core AI, MLX, llama.cpp, or a server.

## How it works

1. **Ingest** — PDF text is read with its fonts (pages without a text layer — scans — are read
   with Vision OCR), split into sections (the PDF's table of contents,
   else large/bold headings, else ~1,500-character windows), then into chunks of ≤ 400 characters
   made of whole sentences. Code blocks are never split.
2. **Search** — every chunk is ranked two ways and the rankings are merged (Reciprocal Rank Fusion):
   by meaning, with the bundled `bge-small-en-v1.5` embedding model (via Core AI), and by exact
   words, with BM25 (good at names like `encode_zorp`).
3. **Pack** — the best matches' **whole sections** are added until a token budget (default 2,000)
   is full; a section too big for the budget contributes a window around the matching chunk.

## Requirements

- iOS 27 / macOS 27 (Core AI), Swift 6.4 tools.
- The embedding model `bge_small_buckets_float16.aimodel` (64 MB). It isn't in the repository; each
  release has it attached. The tokenizer files are bundled with the package.

## Installation

1. In Xcode: **File › Add Package Dependencies…**, enter `https://github.com/DavidOrtegit123/OnDeviceRag`,
   choose "Up to Next Minor Version" from `0.2.0` (0.x releases may change the API), and add the
   `OnDeviceRAG` library to your app target.
2. Download the model and add it to your app target (drag the `.aimodel` folder into Xcode, with your
   app target ticked):

   ```bash
   Tools/download_model.sh <your-app-folder>     # uses gh if installed, else curl; no login needed
   ```

   Or rebuild it yourself with `Tools/export_embedder.py` (`uv run export_embedder.py float16`).

## Usage

```swift
import OnDeviceRAG

let modelURL = Bundle.main.url(forResource: "bge_small_buckets_float16", withExtension: "aimodel")!
let engine = RAGEngine(configuration: RAGConfiguration(embedder: try await BGESmallEmbedder(modelURL: modelURL)))
_ = try await engine.addDocument(pdfAt: pdfURL)

let question = "How do I encode a list?"
let context = try await engine.context(for: question)
let prompt = "Reference material:\n\(context.text)\n\nQuestion: \(question)"
```

### Conversation history (optional)

```swift
var configuration = RAGConfiguration(embedder: embedder)
configuration.history.useForSearch = true      // follow-up questions find the right section
configuration.history.includeInContext = true  // recent turns returned for your prompt
let engine = RAGEngine(configuration: configuration)

let history: [ChatTurn] = [.user("What is ZORP-7?"), .assistant("A serialization format…")]
let context = try await engine.context(for: "And how are lists encoded?", history: history)
let prompt = """
Conversation so far:
\(ChatTurn.format(context.history))

Reference material:
\(context.text)

Question: And how are lists encoded?
"""
```

Both switches are **off by default**: without them, passing history changes nothing. Search uses
only the last `history.searchTurns` (2) user turns; `context.history` holds the latest turns that fit
`history.tokenBudget` (600 tokens, at most `history.maximumTurns` = 6, oldest dropped first). The
package doesn't store conversations — your app keeps the chat log and passes it in.
Without calling `context(for:)` (your own retrieval, a pinned document), use
`ChatTurn.recent(history, settings: configuration.history)` for the same trimming.

`renameDocument(id:to:)` gives a document a new title (re-chunked from its saved source).
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
swift run -c release rag-index build --model $M --output textbook.ragindex chapter1.pdf notes.md
swift run -c release rag-index query --model $M --index textbook.ragindex [--budget 2000] "How do I encode a list?"
```

Document IDs are the input file names without extension, so a rebuild keeps the same IDs.
`.md`/`.txt` inputs are read as text (`## ` headings). The index format is one file on purpose:
Xcode flattens resource folders into the app bundle.

## Limitations

- **English only** (the embedding model). Other languages need a multilingual model.
- **PDF and plain text only.** Math may extract badly; pages heavy in math symbols are flagged
  (`DocumentInfo.mathHeavyPages`).
- **Scanned pages are read with Vision OCR** (`DocumentInfo.ocrPages`): ~0.2 s per page on a Mac.
  The first OCR in a newly built or installed app can take ~45 s while the system prepares its
  model (seen on the Mac; not yet measured on an iPhone). OCR replaces a page's text only when the
  page has < 50 characters of text and OCR finds more than twice as much, so short typed pages keep
  their fonts. OCR can misread lone letters (`l` → `I`) and drops spacing in math (`x = y + 2` →
  `x=y +2`); tables aren't handled specially. Pages still without text are listed in
  `DocumentInfo.pagesWithoutText`. Turn OCR off with `RAGConfiguration.ocrScannedPages = false`.
- **PDFs set (almost) entirely in a monospaced font** (≥ 80% of characters, e.g. a plain-text file
  printed to PDF) are read as ordinary text, since the font can't mark code there; code listings in
  such PDFs are therefore split like prose. Their headings usually aren't found either (every line
  has the same size and weight), so sections fall back to ~1,500-character windows.
- PDFs set in Apple's **system font** lose bold/monospace information when read back, so headings
  are found by size only and code isn't recognized as code.
- Token counts are exact for Apple's model when Apple Intelligence is available; otherwise a
  conservative estimate (characters ÷ 3). Supply a `TokenCounter` for other models.
- **Exact token counting is slow per call** (~48 ms on a Mac, whatever the length), so the engine
  remembers exact counts and pre-counts each document's sections in the background when it's added
  or loaded (~48 ms per section, once per launch). Measured on a Mac with Apple's counter: a repeated
  question 0.014 s, a new question after pre-counting 0.43 s (was ~2.2–2.8 s); a question asked
  while pre-counting is still running isn't faster (~2.5 s). Not yet measured on an iPhone.
- Adds ~64 MB (the embedding model) to your app.

## Tests

```bash
swift test
```

Tests that need the embedding model look for it at `ONDEVICE_RAG_MODEL`, defaulting to
`~/Developer/coreai-models/exports/bge_small_buckets_float16.aimodel`, and are skipped if it's
missing (`Tools/download_model.sh ~/Developer/coreai-models/exports` puts it there). `Tools/make_package_fixtures.py` regenerates the tokenizer files and the Python reference
values the tests compare against.

## License

MIT — see `LICENSE`. The embedding model and the bundled tokenizer files
(`Sources/OnDeviceRAG/Resources/bge-small-en-v1.5/`) come from
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), also MIT-licensed.
