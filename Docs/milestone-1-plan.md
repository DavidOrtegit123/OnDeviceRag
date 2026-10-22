# OnDeviceRAG — Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Project rule (AGENTS.md → Mandatory Pre-Coding Protocol):** before each task's first code edit, post a short What / Why / How / Alternatives pitch and wait for the researcher's approval. Commit only when the researcher says so. Commit messages carry **no AI attribution lines**.

**Goal:** A working, Mac-tested core of the `OnDeviceRAG` Swift package: PDF/text ingest → bge-small embeddings → in-memory hybrid index → retrieval → token-budgeted context.

**Architecture:** One SwiftPM library target `OnDeviceRAG` with five internal areas (Ingest, Embedding, Index/Search, Packing, Engine) behind small protocols (`Embedder`, `TokenCounter`, `Reranker`). The public entry point is the `RAGEngine` actor. Pure logic (section detection, chunking, BM25, fusion, packing) is tested with plain Swift values and a fake embedder; the real Core AI embedder is tested only when the model file is present locally.

**Tech Stack:** Swift 6.4 tools, Swift 6 language mode, Swift Testing, PDFKit, NaturalLanguage (`NLTokenizer` sentences only), Accelerate (`vDSP`), CoreAI (`AIModel`), FoundationModels (token counting), `swift-transformers` 1.3.4 (`Tokenizers` product, WordPiece).

**Spec:** `Packages/OnDeviceRAG/Docs/design.md`

## Global Constraints

- Platforms: **iOS 27, macOS 27**; `// swift-tools-version: 6.4` (first tools version with `.iOS(.v27)` / `.macOS(.v27)`).
- **No network calls** anywhere in the package (Architecture Rule #1). The tokenizer loads from bundled files via `AutoTokenizer.from(modelFolder:)`, never from the Hub.
- Only external dependency: `https://github.com/huggingface/swift-transformers`, `exact: "1.3.4"` (the version the app already resolves), product `Tokenizers`.
- **The app is not modified in milestone 1.** In particular the app's `KnowledgeBase/BM25Index.swift`, `HybridRetriever.swift`, `PassageChunker.swift` stay byte-identical (batch Stage 2/3 depend on them). The package gets its own copy of the BM25 logic.
- The embedding model (`bge_small_buckets_float16.aimodel`, a 64 MB directory) is **not** committed (`*.aimodel` is git-ignored). Tests that need it read its path from `ONDEVICE_RAG_MODEL`, defaulting to `~/Developer/coreai-models/exports/bge_small_buckets_float16.aimodel`, and are skipped when it's missing.
- Frozen defaults (design §8): chunk 400 characters; fallback section window 1,500 characters; embedding lengths 128/256/512 tokens; BM25 k1 1.2, b 0.75, tokens `[a-z0-9_]+` after lowercasing; RRF k 60; 20 candidates; reranker off (top 10 when on); ≤ 2 history messages; token budget 2,000.
- bge query prefix, verbatim: `Represent this sentence for searching relevant passages: ` (queries only, never passages).
- English only. PDF + plain text only.
- Run all tests with: `swift test --package-path Packages/OnDeviceRAG` (from the repo root).

## Deviations from design.md (agreed 2026-10-22, applied to design.md in Task 1)

1. BM25 is **copied** into the package, not moved (the app's file stays until milestone 2).
2. Vision OCR for scanned pages moves to **milestone 2**. Milestone 1 records those pages in `DocumentInfo.pagesWithoutText`.
3. `RAGEngine.init` takes only a configuration in milestone 1 (the index is in memory). `storageDirectory` and `addPrebuiltIndex(at:)` arrive with persistence in milestone 2.
4. A `FoundationModelsReranker` implementation belongs to milestone 3 (benchmark decides the default). Milestone 1 ships the `Reranker` protocol and the hook.
5. Added for testability and for apps with text sources: `DocumentIngestor.ingestText(_:title:id:)` (sections at `## ` lines, code inside ``` fences).
6. With `history`, the query text is **question first, then the most recent history messages**, so the embedder's 512-token truncation can only cut history, never the question.

## File Structure

```
Packages/OnDeviceRAG/
├── Package.swift
├── Sources/OnDeviceRAG/
│   ├── Model/Types.swift                 public value types (DocumentInfo, Section, Chunk, RetrievedChunk, RAGContext…)
│   ├── Ingest/TextLine.swift             TextLine, OutlineEntry (input to section detection)
│   ├── Ingest/SectionDetector.swift      lines → sections (outline / headings / windows)
│   ├── Ingest/Chunker.swift              section lines → section text + ≤400-char chunks (code kept whole)
│   ├── Ingest/PDFTextExtractor.swift     PDFKit → TextLines + outline + pages without text
│   ├── Ingest/DocumentIngestor.swift     public: ingestPDF / ingestText → IngestedDocument
│   ├── Embedding/Embedder.swift          Embedder protocol, EmbeddingKind
│   ├── Embedding/BGESmallEmbedder.swift  Core AI bge-small embedder (actor)
│   ├── Resources/bge-small-en-v1.5/      tokenizer.json, tokenizer_config.json (MIT, from BAAI)
│   ├── Search/BM25Index.swift            copy of the app's BM25 (configurable k1/b)
│   ├── Search/RankFusion.swift           Reciprocal Rank Fusion
│   ├── Search/InMemoryIndex.swift        chunks + flat vectors + BM25; dense & keyword orders
│   ├── Search/Retriever.swift            query text, dense + BM25 + RRF → RetrievedChunk
│   ├── Packing/TokenCounter.swift        TokenCounter protocol, Estimating + Apple counters
│   ├── Packing/ContextPacker.swift       candidates → RAGContext within a token budget
│   ├── Engine/RAGConfiguration.swift     every tunable + pluggable components; Reranker protocol
│   └── Engine/RAGEngine.swift            public actor tying it together
├── Tests/OnDeviceRAGTests/
│   ├── Support/FakeEmbedder.swift        deterministic bag-of-words embedder
│   ├── Support/FixturePDF.swift          builds PDFs in memory (CoreText) for ingest tests
│   ├── Support/ModelLocation.swift       model path + `isAvailable` for gated tests
│   ├── Fixtures/bge_reference.json       Python token IDs + vectors (generated, committed)
│   ├── SectionDetectorTests.swift
│   ├── ChunkerTests.swift
│   ├── PDFIngestTests.swift
│   ├── SearchTests.swift                 BM25, RRF, InMemoryIndex
│   ├── RetrieverTests.swift
│   ├── ContextPackerTests.swift
│   ├── EngineTests.swift
│   └── BGESmallEmbedderTests.swift       gated: tokenizer parity, vector parity, 40-task check
└── Tools/make_package_fixtures.py        writes the tokenizer resources + bge_reference.json
```

---

### Task 1: Package scaffold, core types, design.md deviations

**Files:**
- Create: `Packages/OnDeviceRAG/Package.swift`
- Create: `Packages/OnDeviceRAG/Sources/OnDeviceRAG/Model/Types.swift`
- Create: `Packages/OnDeviceRAG/Tests/OnDeviceRAGTests/TypesTests.swift`
- Modify: `Packages/OnDeviceRAG/Docs/design.md` (§4, §5, §6, §11 — the six deviations above)

**Interfaces:**
- Produces: `DocumentID`, `PageRange`, `SectionSource`, `DocumentInfo`, `Section`, `Chunk`, `RetrievedChunk`, `ContextSection`, `RAGContext`, `TokenCount` (exact definitions below; every later task uses them).

- [ ] **Step 1: Write `Package.swift`**

The `Resources` folder is declared in Task 6. Until then, leave the `resources:` line out.

```swift
// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "OnDeviceRAG",
    platforms: [.iOS(.v27), .macOS(.v27)],
    products: [
        .library(name: "OnDeviceRAG", targets: ["OnDeviceRAG"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.4"),
    ],
    targets: [
        .target(
            name: "OnDeviceRAG",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")]
        ),
        .testTarget(
            name: "OnDeviceRAGTests",
            dependencies: ["OnDeviceRAG"],
            resources: [.copy("Fixtures")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
```

Create an empty `Tests/OnDeviceRAGTests/Fixtures/.gitkeep` so the `.copy("Fixtures")` path exists.

- [ ] **Step 2: Write the failing test**

`Tests/OnDeviceRAGTests/TypesTests.swift`:

```swift
import Testing
@testable import OnDeviceRAG

struct TypesTests {
    @Test func pageRangeUnionAndLabel() {
        let range = PageRange(first: 3, last: 3).union(PageRange(first: 5, last: 6))
        #expect(range == PageRange(first: 3, last: 6))
        #expect(range.label == "pp. 3–6")
        #expect(PageRange(first: 2, last: 2).label == "p. 2")
    }

    @Test func chunkIndexedTextIncludesSectionTitle() {
        let chunk = Chunk(documentID: "d", sectionIndex: 0, sectionTitle: "Rules", position: 0,
                          text: "Integers are encoded as i<n>.", pages: PageRange(first: 1, last: 1), isCode: false)
        #expect(chunk.indexedText == "Rules: Integers are encoded as i<n>.")
        #expect(chunk.id == "d#0.0")
    }
}
```

- [ ] **Step 3: Run it — expect a compile failure** (`cannot find 'PageRange' in scope`)

Run: `swift test --package-path Packages/OnDeviceRAG`

- [ ] **Step 4: Write `Model/Types.swift`**

```swift
import Foundation

public typealias DocumentID = String

/// 1-based, inclusive page numbers.
public struct PageRange: Sendable, Hashable, Codable {
    public let first: Int
    public let last: Int

    public init(first: Int, last: Int) {
        self.first = min(first, last)
        self.last = max(first, last)
    }

    public func union(_ other: PageRange) -> PageRange {
        PageRange(first: min(first, other.first), last: max(last, other.last))
    }

    /// "p. 3" or "pp. 3–6".
    public var label: String {
        first == last ? "p. \(first)" : "pp. \(first)–\(last)"
    }
}

/// How a document's sections were found.
public enum SectionSource: String, Sendable, Codable {
    case outline      // the PDF's table of contents
    case headings     // large or bold short lines
    case windows      // fixed ~1,500-character windows (no structure found)
    case provided     // `## ` headings in text given by the app
}

public struct DocumentInfo: Sendable, Hashable, Codable {
    public let id: DocumentID
    public let title: String
    public let pageCount: Int
    public let sectionCount: Int
    public let chunkCount: Int
    public let sectionSource: SectionSource
    /// Pages with < 50 non-whitespace characters (probably scanned; OCR comes in milestone 2).
    public let pagesWithoutText: [Int]
    /// Pages where math symbols are > 5% of non-whitespace characters (extraction may be garbled).
    public let mathHeavyPages: [Int]
}

public struct Section: Sendable, Hashable, Codable {
    public let documentID: DocumentID
    public let index: Int
    public let title: String
    /// Full section text: prose paragraphs and code blocks separated by blank lines.
    public let text: String
    public let pages: PageRange
}

public struct Chunk: Sendable, Hashable, Codable {
    public let documentID: DocumentID
    public let sectionIndex: Int
    public let sectionTitle: String
    /// 0-based position within its section.
    public let position: Int
    public let text: String
    public let pages: PageRange
    public let isCode: Bool

    public var id: String { "\(documentID)#\(sectionIndex).\(position)" }

    /// What gets embedded and keyword-searched: the heading gives a short chunk context.
    public var indexedText: String { "\(sectionTitle): \(text)" }
}

public struct RetrievedChunk: Sendable, Hashable {
    public let chunk: Chunk
    public let documentTitle: String
    public let fusedScore: Double
    /// 1-based rank in the dense (embedding) ranking.
    public let denseRank: Int?
    /// 1-based rank in the BM25 ranking; nil if the chunk shares no word with the query.
    public let keywordRank: Int?
}

public struct ContextSection: Sendable, Hashable {
    public let documentID: DocumentID
    public let documentTitle: String
    public let sectionIndex: Int
    public let title: String
    public let pages: PageRange
    /// False when only a window of chunks around the match fit the budget.
    public let isWholeSection: Bool
}

public struct RAGContext: Sendable, Hashable {
    /// Ready to insert under "Reference material:".
    public let text: String
    public let sections: [ContextSection]
    public let tokenCount: Int
    public let tokenCountIsEstimate: Bool
}

public struct TokenCount: Sendable, Hashable {
    public let value: Int
    public let isEstimate: Bool

    public init(value: Int, isEstimate: Bool) {
        self.value = value
        self.isEstimate = isEstimate
    }
}
```

- [ ] **Step 5: Run tests — expect PASS** (the first run resolves `swift-transformers`, which takes a minute)

Run: `swift test --package-path Packages/OnDeviceRAG`

- [ ] **Step 6: Apply the deviations to `design.md`**

- §4: replace `public init(storageDirectory: URL, configuration: RAGConfiguration = .default) async throws` with `public init(configuration: RAGConfiguration)` plus the note "(milestone 1: in-memory; `storageDirectory` and `addPrebuiltIndex(at:)` arrive with persistence in milestone 2)". Add `addDocument(_ document: IngestedDocument)` and mention `DocumentIngestor.ingestText`.
- §5 step 1: "… run Vision `RecognizeDocumentsRequest` **(milestone 2; milestone 1 lists these pages in `DocumentInfo.pagesWithoutText`)**".
- §6 Retriever: "(the research app's `BM25Index` logic, **copied** into the package — the app's file stays unchanged because batch Stages 2/3 depend on it)"; history: "query text = the question, then the last ≤ 2 history messages (most recent first), so 512-token truncation only cuts history".
- §11 milestone 1: "… Context packer, `Reranker` protocol (implementation in milestone 3) + unit tests (Mac). OCR → milestone 2."
- Status line: `_Status: reviewed and approved 2026-10-22 (with the milestone-1 deviations listed in Docs/milestone-1-plan.md)._`

- [ ] **Step 7: Commit (when the researcher approves)**

```bash
git add Packages/OnDeviceRAG/Package.swift Packages/OnDeviceRAG/Sources Packages/OnDeviceRAG/Tests Packages/OnDeviceRAG/Docs
git commit -m "OnDeviceRAG: package scaffold and core types"
```

Check that `Packages/OnDeviceRAG/.build/` is not staged (add `Packages/OnDeviceRAG/.build/` to `.gitignore` if `git status` shows it).

---

### Task 2: BM25 and Reciprocal Rank Fusion

**Files:**
- Create: `Sources/OnDeviceRAG/Search/BM25Index.swift`
- Create: `Sources/OnDeviceRAG/Search/RankFusion.swift`
- Test: `Tests/OnDeviceRAGTests/SearchTests.swift`

(All paths in this and later tasks are relative to `Packages/OnDeviceRAG/`.)

**Interfaces:**
- Produces: `struct BM25Index: Sendable { init(documents: [String], k1: Double, b: Double); func scores(for query: String) -> [Double]; static func tokenize(_ text: String) -> [String] }`
- Produces: `enum RankFusion { static func fuse(_ orders: [[Int]], count: Int, constant: Double) -> [(index: Int, score: Double)] }`

- [ ] **Step 1: Write the failing tests**

```swift
import Testing
@testable import OnDeviceRAG

struct SearchTests {
    @Test func tokenizerKeepsUnderscoresAndLowercases() {
        #expect(BM25Index.tokenize("Call encode_ZORP(x), then v2!") == ["call", "encode_zorp", "x", "then", "v2"])
    }

    @Test func bm25RanksExactRareTermFirst() {
        let index = BM25Index(documents: [
            "the cache stores values",
            "zorplib encodes integers",
            "the the the cache",
        ], k1: 1.2, b: 0.75)
        let scores = index.scores(for: "zorplib integers")
        #expect(scores[1] > scores[0])
        #expect(scores[0] == 0)
        #expect(scores[2] == 0)
    }

    @Test func bm25MatchesHandComputedValue() {
        // Two docs; query "a". Doc 0 = "a b" (len 2), doc 1 = "b b" (len 2), avg 2.
        // idf(a) = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln(2)
        // tf part = 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 1)) = 2.2 / 2.2 = 1
        let index = BM25Index(documents: ["a b", "b b"], k1: 1.2, b: 0.75)
        let scores = index.scores(for: "a")
        #expect(abs(scores[0] - log(2.0)) < 1e-12)
        #expect(scores[1] == 0)
    }

    @Test func rrfCombinesRanksAndBreaksTiesByIndex() {
        // Item 2 is 1st in A and 2nd in B; item 0 is 2nd in A only; item 1 is 1st in B only.
        let fused = RankFusion.fuse([[2, 0], [1, 2]], count: 4, constant: 60)
        #expect(fused.map(\.index) == [2, 1, 0, 3])
        #expect(abs(fused[0].score - (1.0 / 61 + 1.0 / 62)) < 1e-12)
        #expect(fused[3].score == 0)
    }
}
```

- [ ] **Step 2: Run — expect compile failure** (`cannot find 'BM25Index'`)

- [ ] **Step 3: Write `Search/BM25Index.swift`**

Copied from the app's `KnowledgeBase/BM25Index.swift` (same formula and tokenizer); only `k1`/`b` became init parameters.

```swift
import Foundation

/// Classic BM25 keyword ranking (Lucene-style IDF). Copied from the research app's
/// `BM25Index` (same formula and tokenizer); k1 and b are parameters here.
struct BM25Index: Sendable {
    let k1: Double
    let b: Double

    private let termCounts: [[String: Int]]
    private let lengths: [Int]
    private let averageLength: Double
    private let documentFrequency: [String: Int]

    init(documents: [String], k1: Double, b: Double) {
        self.k1 = k1
        self.b = b
        let tokenized = documents.map { Self.tokenize($0) }
        termCounts = tokenized.map { tokens in
            tokens.reduce(into: [String: Int]()) { counts, token in counts[token, default: 0] += 1 }
        }
        lengths = tokenized.map(\.count)
        averageLength = lengths.isEmpty ? 0 : Double(lengths.reduce(0, +)) / Double(lengths.count)

        var frequency: [String: Int] = [:]
        for counts in termCounts {
            for term in counts.keys {
                frequency[term, default: 0] += 1
            }
        }
        documentFrequency = frequency
    }

    /// ln(1 + (N − n + 0.5) / (n + 0.5)), always positive.
    func inverseDocumentFrequency(of term: String) -> Double {
        let containing = Double(documentFrequency[term] ?? 0)
        let total = Double(termCounts.count)
        return log(1 + (total - containing + 0.5) / (containing + 0.5))
    }

    /// One score per document, in input order. Repeated query tokens count each time.
    func scores(for query: String) -> [Double] {
        let queryTokens = Self.tokenize(query)
        return termCounts.indices.map { index in
            let relativeLength = averageLength > 0 ? Double(lengths[index]) / averageLength : 0
            return queryTokens.reduce(0.0) { total, term in
                guard let count = termCounts[index][term] else { return total }
                let frequency = Double(count)
                let saturation = frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * relativeLength))
                return total + inverseDocumentFrequency(of: term) * saturation
            }
        }
    }

    /// Lowercases, then keeps maximal runs of a–z, 0–9 and `_` (so `encode_zorp` stays one token).
    static func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        var current = String.UnicodeScalarView()
        for scalar in text.lowercased().unicodeScalars {
            let value = scalar.value
            let isTokenScalar = (97...122).contains(value) || (48...57).contains(value) || value == 95
            if isTokenScalar {
                current.append(scalar)
            } else if !current.isEmpty {
                tokens.append(String(current))
                current = String.UnicodeScalarView()
            }
        }
        if !current.isEmpty {
            tokens.append(String(current))
        }
        return tokens
    }
}
```

- [ ] **Step 4: Write `Search/RankFusion.swift`**

```swift
/// Reciprocal Rank Fusion: score = Σ over rankings of 1 / (constant + rank), ranks 1-based.
/// Items in no ranking score 0. Ties go to the lower index.
enum RankFusion {
    static func fuse(_ orders: [[Int]], count: Int, constant: Double) -> [(index: Int, score: Double)] {
        var scores = [Double](repeating: 0, count: count)
        for order in orders {
            for (position, index) in order.enumerated() {
                scores[index] += 1 / (constant + Double(position + 1))
            }
        }
        return scores.indices
            .map { (index: $0, score: scores[$0]) }
            .sorted { $0.score != $1.score ? $0.score > $1.score : $0.index < $1.index }
    }
}
```

- [ ] **Step 5: Run — expect PASS.** Also confirm the app's BM25 file is untouched: `git diff --exit-code "Prototype On-Device/KnowledgeBase/BM25Index.swift"` (exit 0).

- [ ] **Step 6: Commit (when approved)** — `git commit -m "OnDeviceRAG: BM25 and reciprocal rank fusion"`

---

### Task 3: Chunker (section lines → section text + chunks)

**Files:**
- Create: `Sources/OnDeviceRAG/Ingest/TextLine.swift`
- Create: `Sources/OnDeviceRAG/Ingest/Chunker.swift`
- Test: `Tests/OnDeviceRAGTests/ChunkerTests.swift`

**Interfaces:**
- Consumes: `Chunk`, `Section`, `PageRange` (Task 1).
- Produces: `struct TextLine: Sendable, Hashable { var text: String; var page: Int; var fontSize: Double; var isBold: Bool; var isMonospaced: Bool; init(_ text: String, page: Int = 1, fontSize: Double = 11, isBold: Bool = false, isMonospaced: Bool = false) }`
- Produces: `struct OutlineEntry: Sendable, Hashable { var title: String; var page: Int }`
- Produces: `struct DetectedSection: Sendable, Hashable { var title: String; var lines: [TextLine] }`
- Produces: `enum Chunker { static func build(_ section: DetectedSection, documentID: DocumentID, sectionIndex: Int, maximumLength: Int) -> (section: Section, chunks: [Chunk])? }` — nil if the section has no text.

Rules (from design §5): prose is joined line by line (a trailing `-` followed by a lowercase letter is de-hyphenated; otherwise lines are joined with a space; an empty line is a paragraph break); prose is split into sentences with `NLTokenizer(unit: .sentence)` and packed greedily into chunks of ≤ `maximumLength` characters (a sentence longer than the limit is cut at the last space before the limit); runs of consecutive monospaced lines form **one code chunk** joined with `\n`, indentation kept, never split. Prose and code never share a chunk. A chunk's pages are the pages of the lines it came from.

- [ ] **Step 1: Write the failing tests**

```swift
import Testing
@testable import OnDeviceRAG

struct ChunkerTests {
    private func build(_ lines: [TextLine], limit: Int = 400) -> (section: Section, chunks: [Chunk]) {
        Chunker.build(DetectedSection(title: "S", lines: lines), documentID: "d", sectionIndex: 0, maximumLength: limit)!
    }

    @Test func joinsWrappedLinesAndDehyphenates() {
        let result = build([TextLine("The encoder writes inte-"), TextLine("gers as text.")])
        #expect(result.chunks.map(\.text) == ["The encoder writes integers as text."])
    }

    @Test func packsWholeSentencesUpToTheLimit() {
        let sentence = String(repeating: "word ", count: 15).trimmingCharacters(in: .whitespaces) + "." // 74 chars
        let text = Array(repeating: sentence, count: 8).joined(separator: " ")
        let result = build([TextLine(text)], limit: 200)
        #expect(result.chunks.allSatisfy { $0.text.count <= 200 })
        #expect(result.chunks.allSatisfy { $0.text.hasSuffix(".") })
        #expect(result.chunks.count == 4) // 2 sentences (75 + 1 + 75 = 151 chars) per chunk
    }

    @Test func cutsAnOverlongSentenceAtASpace() {
        let long = Array(repeating: "abcdefghi", count: 60).joined(separator: " ") // 599 chars, no period
        let result = build([TextLine(long)])
        #expect(result.chunks.count == 2)
        #expect(result.chunks.allSatisfy { $0.text.count <= 400 && !$0.text.hasPrefix(" ") })
    }

    @Test func keepsCodeBlockWholeWithIndentation() {
        let code = (0..<30).map { TextLine("    value_\($0) = compute(\($0))", isMonospaced: true) }
        let result = build([TextLine("Example:")] + code + [TextLine("That is all.")])
        let codeChunks = result.chunks.filter(\.isCode)
        #expect(codeChunks.count == 1)
        #expect(codeChunks[0].text.count > 400)
        #expect(codeChunks[0].text.hasPrefix("    value_0 = compute(0)\n    value_1"))
        #expect(result.chunks.map(\.isCode) == [false, true, false])
        #expect(result.chunks.map(\.position) == [0, 1, 2])
    }

    @Test func tracksPagesPerChunkAndSection() {
        let result = build([TextLine("First page sentence.", page: 2), TextLine("Second page sentence.", page: 3)], limit: 25)
        #expect(result.chunks.map(\.pages) == [PageRange(first: 2, last: 2), PageRange(first: 3, last: 3)])
        #expect(result.section.pages == PageRange(first: 2, last: 3))
    }

    @Test func sectionTextSeparatesParagraphsAndCode() {
        let result = build([
            TextLine("Para one."), TextLine(""), TextLine("Para two."),
            TextLine("x = 1", isMonospaced: true),
        ])
        #expect(result.section.text == "Para one.\n\nPara two.\n\nx = 1")
    }

    @Test func emptySectionReturnsNil() {
        #expect(Chunker.build(DetectedSection(title: "S", lines: [TextLine("  ")]), documentID: "d", sectionIndex: 0, maximumLength: 400) == nil)
    }
}
```

- [ ] **Step 2: Run — expect compile failure.**

- [ ] **Step 3: Write `Ingest/TextLine.swift`**

```swift
/// One line of extracted text with the font facts section detection needs.
struct TextLine: Sendable, Hashable {
    var text: String
    /// 1-based page number.
    var page: Int
    var fontSize: Double
    var isBold: Bool
    var isMonospaced: Bool

    init(_ text: String, page: Int = 1, fontSize: Double = 11, isBold: Bool = false, isMonospaced: Bool = false) {
        self.text = text
        self.page = page
        self.fontSize = fontSize
        self.isBold = isBold
        self.isMonospaced = isMonospaced
    }

    var isBlank: Bool { text.allSatisfy(\.isWhitespace) }
}

/// One table-of-contents entry from a PDF outline.
struct OutlineEntry: Sendable, Hashable {
    var title: String
    /// 1-based page number.
    var page: Int
}

/// A titled run of lines, before chunking.
struct DetectedSection: Sendable, Hashable {
    var title: String
    var lines: [TextLine]
}
```

- [ ] **Step 4: Write `Ingest/Chunker.swift`**

```swift
import Foundation
import NaturalLanguage

/// Turns a detected section into its full text and ≤ `maximumLength`-character chunks.
/// Prose is packed from whole sentences; monospaced runs become one code chunk each.
enum Chunker {
    /// A prose run (with each character's page) or a code block.
    private enum Block {
        case prose(text: String, pages: [Int])   // pages[i] = page of character i
        case code(text: String, pages: PageRange)
    }

    static func build(_ section: DetectedSection, documentID: DocumentID, sectionIndex: Int, maximumLength: Int) -> (section: Section, chunks: [Chunk])? {
        let blocks = makeBlocks(section.lines)
        var pieces: [(text: String, pages: PageRange, isCode: Bool)] = []
        var sectionParts: [String] = []

        for block in blocks {
            switch block {
            case let .code(text, pages):
                pieces.append((text, pages, true))
                sectionParts.append(text)
            case let .prose(text, pages):
                sectionParts.append(text)
                pieces += packSentences(text: text, pages: pages, maximumLength: maximumLength).map { ($0.text, $0.pages, false) }
            }
        }
        guard !pieces.isEmpty else { return nil }

        let chunks = pieces.enumerated().map { position, piece in
            Chunk(documentID: documentID, sectionIndex: sectionIndex, sectionTitle: section.title,
                  position: position, text: piece.text, pages: piece.pages, isCode: piece.isCode)
        }
        let pages = chunks.dropFirst().reduce(chunks[0].pages) { $0.union($1.pages) }
        let sectionValue = Section(documentID: documentID, index: sectionIndex, title: section.title,
                                   text: sectionParts.joined(separator: "\n\n"), pages: pages)
        return (sectionValue, chunks)
    }

    // MARK: - Blocks

    private static func makeBlocks(_ lines: [TextLine]) -> [Block] {
        var blocks: [Block] = []
        var prose = ""
        var prosePages: [Int] = []
        var code: [TextLine] = []

        func closeProse() {
            // Normalize: trim, and collapse 3+ newlines left by repeated blank lines.
            let trimmed = prose.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                let start = prose.distance(from: prose.startIndex, to: prose.firstIndex { !$0.isWhitespace }!)
                blocks.append(.prose(text: trimmed, pages: Array(prosePages[start..<(start + trimmed.count)])))
            }
            prose = ""
            prosePages = []
        }
        func closeCode() {
            guard !code.isEmpty else { return }
            let text = code.map(\.text).joined(separator: "\n")
            let pages = PageRange(first: code.map(\.page).min()!, last: code.map(\.page).max()!)
            blocks.append(.code(text: text, pages: pages))
            code = []
        }
        func appendProse(_ piece: String, page: Int) {
            prose += piece
            prosePages += Array(repeating: page, count: piece.count)
        }

        for line in lines {
            if line.isMonospaced && !line.isBlank {
                closeProse()
                code.append(line)
                continue
            }
            closeCode()
            if line.isBlank {
                if !prose.isEmpty && !prose.hasSuffix("\n\n") { appendProse("\n\n", page: prosePages.last ?? line.page) }
                continue
            }
            let text = line.text.trimmingCharacters(in: .whitespaces)
            if prose.isEmpty || prose.hasSuffix("\n\n") {
                appendProse(text, page: line.page)
            } else if prose.hasSuffix("-"), text.first?.isLowercase == true {
                prose.removeLast()
                prosePages.removeLast()
                appendProse(text, page: line.page)
            } else {
                appendProse(" " + text, page: line.page)
            }
        }
        closeProse()
        closeCode()
        return blocks
    }

    // MARK: - Sentence packing

    private static func packSentences(text: String, pages: [Int], maximumLength: Int) -> [(text: String, pages: PageRange)] {
        var sentences: [(text: String, pages: PageRange)] = []
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let start = text.distance(from: text.startIndex, to: range.lowerBound)
            let raw = String(text[range])
            let leading = raw.prefix { $0.isWhitespace }.count
            let sentence = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !sentence.isEmpty else { return true }
            for piece in cutAtSpaces(sentence, maximumLength: maximumLength, offset: start + leading) {
                let slice = pages[piece.offset..<(piece.offset + piece.text.count)]
                sentences.append((piece.text, PageRange(first: slice.min()!, last: slice.max()!)))
            }
            return true
        }

        var chunks: [(text: String, pages: PageRange)] = []
        for sentence in sentences {
            if let last = chunks.last, last.text.count + 1 + sentence.text.count <= maximumLength {
                chunks[chunks.count - 1] = (last.text + " " + sentence.text, last.pages.union(sentence.pages))
            } else {
                chunks.append(sentence)
            }
        }
        return chunks
    }

    /// Splits text longer than the limit at the last space before it. `offset` is the
    /// text's character offset in the prose run (for page lookup).
    private static func cutAtSpaces(_ text: String, maximumLength: Int, offset: Int) -> [(text: String, offset: Int)] {
        var result: [(text: String, offset: Int)] = []
        var remaining = Substring(text)
        var position = offset
        while remaining.count > maximumLength {
            let window = remaining.prefix(maximumLength + 1)
            let cut = window.lastIndex(of: " ") ?? remaining.index(remaining.startIndex, offsetBy: maximumLength)
            let head = remaining[..<cut]
            result.append((String(head), position))
            var rest = remaining[cut...]
            let skipped = rest.prefix { $0 == " " }.count
            rest = rest.dropFirst(skipped)
            position += head.count + skipped
            remaining = rest
        }
        if !remaining.isEmpty { result.append((String(remaining), position)) }
        return result
    }
}
```

Note: `\n\n` paragraph breaks inside a prose run are kept in the section text (so `sectionTextSeparatesParagraphsAndCode` passes) and treated as whitespace between sentences by `NLTokenizer`.

- [ ] **Step 5: Run — expect PASS.** If `packsWholeSentencesUpToTheLimit` fails because `NLTokenizer` treats the repeated "word word …" text unusually, print `result.chunks.map(\.text)` and fix the test data (use varied real sentences), not the 400-character rule.

- [ ] **Step 6: Commit (when approved)** — `git commit -m "OnDeviceRAG: sentence chunker that keeps code blocks whole"`

---

### Task 4: Section detection (outline → headings → windows)

**Files:**
- Create: `Sources/OnDeviceRAG/Ingest/SectionDetector.swift`
- Test: `Tests/OnDeviceRAGTests/SectionDetectorTests.swift`

**Interfaces:**
- Consumes: `TextLine`, `OutlineEntry`, `DetectedSection` (Task 3), `SectionSource` (Task 1).
- Produces: `enum SectionDetector { static func detect(lines: [TextLine], outline: [OutlineEntry], documentTitle: String, windowLength: Int) -> (sections: [DetectedSection], source: SectionSource) }`

Rules (design §5):
1. **Outline** if it has ≥ 1 entry. Entries are used in order. Each starts at the first line (at or after the previous start) on the entry's page whose normalized text equals the normalized title (that heading line is then dropped from the body); if no such line, at the first line on or after that page. Normalization = lowercased, whitespace collapsed. Lines before the first start form a section titled `documentTitle` (only if non-blank).
2. Else **headings**: body size = character-weighted median font size of non-blank, non-monospaced lines. A line is a heading if it's non-monospaced, trimmed length 1…99, doesn't end with `.`, and (`fontSize ≥ 1.2 × body size` **or** `isBold` while the body size lines are not bold). Consecutive heading lines merge into one title (joined with a space). Used only if ≥ 2 headings are found.
3. Else **windows**: consecutive lines grouped until the group reaches `windowLength` characters (lines are never split); titled `"<documentTitle> (part N)"`, N from 1.

- [ ] **Step 1: Write the failing tests**

```swift
import Testing
@testable import OnDeviceRAG

struct SectionDetectorTests {
    @Test func outlineSplitsAtMatchingHeadingLines() {
        let lines = [
            TextLine("Preface text.", page: 1),
            TextLine("1 Integers", page: 2, fontSize: 16), TextLine("Ints are i<n>.", page: 2),
            TextLine("2  Strings", page: 3, fontSize: 16), TextLine("Strings are s<n>.", page: 3),
        ]
        let outline = [OutlineEntry(title: "1 Integers", page: 2), OutlineEntry(title: "2 Strings", page: 3)]
        let result = SectionDetector.detect(lines: lines, outline: outline, documentTitle: "Doc", windowLength: 1500)
        #expect(result.source == .outline)
        #expect(result.sections.map(\.title) == ["Doc", "1 Integers", "2 Strings"])
        #expect(result.sections[1].lines.map(\.text) == ["Ints are i<n>."])
        #expect(result.sections[2].lines.map(\.text) == ["Strings are s<n>."])
    }

    @Test func outlineEntryWithoutMatchingLineStartsAtItsPage() {
        let lines = [TextLine("A.", page: 1), TextLine("B.", page: 2)]
        let result = SectionDetector.detect(lines: lines, outline: [OutlineEntry(title: "Second", page: 2)],
                                            documentTitle: "Doc", windowLength: 1500)
        #expect(result.sections.map(\.title) == ["Doc", "Second"])
        #expect(result.sections[1].lines.map(\.text) == ["B."])
    }

    @Test func largeFontLinesBecomeHeadingsAndMultiLineTitlesMerge() {
        let body = "This is ordinary body text that is long enough to dominate."
        let lines = [
            TextLine("Chapter One", fontSize: 18), TextLine("The Start", fontSize: 18), TextLine(body),
            TextLine("Chapter Two", fontSize: 18), TextLine(body),
        ]
        let result = SectionDetector.detect(lines: lines, outline: [], documentTitle: "Doc", windowLength: 1500)
        #expect(result.source == .headings)
        #expect(result.sections.map(\.title) == ["Chapter One The Start", "Chapter Two"])
    }

    @Test func boldShortLinesAreHeadingsButBoldSentencesAreNot() {
        let body = "Plain body text that is clearly the most common style here."
        let lines = [
            TextLine("Setup", isBold: true), TextLine(body),
            TextLine("Remember to save.", isBold: true), TextLine(body),
            TextLine("Usage", isBold: true), TextLine(body),
        ]
        let result = SectionDetector.detect(lines: lines, outline: [], documentTitle: "Doc", windowLength: 1500)
        #expect(result.sections.map(\.title) == ["Setup", "Usage"])
        #expect(result.sections[0].lines.map(\.text).contains("Remember to save."))
    }

    @Test func fallsBackToWindowsWithoutStructure() {
        let lines = (0..<10).map { TextLine(String(repeating: "x", count: 99) + "\($0)") } // 100 chars each
        let result = SectionDetector.detect(lines: lines, outline: [], documentTitle: "Doc", windowLength: 300)
        #expect(result.source == .windows)
        #expect(result.sections.map(\.title) == ["Doc (part 1)", "Doc (part 2)", "Doc (part 3)", "Doc (part 4)"])
        #expect(result.sections.map(\.lines.count) == [3, 3, 3, 1])
    }
}
```

- [ ] **Step 2: Run — expect compile failure.**

- [ ] **Step 3: Write `Ingest/SectionDetector.swift`**

```swift
import Foundation

/// Finds a document's sections: the PDF outline if there is one, else headings detected
/// from font size or bold short lines, else fixed-size windows.
enum SectionDetector {
    static func detect(lines: [TextLine], outline: [OutlineEntry], documentTitle: String, windowLength: Int) -> (sections: [DetectedSection], source: SectionSource) {
        if !outline.isEmpty {
            return (fromOutline(lines, outline, documentTitle), .outline)
        }
        if let sections = fromHeadings(lines) {
            return (sections, .headings)
        }
        return (windows(lines, documentTitle, windowLength), .windows)
    }

    static func normalize(_ text: String) -> String {
        text.lowercased().split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    // MARK: - Outline

    private static func fromOutline(_ lines: [TextLine], _ outline: [OutlineEntry], _ documentTitle: String) -> [DetectedSection] {
        // (start line index, title, whether the start line is the heading itself)
        var starts: [(index: Int, title: String, skipsHeading: Bool)] = []
        var searchFrom = 0
        for entry in outline {
            let target = normalize(entry.title)
            if let match = lines.indices.first(where: { $0 >= searchFrom && lines[$0].page == entry.page && normalize(lines[$0].text) == target }) {
                starts.append((match, entry.title, true))
                searchFrom = match + 1
            } else {
                let index = lines.indices.first(where: { $0 >= searchFrom && lines[$0].page >= entry.page }) ?? lines.count
                starts.append((index, entry.title, false))
                searchFrom = index
            }
        }

        var sections: [DetectedSection] = []
        let firstStart = starts.first?.index ?? lines.count
        let preamble = Array(lines[..<firstStart])
        if preamble.contains(where: { !$0.isBlank }) {
            sections.append(DetectedSection(title: documentTitle, lines: preamble))
        }
        for (position, start) in starts.enumerated() {
            let end = position + 1 < starts.count ? starts[position + 1].index : lines.count
            let bodyStart = min(start.skipsHeading ? start.index + 1 : start.index, end)
            sections.append(DetectedSection(title: start.title, lines: Array(lines[bodyStart..<end])))
        }
        return sections
    }

    // MARK: - Headings

    private static func fromHeadings(_ lines: [TextLine]) -> [DetectedSection]? {
        let body = lines.filter { !$0.isBlank && !$0.isMonospaced }
        guard let bodySize = weightedMedianFontSize(body) else { return nil }
        let bodyIsBold = body.filter { $0.fontSize == bodySize }.allSatisfy(\.isBold)

        func isHeading(_ line: TextLine) -> Bool {
            let text = line.text.trimmingCharacters(in: .whitespaces)
            guard !line.isMonospaced, (1..<100).contains(text.count), !text.hasSuffix(".") else { return false }
            return line.fontSize >= 1.2 * bodySize || (line.isBold && !bodyIsBold)
        }

        var sections: [DetectedSection] = []
        var preamble: [TextLine] = []
        var previousWasHeading = false
        for line in lines {
            if isHeading(line) {
                let title = line.text.trimmingCharacters(in: .whitespaces)
                if previousWasHeading, let last = sections.indices.last {
                    sections[last].title += " " + title
                } else {
                    sections.append(DetectedSection(title: title, lines: []))
                }
                previousWasHeading = true
            } else {
                if line.isBlank == false { previousWasHeading = false }
                if sections.isEmpty { preamble.append(line) } else { sections[sections.count - 1].lines.append(line) }
            }
        }
        guard sections.count >= 2 else { return nil }
        if preamble.contains(where: { !$0.isBlank }) {
            sections[0].lines = preamble + sections[0].lines
        }
        return sections
    }

    /// Median font size where each line counts once per character.
    private static func weightedMedianFontSize(_ lines: [TextLine]) -> Double? {
        let sizes = lines.map { (size: $0.fontSize, weight: $0.text.count) }.sorted { $0.size < $1.size }
        let total = sizes.reduce(0) { $0 + $1.weight }
        guard total > 0 else { return nil }
        var running = 0
        for entry in sizes {
            running += entry.weight
            if running * 2 >= total { return entry.size }
        }
        return sizes.last?.size
    }

    // MARK: - Windows

    private static func windows(_ lines: [TextLine], _ documentTitle: String, _ windowLength: Int) -> [DetectedSection] {
        var sections: [DetectedSection] = []
        var current: [TextLine] = []
        var length = 0
        for line in lines {
            current.append(line)
            length += line.text.count
            if length >= windowLength {
                sections.append(DetectedSection(title: "\(documentTitle) (part \(sections.count + 1))", lines: current))
                current = []
                length = 0
            }
        }
        if current.contains(where: { !$0.isBlank }) {
            sections.append(DetectedSection(title: "\(documentTitle) (part \(sections.count + 1))", lines: current))
        }
        return sections
    }
}
```

Note on the preamble in heading mode: text before the first heading is prepended to the first section (not a separate untitled section), because heading-detected documents usually start with a title line that is itself detected as the first heading.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit (when approved)** — `git commit -m "OnDeviceRAG: section detection from outline, headings or windows"`

---

### Task 5: PDF extraction and the public ingestor

**Files:**
- Create: `Sources/OnDeviceRAG/Ingest/PDFTextExtractor.swift`
- Create: `Sources/OnDeviceRAG/Ingest/DocumentIngestor.swift`
- Create: `Tests/OnDeviceRAGTests/Support/FixturePDF.swift`
- Test: `Tests/OnDeviceRAGTests/PDFIngestTests.swift`

**Interfaces:**
- Consumes: `SectionDetector.detect`, `Chunker.build`, `TextLine`, `OutlineEntry`, all Task 1 types.
- Produces: `public struct IngestedDocument: Sendable { public let info: DocumentInfo; public let sections: [Section]; public let chunks: [Chunk] }`
- Produces: `public struct IngestSettings: Sendable, Hashable { public var chunkLength = 400; public var sectionWindowLength = 1500; public init() }`
- Produces: `public enum DocumentIngestor { public static func ingestPDF(at url: URL, title: String? = nil, id: DocumentID = UUID().uuidString, settings: IngestSettings = IngestSettings()) throws -> IngestedDocument; public static func ingestText(_ text: String, title: String, id: DocumentID = UUID().uuidString, settings: IngestSettings = IngestSettings()) -> IngestedDocument }`
- Produces (internal, for tests): `static func ingest(pdf document: PDFDocument, title: String, id: DocumentID, settings: IngestSettings) -> IngestedDocument`
- Produces: `public enum IngestError: Error, Equatable { case unreadablePDF(URL) }`

- [ ] **Step 1: Write the fixture-PDF helper (test support, macOS only)**

`Tests/OnDeviceRAGTests/Support/FixturePDF.swift`:

```swift
#if os(macOS)
import AppKit
import CoreText
import PDFKit

/// Builds small PDFs in memory with chosen fonts, one CoreText line per text line.
enum FixturePDF {
    struct Line {
        var text: String
        var font: NSFont
        static func body(_ text: String) -> Line { Line(text: text, font: .systemFont(ofSize: 11)) }
        static func heading(_ text: String) -> Line { Line(text: text, font: .boldSystemFont(ofSize: 18)) }
        static func code(_ text: String) -> Line { Line(text: text, font: .monospacedSystemFont(ofSize: 10, weight: .regular)) }
    }

    static func make(pages: [[Line]], outline: [(title: String, page: Int)] = []) -> PDFDocument {
        let data = NSMutableData()
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let context = CGContext(consumer: CGDataConsumer(data: data)!, mediaBox: &box, nil)!
        for page in pages {
            context.beginPDFPage(nil)
            var y = 740.0
            for line in page {
                let attributed = NSAttributedString(string: line.text, attributes: [.font: line.font])
                context.textPosition = CGPoint(x: 72, y: y)
                CTLineDraw(CTLineCreateWithAttributedString(attributed), context)
                y -= line.font.pointSize * 1.8
            }
            context.endPDFPage()
        }
        context.closePDF()
        let document = PDFDocument(data: data as Data)!
        if !outline.isEmpty {
            let root = PDFOutline()
            for (position, entry) in outline.enumerated() {
                let item = PDFOutline()
                item.label = entry.title
                item.destination = PDFDestination(page: document.page(at: entry.page - 1)!, at: CGPoint(x: 0, y: 792))
                root.insertChild(item, at: position)
            }
            document.outlineRoot = root
        }
        return document
    }
}
#endif
```

- [ ] **Step 2: Write the failing tests**

```swift
#if os(macOS)
import PDFKit
import Testing
@testable import OnDeviceRAG

struct PDFIngestTests {
    @Test func extractsLinesWithFontFacts() {
        let pdf = FixturePDF.make(pages: [[.heading("Integers"), .body("Integers are written as i<n>."), .code("    encode(5)")]])
        let result = PDFTextExtractor.lines(from: pdf)
        let texts = result.lines.map { $0.text.trimmingCharacters(in: .whitespaces) }
        #expect(texts == ["Integers", "Integers are written as i<n>.", "encode(5)"])
        #expect(result.lines[0].fontSize > result.lines[1].fontSize)
        #expect(result.lines[0].isBold)
        #expect(result.lines[2].isMonospaced)
        #expect(!result.lines[1].isMonospaced)
    }

    @Test func headingsBecomeSectionsAndCodeStaysOneChunk() {
        let pdf = FixturePDF.make(pages: [
            [.heading("Integers"), .body("Integers are written as i<n>."), .body("Negative numbers keep their sign.")],
            [.heading("Lists"), .body("Lists are written as l<items>e."), .code("encode([1, 2])"), .code("# -> li1ei2ee")],
        ])
        let document = DocumentIngestor.ingest(pdf: pdf, title: "ZORP", id: "z", settings: IngestSettings())
        #expect(document.info.sectionSource == .headings)
        #expect(document.sections.map(\.title) == ["Integers", "Lists"])
        #expect(document.sections[1].pages == PageRange(first: 2, last: 2))
        let code = document.chunks.filter(\.isCode)
        #expect(code.count == 1)
        #expect(code[0].text.contains("encode([1, 2])\n"))
        #expect(document.info.pageCount == 2)
        #expect(document.info.chunkCount == document.chunks.count)
    }

    @Test func outlineWinsOverHeadings() {
        let pdf = FixturePDF.make(pages: [[.heading("Alpha"), .body("First body sentence here.")],
                                          [.heading("Beta"), .body("Second body sentence here.")]],
                                  outline: [(title: "Alpha", page: 1), (title: "Beta", page: 2)])
        let document = DocumentIngestor.ingest(pdf: pdf, title: "Doc", id: "d", settings: IngestSettings())
        #expect(document.info.sectionSource == .outline)
        #expect(document.sections.map(\.title) == ["Alpha", "Beta"])
    }

    @Test func emptyPageIsReportedAsWithoutText() {
        // Page 1 needs ≥ 50 non-whitespace characters to count as a text page.
        let pdf = FixturePDF.make(pages: [[.body("Some real text on the first page, clearly long enough to count as text.")], []])
        let document = DocumentIngestor.ingest(pdf: pdf, title: "Doc", id: "d", settings: IngestSettings())
        #expect(document.info.pagesWithoutText == [2])
    }

    @Test func mathHeavyPageIsFlagged() {
        let pdf = FixturePDF.make(pages: [[.body("x² + 2x − 3 = 0 ⇒ x = −3 ∨ x = 1, √9 = 3, ∑ k ≤ n")],
                                          [.body("A page of plain words without any formulas at all here.")]])
        let document = DocumentIngestor.ingest(pdf: pdf, title: "Doc", id: "d", settings: IngestSettings())
        #expect(document.info.mathHeavyPages == [1])
    }

    @Test func textIngestUsesHashHeadingsAndFences() {
        let text = """
        Intro line.

        ## Integers
        Integers are i<n>.

        ## Lists
        Lists are l<items>e.
        ```
        encode([1])
        ```
        """
        let document = DocumentIngestor.ingestText(text, title: "ZORP", id: "z")
        #expect(document.info.sectionSource == .provided)
        #expect(document.sections.map(\.title) == ["ZORP", "Integers", "Lists"])
        #expect(document.chunks.filter(\.isCode).map(\.text) == ["encode([1])"])
    }

    @Test func unreadableFileThrows() {
        let url = URL(fileURLWithPath: "/nonexistent/file.pdf")
        #expect(throws: IngestError.unreadablePDF(url)) { try DocumentIngestor.ingestPDF(at: url) }
    }
}
#endif
```

- [ ] **Step 3: Run — expect compile failure.**

- [ ] **Step 4: Write `Ingest/PDFTextExtractor.swift`**

```swift
import Foundation
import PDFKit
#if canImport(AppKit)
import AppKit
private typealias PlatformFont = NSFont
#else
import UIKit
private typealias PlatformFont = UIFont
#endif

/// Reads a PDF's text line by line with font size, bold and monospace facts, plus its outline.
enum PDFTextExtractor {
    static let minimumCharactersForTextPage = 50

    static func lines(from document: PDFDocument) -> (lines: [TextLine], pagesWithoutText: [Int]) {
        var lines: [TextLine] = []
        var empty: [Int] = []
        for pageIndex in 0..<document.pageCount {
            let pageNumber = pageIndex + 1
            guard let attributed = document.page(at: pageIndex)?.attributedString else {
                empty.append(pageNumber)
                continue
            }
            let string = attributed.string as NSString
            var pageLines: [TextLine] = []
            string.enumerateSubstrings(in: NSRange(location: 0, length: string.length), options: .byLines) { substring, range, _, _ in
                guard let substring else { return }
                pageLines.append(makeLine(substring, range: range, in: attributed, page: pageNumber))
            }
            let characters = pageLines.reduce(0) { $0 + $1.text.filter { !$0.isWhitespace }.count }
            if characters < minimumCharactersForTextPage { empty.append(pageNumber) }
            lines += pageLines
        }
        return (lines, empty)
    }

    /// Top two outline levels, in document order.
    static func outline(of document: PDFDocument) -> [OutlineEntry] {
        guard let root = document.outlineRoot else { return [] }
        var entries: [OutlineEntry] = []
        func walk(_ node: PDFOutline, depth: Int) {
            for childIndex in 0..<node.numberOfChildren {
                guard let child = node.child(at: childIndex) else { continue }
                if let label = child.label?.trimmingCharacters(in: .whitespacesAndNewlines), !label.isEmpty,
                   let page = child.destination?.page {
                    entries.append(OutlineEntry(title: label, page: document.index(for: page) + 1))
                }
                if depth < 2 { walk(child, depth: depth + 1) }
            }
        }
        walk(root, depth: 1)
        return entries
    }

    private static func makeLine(_ text: String, range: NSRange, in attributed: NSAttributedString, page: Int) -> TextLine {
        let firstVisible = text.firstIndex { !$0.isWhitespace }
        guard let firstVisible else { return TextLine(text, page: page) }
        let location = range.location + text.utf16.distance(from: text.startIndex, to: firstVisible)
        guard let font = attributed.attribute(.font, at: location, effectiveRange: nil) as? PlatformFont else {
            return TextLine(text, page: page)
        }
        let name = font.fontName.lowercased()
        #if canImport(AppKit)
        let traits = font.fontDescriptor.symbolicTraits
        let isBold = traits.contains(.bold)
        let isMonospaced = traits.contains(.monoSpace)
        #else
        let traits = font.fontDescriptor.symbolicTraits
        let isBold = traits.contains(.traitBold)
        let isMonospaced = traits.contains(.traitMonoSpace)
        #endif
        return TextLine(
            text,
            page: page,
            fontSize: Double(font.pointSize),
            isBold: isBold || name.contains("bold"),
            isMonospaced: isMonospaced || ["mono", "courier", "menlo", "consolas"].contains { name.contains($0) }
        )
    }
}
```

- [ ] **Step 5: Write `Ingest/DocumentIngestor.swift`**

```swift
import Foundation
import PDFKit

public struct IngestSettings: Sendable, Hashable {
    public var chunkLength = 400
    public var sectionWindowLength = 1500
    public init() {}
}

public struct IngestedDocument: Sendable {
    public let info: DocumentInfo
    public let sections: [Section]
    public let chunks: [Chunk]
}

public enum IngestError: Error, Equatable {
    case unreadablePDF(URL)
}

/// Turns a PDF (or plain text) into sections and chunks. No embedding happens here.
public enum DocumentIngestor {
    static let mathSymbols: Set<Character> = Set("=+−×÷^√∑∏∫≤≥≠≈∞πθλμσΔ∂±⇒⇔∨∧∈∉⊂⊆∪∩²³")
    static let mathHeavyShare = 0.05

    public static func ingestPDF(at url: URL, title: String? = nil, id: DocumentID = UUID().uuidString,
                                 settings: IngestSettings = IngestSettings()) throws -> IngestedDocument {
        guard let document = PDFDocument(url: url) else { throw IngestError.unreadablePDF(url) }
        let fallbackTitle = (document.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String)
            ?? url.deletingPathExtension().lastPathComponent
        return ingest(pdf: document, title: title ?? fallbackTitle, id: id, settings: settings)
    }

    /// Plain text: `## ` lines start sections (text before the first one is titled `title`);
    /// lines between ``` fences are code.
    public static func ingestText(_ text: String, title: String, id: DocumentID = UUID().uuidString,
                                  settings: IngestSettings = IngestSettings()) -> IngestedDocument {
        var sections: [DetectedSection] = [DetectedSection(title: title, lines: [])]
        var inCode = false
        for raw in text.components(separatedBy: "\n") {
            if raw.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                inCode.toggle()
                continue
            }
            if !inCode, raw.hasPrefix("## ") {
                sections.append(DetectedSection(title: String(raw.dropFirst(3)).trimmingCharacters(in: .whitespaces), lines: []))
                continue
            }
            sections[sections.count - 1].lines.append(TextLine(raw, isMonospaced: inCode))
        }
        return assemble(sections: sections, source: .provided, title: title, id: id, pageCount: 1,
                        pagesWithoutText: [], mathHeavyPages: [], settings: settings)
    }

    static func ingest(pdf document: PDFDocument, title: String, id: DocumentID, settings: IngestSettings) -> IngestedDocument {
        let extracted = PDFTextExtractor.lines(from: document)
        let detected = SectionDetector.detect(lines: extracted.lines, outline: PDFTextExtractor.outline(of: document),
                                              documentTitle: title, windowLength: settings.sectionWindowLength)
        return assemble(sections: detected.sections, source: detected.source, title: title, id: id,
                        pageCount: document.pageCount, pagesWithoutText: extracted.pagesWithoutText,
                        mathHeavyPages: mathHeavyPages(extracted.lines), settings: settings)
    }

    private static func assemble(sections detected: [DetectedSection], source: SectionSource, title: String, id: DocumentID,
                                 pageCount: Int, pagesWithoutText: [Int], mathHeavyPages: [Int],
                                 settings: IngestSettings) -> IngestedDocument {
        var sections: [Section] = []
        var chunks: [Chunk] = []
        for candidate in detected {
            guard let built = Chunker.build(candidate, documentID: id, sectionIndex: sections.count,
                                            maximumLength: settings.chunkLength) else { continue }
            sections.append(built.section)
            chunks += built.chunks
        }
        let info = DocumentInfo(id: id, title: title, pageCount: pageCount, sectionCount: sections.count,
                                chunkCount: chunks.count, sectionSource: source,
                                pagesWithoutText: pagesWithoutText, mathHeavyPages: mathHeavyPages)
        return IngestedDocument(info: info, sections: sections, chunks: chunks)
    }

    private static func mathHeavyPages(_ lines: [TextLine]) -> [Int] {
        let byPage = Dictionary(grouping: lines, by: \.page)
        return byPage.keys.sorted().filter { page in
            let characters = byPage[page]!.flatMap { $0.text.filter { !$0.isWhitespace } }
            guard !characters.isEmpty else { return false }
            let math = characters.filter { mathSymbols.contains($0) }.count
            return Double(math) / Double(characters.count) > mathHeavyShare
        }
    }
}
```

Note: sections that produce no chunks are dropped and indexes are reassigned, so `Section.index` is always the position in `sections`.

- [ ] **Step 6: Run — expect PASS.** This is the first contact with real PDFKit output, so check failures carefully:
  - If PDFKit merges lines or adds trailing spaces, adjust **the test's expectations of the text** only after printing `result.lines`. Don't weaken the font assertions.
  - If `isBold`/`isMonospaced` come back false, print `font.fontName` for each line and extend the name list in `makeLine`.
  - If the math test fails because PDFKit splits `x²` oddly, print the page characters and adjust the fixture text so the symbol share is clearly > 5%.

- [ ] **Step 7: Commit (when approved)** — `git commit -m "OnDeviceRAG: PDF and text ingest"`

---

### Task 6: Embedder protocol and the bge-small Core AI embedder

**Files:**
- Create: `Tools/make_package_fixtures.py`
- Create (generated): `Sources/OnDeviceRAG/Resources/bge-small-en-v1.5/tokenizer.json`, `tokenizer_config.json`, `LICENSE.txt`
- Create (generated): `Tests/OnDeviceRAGTests/Fixtures/bge_reference.json`
- Modify: `Package.swift` (add `resources: [.copy("Resources/bge-small-en-v1.5")]` to the `OnDeviceRAG` target)
- Create: `Sources/OnDeviceRAG/Embedding/Embedder.swift`
- Create: `Sources/OnDeviceRAG/Embedding/BGESmallEmbedder.swift`
- Create: `Tests/OnDeviceRAGTests/Support/ModelLocation.swift`
- Create: `Tests/OnDeviceRAGTests/Support/FakeEmbedder.swift`
- Test: `Tests/OnDeviceRAGTests/BGESmallEmbedderTests.swift`

**Interfaces:**
- Produces: `public enum EmbeddingKind: Sendable { case query, passage }`
- Produces: `public protocol Embedder: Sendable { var identifier: String { get }; var dimension: Int { get }; func embed(_ texts: [String], as kind: EmbeddingKind) async throws -> [[Float]] }` — vectors must be L2-normalized.
- Produces: `public actor BGESmallEmbedder: Embedder { public init(modelURL: URL) async throws; nonisolated func tokenIDs(for text: String, as kind: EmbeddingKind) -> [Int32] }`
- Produces: `public enum EmbedderError: Error { case missingFunction(String), missingOutput, missingTokenizer }`
- Produces (tests): `struct FakeEmbedder: Embedder` and `enum ModelLocation { static let url: URL; static var isAvailable: Bool }`

- [ ] **Step 1: Write `Tools/make_package_fixtures.py`**

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["sentence-transformers==5.1.2", "transformers==4.57.3"]
# ///
# Writes the bundled tokenizer files and the Swift tests' reference token IDs + vectors.
# Usage (from Packages/OnDeviceRAG): uv run Tools/make_package_fixtures.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

NAME = "BAAI/bge-small-en-v1.5"
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "Sources/OnDeviceRAG/Resources/bge-small-en-v1.5"
FIX = ROOT / "Tests/OnDeviceRAGTests/Fixtures/bge_reference.json"
PREFIX = "Represent this sentence for searching relevant passages: "

model = SentenceTransformer(NAME)
tok = model.tokenizer
RES.mkdir(parents=True, exist_ok=True)
tok.save_pretrained(str(RES))
for extra in ("vocab.txt", "special_tokens_map.json"):  # tokenizer.json already contains these
    (RES / extra).unlink(missing_ok=True)
(RES / "LICENSE.txt").write_text("bge-small-en-v1.5 tokenizer files (c) BAAI, MIT License. https://huggingface.co/BAAI/bge-small-en-v1.5\n")

texts = [
    ("passage", "Hello, my dog is cute"),
    ("passage", "Integers are encoded as i<n>e, e.g. encode_zorp(42) -> 'i42e'."),
    ("passage", "Café naïve résumé — accents and dashes… plus emoji 🙂 and 中文."),
    ("passage", "    def merge(a, b):\n        return sorted(a + b)"),
    ("passage", "x² + 2x − 3 = 0, so x = −3 or x = 1."),
    ("query", "How do I encode a list in ZORP-7?"),
    ("passage", " ".join(["token"] * 600)),  # longer than 512 tokens -> truncated
]
out = []
for kind, text in texts:
    full = PREFIX + text if kind == "query" else text
    ids = tok(full, truncation=True, max_length=512)["input_ids"]
    vec = model.encode(full, normalize_embeddings=True).tolist()
    out.append({"kind": kind, "text": text, "ids": ids, "vector": vec})
FIX.write_text(json.dumps(out))
print("wrote", RES, "and", FIX, [len(o["ids"]) for o in out])
```

- [ ] **Step 2: Run it** — `cd Packages/OnDeviceRAG && uv run Tools/make_package_fixtures.py`. Expected: the last count is `512`. Confirm `tokenizer_config.json` has `"tokenizer_class": "BertTokenizer"` (`grep tokenizer_class Sources/OnDeviceRAG/Resources/bge-small-en-v1.5/tokenizer_config.json`). If `sentence-transformers==5.1.2` doesn't resolve, use the newest version that works with `transformers==4.57.3` and pin it in the script.

- [ ] **Step 3: Add the resource line to `Package.swift`**

```swift
        .target(
            name: "OnDeviceRAG",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            resources: [.copy("Resources/bge-small-en-v1.5")]
        ),
```

- [ ] **Step 4: Write the test support files**

`Support/ModelLocation.swift`:

```swift
import Foundation

/// Where the (git-ignored) bge-small Core AI model lives on this Mac.
enum ModelLocation {
    static let url: URL = {
        if let path = ProcessInfo.processInfo.environment["ONDEVICE_RAG_MODEL"] {
            return URL(fileURLWithPath: path)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Developer/coreai-models/exports/bge_small_buckets_float16.aimodel")
    }()

    static var isAvailable: Bool { FileManager.default.fileExists(atPath: url.path) }
}
```

`Support/FakeEmbedder.swift`:

```swift
import Foundation
@testable import OnDeviceRAG

/// Deterministic bag-of-words embedder: each BM25 token adds 1 to a hashed dimension,
/// then the vector is L2-normalized. Texts sharing words get high cosine.
struct FakeEmbedder: Embedder {
    let identifier = "fake-bag-of-words"
    let dimension = 64

    func embed(_ texts: [String], as kind: EmbeddingKind) async throws -> [[Float]] {
        texts.map { text in
            var vector = [Float](repeating: 0, count: dimension)
            for token in BM25Index.tokenize(text) {
                let hash = token.unicodeScalars.reduce(UInt32(2166136261)) { ($0 ^ $1.value) &* 16777619 }
                vector[Int(hash % UInt32(dimension))] += 1
            }
            let norm = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
            return norm > 0 ? vector.map { $0 / norm } : vector
        }
    }
}
```

- [ ] **Step 5: Write the failing (gated) tests**

`BGESmallEmbedderTests.swift`:

```swift
import Foundation
import Testing
@testable import OnDeviceRAG

private struct ReferenceItem: Decodable {
    let kind: String
    let text: String
    let ids: [Int32]
    let vector: [Float]
}

private func loadReference() throws -> [ReferenceItem] {
    let url = try #require(Bundle.module.url(forResource: "bge_reference", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode([ReferenceItem].self, from: Data(contentsOf: url))
}

@Suite(.enabled(if: ModelLocation.isAvailable, "bge-small model not found; set ONDEVICE_RAG_MODEL"), .serialized)
struct BGESmallEmbedderTests {
    @Test func tokenIDsMatchPython() async throws {
        let embedder = try await BGESmallEmbedder(modelURL: ModelLocation.url)
        for item in try loadReference() {
            let ids = embedder.tokenIDs(for: item.text, as: item.kind == "query" ? .query : .passage)
            #expect(ids == item.ids, "token IDs differ for: \(item.text.prefix(40))")
        }
    }

    @Test func vectorsMatchPyTorch() async throws {
        let embedder = try await BGESmallEmbedder(modelURL: ModelLocation.url)
        #expect(embedder.dimension == 384)
        for item in try loadReference() {
            let vector = try await embedder.embed([item.text], as: item.kind == "query" ? .query : .passage)[0]
            let cosine = zip(vector, item.vector).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            #expect(cosine >= 0.999, "cosine \(cosine) for: \(item.text.prefix(40))")
        }
    }
}
```

- [ ] **Step 6: Run — expect compile failure** (`cannot find 'BGESmallEmbedder'`). Use `swift test --package-path Packages/OnDeviceRAG --filter BGESmallEmbedderTests`.

- [ ] **Step 7: Write `Embedding/Embedder.swift`**

```swift
public enum EmbeddingKind: Sendable {
    /// A search question (bge adds its query instruction).
    case query
    /// A chunk of a document.
    case passage
}

/// Turns texts into L2-normalized vectors. Swap in another implementation to use a different model.
public protocol Embedder: Sendable {
    /// Stable ID of the model + export (stored in indexes from milestone 2 on).
    var identifier: String { get }
    var dimension: Int { get }
    func embed(_ texts: [String], as kind: EmbeddingKind) async throws -> [[Float]]
}
```

- [ ] **Step 8: Write `Embedding/BGESmallEmbedder.swift`**

Core AI calls are the ones the spike verified on iPhone and Mac (`AIModel(contentsOf:)`, `loadFunction(named:)`, `run(inputs:)`, float16 output read as `Float16`).

```swift
import CoreAI
import Foundation
import Tokenizers

public enum EmbedderError: Error {
    case missingFunction(String)
    case missingOutput
    case missingTokenizer
}

/// bge-small-en-v1.5 (33M parameters, MIT) running through Core AI. The export has one
/// function per fixed input length (`embed_128/256/512`); texts are padded to the nearest one,
/// because flexible lengths cost ~2.5 s per new length on iPhone. Pooling and L2 normalization
/// are inside the model.
public actor BGESmallEmbedder: Embedder {
    public nonisolated let identifier = "bge-small-en-v1.5/float16/fixed-128-256-512"
    public nonisolated let dimension = 384

    static let queryPrefix = "Represent this sentence for searching relevant passages: "
    static let lengths = [128, 256, 512]
    static let separatorID: Int32 = 102  // [SEP]
    static let paddingID: Int32 = 0      // [PAD]

    private nonisolated let tokenizer: any Tokenizer
    private let functions: [Int: InferenceFunction]

    public init(modelURL: URL) async throws {
        guard let folder = Bundle.module.url(forResource: "bge-small-en-v1.5", withExtension: nil) else {
            throw EmbedderError.missingTokenizer
        }
        tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        let model = try await AIModel(contentsOf: modelURL)
        var loaded: [Int: InferenceFunction] = [:]
        for length in Self.lengths {
            guard let function = try model.loadFunction(named: "embed_\(length)") else {
                throw EmbedderError.missingFunction("embed_\(length)")
            }
            loaded[length] = function
        }
        functions = loaded
    }

    /// Token IDs with [CLS]/[SEP], truncated to 512 like Python's `truncation=True, max_length=512`.
    nonisolated func tokenIDs(for text: String, as kind: EmbeddingKind) -> [Int32] {
        let full = kind == .query ? Self.queryPrefix + text : text
        var ids = tokenizer.encode(text: full).map(Int32.init)
        if ids.count > 512 {
            ids = Array(ids.prefix(511)) + [Self.separatorID]
        }
        return ids
    }

    public func embed(_ texts: [String], as kind: EmbeddingKind) async throws -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        for text in texts {
            vectors.append(try await embed(ids: tokenIDs(for: text, as: kind)))
        }
        return vectors
    }

    private func embed(ids: [Int32]) async throws -> [Float] {
        let length = Self.lengths.first { $0 >= ids.count } ?? 512
        let padding = length - ids.count
        let paddedIDs = ids + Array(repeating: Self.paddingID, count: padding)
        let mask = Array(repeating: Int32(1), count: ids.count) + Array(repeating: Int32(0), count: padding)
        guard let function = functions[length] else { throw EmbedderError.missingFunction("embed_\(length)") }
        var outputs = try await function.run(inputs: [
            "input_ids": NDArray(scalars: paddedIDs, shape: [1, length]),
            "attention_mask": NDArray(scalars: mask, shape: [1, length]),
        ])
        guard let output = outputs.remove("embedding")?.ndArray else { throw EmbedderError.missingOutput }
        let count = output.shape.reduce(1, *)
        var result: [Float] = []
        if output.scalarType == .float16 {
            output.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
                result = UnsafeBufferPointer(start: pointer, count: count).map(Float.init)
            }
        } else {
            output.view(as: Float.self).withUnsafePointer { pointer, _, _ in
                result = Array(UnsafeBufferPointer(start: pointer, count: count))
            }
        }
        return result
    }
}
```

- [ ] **Step 9: Run — expect PASS** (first run includes the one-time Core AI specialization; on the Mac that took ~0.1–0.3 s per length in the spike, but allow a few minutes).
  - **If `tokenIDsMatchPython` fails**, stop and report the first differing text and both ID lists. Don't hand-patch IDs; the likely fix is in how the tokenizer is loaded (e.g. `strict:`/added tokens), and the researcher decides.
  - **If Swift 6 strict concurrency rejects** calling `function.run` (non-`Sendable` `InferenceFunction`/`NDArray` crossing the actor boundary), first try marking the import `@preconcurrency import CoreAI`. If that isn't enough, report the exact diagnostic before changing the language mode.

- [ ] **Step 10: Commit (when approved)** — check that no `.aimodel` is staged (`git status --short | grep aimodel` prints nothing), then `git commit -m "OnDeviceRAG: bge-small Core AI embedder with bundled tokenizer"`

---

### Task 7: In-memory index and retriever

**Files:**
- Create: `Sources/OnDeviceRAG/Search/InMemoryIndex.swift`
- Create: `Sources/OnDeviceRAG/Search/Retriever.swift`
- Test: `Tests/OnDeviceRAGTests/RetrieverTests.swift`

**Interfaces:**
- Consumes: `BM25Index`, `RankFusion` (Task 2), `IngestedDocument` (Task 5), `Embedder`, `FakeEmbedder` (Task 6).
- Produces: `struct InMemoryIndex: Sendable { init(dimension: Int, k1: Double, b: Double); mutating func add(_ document: IngestedDocument, vectors: [[Float]]); mutating func remove(_ id: DocumentID); var documents: [IngestedDocument] { get }; var chunks: [Chunk] { get }; func denseOrder(for query: [Float]) -> [Int]; func keywordOrder(for query: String) -> [Int]; func document(_ id: DocumentID) -> IngestedDocument? }`
- Produces: `struct RetrievalSettings: Sendable, Hashable { var fusionConstant = 60.0; var candidateCount = 20; var historyMessages = 2 }`
- Produces: `enum Retriever { static func queryText(question: String, history: [String], historyMessages: Int) -> String; static func retrieve(question: String, history: [String], index: InMemoryIndex, embedder: any Embedder, settings: RetrievalSettings, limit: Int) async throws -> [RetrievedChunk] }`

- [ ] **Step 1: Write the failing tests**

```swift
import Testing
@testable import OnDeviceRAG

struct RetrieverTests {
    private func makeIndex() async throws -> InMemoryIndex {
        let docs = [
            DocumentIngestor.ingestText("## Integers\nIntegers are encoded as i<n>e.\n\n## Lists\nLists are encoded as l<items>e.", title: "ZORP", id: "zorp"),
            DocumentIngestor.ingestText("## Cache\nThe cache stores loaded values.", title: "Cache SDK", id: "cache"),
        ]
        var index = InMemoryIndex(dimension: 64, k1: 1.2, b: 0.75)
        for doc in docs {
            index.add(doc, vectors: try await FakeEmbedder().embed(doc.chunks.map(\.indexedText), as: .passage))
        }
        return index
    }

    @Test func questionComesFirstThenRecentHistoryNewestFirst() {
        let text = Retriever.queryText(question: "Q?", history: ["old", "middle", "newest"], historyMessages: 2)
        #expect(text == "Q?\nnewest\nmiddle")
        #expect(Retriever.queryText(question: "Q?", history: [], historyMessages: 2) == "Q?")
    }

    @Test func retrievesTheMatchingChunkFirstWithBothRanks() async throws {
        let index = try await makeIndex()
        let results = try await Retriever.retrieve(question: "how are lists encoded", history: [], index: index,
                                                   embedder: FakeEmbedder(), settings: RetrievalSettings(), limit: 20)
        #expect(results.first?.chunk.sectionTitle == "Lists")
        #expect(results.first?.documentTitle == "ZORP")
        #expect(results.first?.denseRank != nil)
        #expect(results.first?.keywordRank != nil)
        #expect(results.count == index.chunks.count) // 3 chunks < limit
    }

    @Test func limitCapsResults() async throws {
        let index = try await makeIndex()
        let results = try await Retriever.retrieve(question: "cache", history: [], index: index,
                                                   embedder: FakeEmbedder(), settings: RetrievalSettings(), limit: 1)
        #expect(results.map(\.chunk.sectionTitle) == ["Cache"])
    }

    @Test func removingADocumentDropsItsChunks() async throws {
        var index = try await makeIndex()
        index.remove("zorp")
        #expect(index.chunks.map(\.documentID) == ["cache"])
        #expect(index.denseOrder(for: try await FakeEmbedder().embed(["lists"], as: .query)[0]).count == 1)
        #expect(index.document("zorp") == nil)
    }

    @Test func emptyIndexReturnsNothing() async throws {
        let results = try await Retriever.retrieve(question: "anything", history: [], index: InMemoryIndex(dimension: 64, k1: 1.2, b: 0.75),
                                                   embedder: FakeEmbedder(), settings: RetrievalSettings(), limit: 20)
        #expect(results.isEmpty)
    }
}
```

- [ ] **Step 2: Run — expect compile failure.**

- [ ] **Step 3: Write `Search/InMemoryIndex.swift`**

```swift
import Accelerate

/// All chunks of all documents, their vectors (one flat array, `dimension` floats per chunk)
/// and a BM25 index over their indexed text. Milestone 2 adds saving/loading.
struct InMemoryIndex: Sendable {
    let dimension: Int
    let k1: Double
    let b: Double

    private(set) var documents: [IngestedDocument] = []
    private(set) var chunks: [Chunk] = []
    private var vectors: [Float] = []
    private var keywordIndex: BM25Index

    init(dimension: Int, k1: Double, b: Double) {
        self.dimension = dimension
        self.k1 = k1
        self.b = b
        keywordIndex = BM25Index(documents: [], k1: k1, b: b)
    }

    /// `vectors` must be in the document's chunk order, `dimension` floats each.
    mutating func add(_ document: IngestedDocument, vectors documentVectors: [[Float]]) {
        precondition(documentVectors.count == document.chunks.count, "one vector per chunk")
        precondition(documentVectors.allSatisfy { $0.count == dimension }, "vector dimension mismatch")
        remove(document.info.id)
        documents.append(document)
        chunks += document.chunks
        vectors += documentVectors.flatMap { $0 }
        rebuildKeywordIndex()
    }

    mutating func remove(_ id: DocumentID) {
        guard documents.contains(where: { $0.info.id == id }) else { return }
        var keptVectors: [Float] = []
        for (position, chunk) in chunks.enumerated() where chunk.documentID != id {
            keptVectors += vectors[(position * dimension)..<((position + 1) * dimension)]
        }
        vectors = keptVectors
        chunks.removeAll { $0.documentID == id }
        documents.removeAll { $0.info.id == id }
        rebuildKeywordIndex()
    }

    func document(_ id: DocumentID) -> IngestedDocument? {
        documents.first { $0.info.id == id }
    }

    /// Chunk indices by cosine similarity (vectors are normalized, so a dot product), best first;
    /// ties go to the earlier chunk.
    func denseOrder(for query: [Float]) -> [Int] {
        guard !chunks.isEmpty else { return [] }
        var scores = [Float](repeating: 0, count: chunks.count)
        vDSP_mmul(vectors, 1, query, 1, &scores, 1, vDSP_Length(chunks.count), 1, vDSP_Length(dimension))
        return chunks.indices.sorted { scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1 }
    }

    /// Chunk indices with a positive BM25 score, best first; ties go to the earlier chunk.
    func keywordOrder(for query: String) -> [Int] {
        let scores = keywordIndex.scores(for: query)
        return chunks.indices
            .filter { scores[$0] > 0 }
            .sorted { scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1 }
    }

    private mutating func rebuildKeywordIndex() {
        keywordIndex = BM25Index(documents: chunks.map(\.indexedText), k1: k1, b: b)
    }
}
```

- [ ] **Step 4: Write `Search/Retriever.swift`**

```swift
public struct RetrievalSettings: Sendable, Hashable {
    public var fusionConstant = 60.0
    public var candidateCount = 20
    public var historyMessages = 2
    public init() {}
}

/// Dense (embedding) + BM25 rankings over every chunk, merged with Reciprocal Rank Fusion.
enum Retriever {
    /// The question, then up to `historyMessages` recent messages, newest first — so when the
    /// embedder truncates at 512 tokens, only history is cut.
    static func queryText(question: String, history: [String], historyMessages: Int) -> String {
        ([question] + history.suffix(historyMessages).reversed()).joined(separator: "\n")
    }

    static func retrieve(question: String, history: [String], index: InMemoryIndex, embedder: any Embedder,
                         settings: RetrievalSettings, limit: Int) async throws -> [RetrievedChunk] {
        guard !index.chunks.isEmpty else { return [] }
        let query = queryText(question: question, history: history, historyMessages: settings.historyMessages)
        let queryVector = try await embedder.embed([query], as: .query)[0]
        let dense = index.denseOrder(for: queryVector)
        let keyword = index.keywordOrder(for: query)
        let denseRanks = Dictionary(uniqueKeysWithValues: dense.enumerated().map { ($1, $0 + 1) })
        let keywordRanks = Dictionary(uniqueKeysWithValues: keyword.enumerated().map { ($1, $0 + 1) })
        let titles = Dictionary(uniqueKeysWithValues: index.documents.map { ($0.info.id, $0.info.title) })

        return RankFusion.fuse([dense, keyword], count: index.chunks.count, constant: settings.fusionConstant)
            .prefix(limit)
            .map { entry in
                let chunk = index.chunks[entry.index]
                return RetrievedChunk(chunk: chunk, documentTitle: titles[chunk.documentID] ?? "",
                                      fusedScore: entry.score, denseRank: denseRanks[entry.index],
                                      keywordRank: keywordRanks[entry.index])
            }
    }
}
```

`RetrievedChunk` needs a memberwise init usable here; it is internal by default for a public struct in the same module, so no change is needed.

- [ ] **Step 5: Run — expect PASS.**

- [ ] **Step 6: Commit (when approved)** — `git commit -m "OnDeviceRAG: in-memory hybrid index and retriever"`

---

### Task 8: Token counters and the context packer

**Files:**
- Create: `Sources/OnDeviceRAG/Packing/TokenCounter.swift`
- Create: `Sources/OnDeviceRAG/Packing/ContextPacker.swift`
- Test: `Tests/OnDeviceRAGTests/ContextPackerTests.swift`

**Interfaces:**
- Consumes: `RetrievedChunk`, `RAGContext`, `ContextSection`, `TokenCount`, `Section`, `Chunk` (Task 1); `InMemoryIndex.document(_:)` (Task 7).
- Produces: `public protocol TokenCounter: Sendable { func count(_ text: String) async -> TokenCount }`
- Produces: `public struct EstimatingTokenCounter: TokenCounter { public var charactersPerToken = 3.0; public init() }` — `ceil(characters / charactersPerToken)`, `isEstimate = true`.
- Produces: `public struct AppleTokenCounter: TokenCounter { public init() }` — Foundation Models count when available, else the estimate.
- Produces: `enum ContextPacker { static func pack(_ candidates: [RetrievedChunk], index: InMemoryIndex, tokenBudget: Int, counter: any TokenCounter) async -> RAGContext }`

Rules (design §7): walk candidates in rank order; skip a section already used; add the whole section block (`## <title> (<document>, <pages>)` + newline + section text) if it fits the remaining budget; otherwise add a **window** — the matched chunk, then neighbors alternately after/before while they fit (window text = chunks in position order joined with `\n`); blocks are joined with a blank line; stop when nothing fits or candidates run out. If nothing was added and there is a candidate, include the top chunk truncated to fit. Finally count the whole text; while it is over budget, drop the last block (counts aren't always additive). Never exceed the budget.

- [ ] **Step 1: Write the failing tests**

All tests use `EstimatingTokenCounter` (3 characters per token), so numbers are exact.

```swift
import Testing
@testable import OnDeviceRAG

struct ContextPackerTests {
    private let counter = EstimatingTokenCounter()

    private func index(_ text: String, title: String = "Doc", id: String = "d") async throws -> InMemoryIndex {
        let doc = DocumentIngestor.ingestText(text, title: title, id: id)
        var index = InMemoryIndex(dimension: 64, k1: 1.2, b: 0.75)
        index.add(doc, vectors: try await FakeEmbedder().embed(doc.chunks.map(\.indexedText), as: .passage))
        return index
    }

    private func candidates(_ index: InMemoryIndex, _ positions: [(section: Int, chunk: Int)]) -> [RetrievedChunk] {
        positions.map { position in
            let chunk = index.chunks.first { $0.sectionIndex == position.section && $0.position == position.chunk }!
            return RetrievedChunk(chunk: chunk, documentTitle: "Doc", fusedScore: 0, denseRank: nil, keywordRank: nil)
        }
    }

    @Test func wholeSectionsInRankOrderWithHeaders() async throws {
        let idx = try await index("## A\nAlpha text.\n\n## B\nBeta text.")
        // Section 0 is the empty preamble → dropped, so A = 0, B = 1.
        let context = await ContextPacker.pack(candidates(idx, [(1, 0), (0, 0)]), index: idx, tokenBudget: 2000, counter: counter)
        #expect(context.text == "## B (Doc, p. 1)\nBeta text.\n\n## A (Doc, p. 1)\nAlpha text.")
        #expect(context.sections.map(\.title) == ["B", "A"])
        #expect(context.sections.allSatisfy(\.isWholeSection))
        #expect(context.tokenCountIsEstimate)
        #expect(context.tokenCount == Int((Double(context.text.count) / 3).rounded(.up)))
    }

    @Test func sameSectionIsUsedOnce() async throws {
        let sentences = (1...6).map { "Sentence number \($0) is here." }.joined(separator: " ")
        let idx = try await index("## A\n\(sentences)")
        let chunkCount = idx.chunks.count
        let context = await ContextPacker.pack(candidates(idx, (0..<chunkCount).map { (0, $0) }), index: idx, tokenBudget: 2000, counter: counter)
        #expect(context.sections.count == 1)
    }

    @Test func oversizedSectionBecomesAWindowAroundTheMatch() async throws {
        // 12 sentences of ~200 chars → several 400-char chunks; budget fits ~2 chunks.
        let long = (1...12).map { "Sentence \($0) " + String(repeating: "filler ", count: 26) + "end." }.joined(separator: " ")
        let idx = try await index("## Big\n\(long)")
        #expect(idx.chunks.count >= 5)
        let context = await ContextPacker.pack(candidates(idx, [(0, 3)]), index: idx, tokenBudget: 300, counter: counter)
        #expect(context.sections.count == 1)
        #expect(context.sections[0].isWholeSection == false)
        #expect(context.text.contains(idx.chunks[3].text))
        #expect(context.tokenCount <= 300)
    }

    @Test func neverExceedsTheBudget() async throws {
        let text = (1...8).map { "## S\($0)\n" + String(repeating: "word\($0) ", count: 60) + "." }.joined(separator: "\n\n")
        let idx = try await index(text)
        let all = candidates(idx, idx.chunks.map { ($0.sectionIndex, $0.position) })
        for budget in [5, 50, 150, 400, 900] {
            let context = await ContextPacker.pack(all, index: idx, tokenBudget: budget, counter: counter)
            #expect(context.tokenCount <= budget, "budget \(budget)")
            #expect(!context.text.isEmpty, "top chunk must always be included (budget \(budget))")
        }
    }

    @Test func tinyBudgetTruncatesTheTopChunk() async throws {
        let idx = try await index("## A\n" + String(repeating: "alpha ", count: 50) + ".")
        let context = await ContextPacker.pack(candidates(idx, [(0, 0)]), index: idx, tokenBudget: 10, counter: counter)
        #expect(context.tokenCount <= 10)
        #expect(context.text.hasPrefix("## A"))
    }

    @Test func noCandidatesGiveEmptyContext() async throws {
        let idx = try await index("## A\nAlpha.")
        let context = await ContextPacker.pack([], index: idx, tokenBudget: 2000, counter: counter)
        #expect(context.text.isEmpty)
        #expect(context.sections.isEmpty)
        #expect(context.tokenCount == 0)
    }
}
```

- [ ] **Step 2: Run — expect compile failure.**

- [ ] **Step 3: Write `Packing/TokenCounter.swift`**

```swift
import Foundation
import FoundationModels

/// Counts tokens for a model's prompt. Supply your own for models other than Apple's.
public protocol TokenCounter: Sendable {
    func count(_ text: String) async -> TokenCount
}

/// Characters ÷ 3, rounded up — deliberately conservative (English is ~4 characters per token).
public struct EstimatingTokenCounter: TokenCounter {
    public var charactersPerToken = 3.0
    public init() {}

    public func count(_ text: String) async -> TokenCount {
        TokenCount(value: Int((Double(text.count) / charactersPerToken).rounded(.up)), isEstimate: true)
    }
}

/// Apple's on-device model's own count when Apple Intelligence is available; otherwise the estimate.
public struct AppleTokenCounter: TokenCounter {
    public init() {}

    public func count(_ text: String) async -> TokenCount {
        let model = SystemLanguageModel.default
        if case .available = model.availability, let value = try? await model.tokenCount(for: text) {
            return TokenCount(value: value, isEstimate: false)
        }
        return await EstimatingTokenCounter().count(text)
    }
}
```

If `tokenCount(for: text)` doesn't compile because `String` isn't `PromptRepresentable`, use `model.tokenCount(for: Prompt(text))`.

- [ ] **Step 4: Write `Packing/ContextPacker.swift`**

```swift
/// Packs retrieved chunks into prompt text within a token budget, "small-to-big":
/// whole sections when they fit, otherwise a window of chunks around the match.
enum ContextPacker {
    private struct Block {
        var text: String
        var section: ContextSection
    }

    static let blockSeparator = "\n\n"

    static func pack(_ candidates: [RetrievedChunk], index: InMemoryIndex, tokenBudget: Int, counter: any TokenCounter) async -> RAGContext {
        var blocks: [Block] = []
        var used = 0
        var seen: Set<String> = []

        for candidate in candidates {
            let chunk = candidate.chunk
            let key = "\(chunk.documentID)#\(chunk.sectionIndex)"
            guard !seen.contains(key), let document = index.document(chunk.documentID),
                  chunk.sectionIndex < document.sections.count else { continue }
            let section = document.sections[chunk.sectionIndex]
            let separatorCost = blocks.isEmpty ? 0 : await counter.count(blockSeparator).value
            let remaining = tokenBudget - used - separatorCost
            if remaining <= 0 { break }

            let header = "## \(section.title) (\(document.info.title), \(section.pages.label))"
            let whole = header + "\n" + section.text
            let wholeCost = await counter.count(whole).value
            if wholeCost <= remaining {
                blocks.append(Block(text: whole, section: contextSection(section, document, section.pages, whole: true)))
                used += separatorCost + wholeCost
                seen.insert(key)
                continue
            }
            let sectionChunks = document.chunks.filter { $0.sectionIndex == chunk.sectionIndex }
            if let window = await window(around: chunk.position, in: sectionChunks, documentTitle: document.info.title,
                                         section: section, remaining: remaining, counter: counter) {
                blocks.append(Block(text: window.text, section: contextSection(section, document, window.pages, whole: false)))
                used += separatorCost + window.cost
                seen.insert(key)
            }
        }

        if blocks.isEmpty, let top = candidates.first, let document = index.document(top.chunk.documentID),
           top.chunk.sectionIndex < document.sections.count {
            let section = document.sections[top.chunk.sectionIndex]
            let full = "## \(section.title) (\(document.info.title), \(top.chunk.pages.label))\n" + top.chunk.text
            let text = await truncate(full, toFit: tokenBudget, counter: counter)
            if !text.isEmpty {
                blocks.append(Block(text: text, section: contextSection(section, document, top.chunk.pages, whole: false)))
            }
        }

        // Token counts aren't always additive: re-count the joined text and drop blocks until it fits.
        while true {
            let text = blocks.map(\.text).joined(separator: blockSeparator)
            let count = text.isEmpty ? TokenCount(value: 0, isEstimate: false) : await counter.count(text)
            if count.value <= tokenBudget || blocks.count <= 1 {
                let estimate = text.isEmpty ? false : count.isEstimate
                return RAGContext(text: text, sections: blocks.map(\.section), tokenCount: count.value, tokenCountIsEstimate: estimate)
            }
            blocks.removeLast()
        }
    }

    private static func contextSection(_ section: Section, _ document: IngestedDocument, _ pages: PageRange, whole: Bool) -> ContextSection {
        ContextSection(documentID: document.info.id, documentTitle: document.info.title, sectionIndex: section.index,
                       title: section.title, pages: pages, isWholeSection: whole)
    }

    /// The matched chunk, then neighbors after/before alternately, while the block fits.
    private static func window(around position: Int, in chunks: [Chunk], documentTitle: String, section: Section,
                               remaining: Int, counter: any TokenCounter) async -> (text: String, pages: PageRange, cost: Int)? {
        guard let matched = chunks.firstIndex(where: { $0.position == position }) else { return nil }
        func render(_ range: ClosedRange<Int>) -> (String, PageRange) {
            let selected = chunks[range]
            let pages = selected.dropFirst().reduce(selected.first!.pages) { $0.union($1.pages) }
            return ("## \(section.title) (\(documentTitle), \(pages.label))\n" + selected.map(\.text).joined(separator: "\n"), pages)
        }

        var range = matched...matched
        var (text, pages) = render(range)
        var cost = await counter.count(text).value
        guard cost <= remaining else { return nil }

        var growAfter = true
        var blockedAfter = false
        var blockedBefore = false
        while !(blockedAfter && blockedBefore) {
            let candidate: ClosedRange<Int>?
            if growAfter && !blockedAfter {
                candidate = range.upperBound + 1 < chunks.count ? range.lowerBound...(range.upperBound + 1) : nil
                if candidate == nil { blockedAfter = true }
            } else if !blockedBefore {
                candidate = range.lowerBound > 0 ? (range.lowerBound - 1)...range.upperBound : nil
                if candidate == nil { blockedBefore = true }
            } else {
                candidate = nil
            }
            if let candidate {
                let (candidateText, candidatePages) = render(candidate)
                let candidateCost = await counter.count(candidateText).value
                if candidateCost <= remaining {
                    range = candidate
                    (text, pages, cost) = (candidateText, candidatePages, candidateCost)
                } else if growAfter {
                    blockedAfter = true
                } else {
                    blockedBefore = true
                }
            }
            growAfter.toggle()
        }
        return (text, pages, cost)
    }

    /// Longest prefix (cut at a space when possible) whose count fits; binary search on length.
    private static func truncate(_ text: String, toFit budget: Int, counter: any TokenCounter) async -> String {
        if await counter.count(text).value <= budget { return text }
        var low = 0
        var high = text.count
        while low < high {
            let middle = (low + high + 1) / 2
            if await counter.count(String(text.prefix(middle))).value <= budget { low = middle } else { high = middle - 1 }
        }
        let prefix = text.prefix(low)
        if let space = prefix.lastIndex(of: " "), prefix.distance(from: prefix.startIndex, to: space) > low / 2 {
            return String(prefix[..<space])
        }
        return String(prefix)
    }
}
```

- [ ] **Step 5: Run — expect PASS.** If `oversizedSectionBecomesAWindowAroundTheMatch` fails on `chunks.count >= 5`, print the chunk lengths and lengthen the fixture text; the window logic is what's under test.

- [ ] **Step 6: Commit (when approved)** — `git commit -m "OnDeviceRAG: token counters and small-to-big context packer"`

---

### Task 9: `RAGEngine`, configuration, reranker hook, end-to-end checks

**Files:**
- Create: `Sources/OnDeviceRAG/Engine/RAGConfiguration.swift`
- Create: `Sources/OnDeviceRAG/Engine/RAGEngine.swift`
- Test: `Tests/OnDeviceRAGTests/EngineTests.swift`
- Modify: `Tests/OnDeviceRAGTests/BGESmallEmbedderTests.swift` (add the 40-task retrieval check)

**Interfaces:**
- Consumes: everything above.
- Produces: `public protocol Reranker: Sendable { func rerank(_ question: String, _ candidates: [RetrievedChunk]) async throws -> [RetrievedChunk] }`
- Produces: `public struct RAGConfiguration: Sendable { public var embedder: any Embedder; public var tokenCounter: any TokenCounter = AppleTokenCounter(); public var reranker: (any Reranker)? = nil; public var rerankCount = 10; public var ingest = IngestSettings(); public var retrieval = RetrievalSettings(); public var bm25K1 = 1.2; public var bm25B = 0.75; public var tokenBudget = 2000; public init(embedder: any Embedder) }`
- Produces: `public actor RAGEngine { public init(configuration: RAGConfiguration); public var documents: [DocumentInfo] { get }; public func addDocument(pdfAt url: URL, title: String? = nil) async throws -> DocumentInfo; public func addDocument(_ document: IngestedDocument) async throws -> DocumentInfo; public func removeDocument(id: DocumentID); public func search(_ question: String, history: [String] = [], limit: Int? = nil) async throws -> [RetrievedChunk]; public func context(for question: String, history: [String] = [], tokenBudget: Int? = nil) async throws -> RAGContext }`

Reranker behavior: when set, `context(for:)` passes the first `rerankCount` candidates to it and puts its result in front of the remaining candidates. If it throws, or returns anything other than a reordering of exactly those chunks, the original order is kept.

- [ ] **Step 1: Write the failing tests**

`EngineTests.swift`:

```swift
import Foundation
import Testing
@testable import OnDeviceRAG

private struct ReversingReranker: Reranker {
    func rerank(_ question: String, _ candidates: [RetrievedChunk]) async throws -> [RetrievedChunk] { candidates.reversed() }
}

private struct BrokenReranker: Reranker {
    func rerank(_ question: String, _ candidates: [RetrievedChunk]) async throws -> [RetrievedChunk] { Array(candidates.prefix(1)) }
}

private func configuration(reranker: (any Reranker)? = nil) -> RAGConfiguration {
    var configuration = RAGConfiguration(embedder: FakeEmbedder())
    configuration.tokenCounter = EstimatingTokenCounter()
    configuration.reranker = reranker
    return configuration
}

private let zorp = "## Integers\nIntegers are encoded as i<n>e.\n\n## Lists\nLists are encoded as l<items>e."
private let cache = "## Cache\nThe cache stores loaded values."

struct EngineTests {
    @Test func addSearchContextRemove() async throws {
        let engine = RAGEngine(configuration: configuration())
        _ = try await engine.addDocument(DocumentIngestor.ingestText(zorp, title: "ZORP", id: "zorp"))
        let info = try await engine.addDocument(DocumentIngestor.ingestText(cache, title: "Cache SDK", id: "cache"))
        #expect(info.sectionCount == 1)
        #expect(await engine.documents.map(\.id) == ["zorp", "cache"])

        let hits = try await engine.search("how are lists encoded")
        #expect(hits.first?.chunk.sectionTitle == "Lists")

        let context = try await engine.context(for: "how are lists encoded")
        #expect(context.text.hasPrefix("## Lists (ZORP, p. 1)\nLists are encoded as l<items>e."))
        #expect(context.tokenCount <= 2000)

        await engine.removeDocument(id: "zorp")
        #expect(try await engine.search("lists").allSatisfy { $0.chunk.documentID == "cache" })
    }

    @Test func historyIsOptionalAndUsed() async throws {
        let engine = RAGEngine(configuration: configuration())
        _ = try await engine.addDocument(DocumentIngestor.ingestText(zorp + "\n\n" + cache, title: "Mixed", id: "m"))
        let withoutHistory = try await engine.search("and how does it work?")
        let withHistory = try await engine.search("and how does it work?", history: ["Tell me about the cache"])
        #expect(withHistory.first?.chunk.sectionTitle == "Cache")
        #expect(withoutHistory.count == withHistory.count)
    }

    @Test func rerankerReordersAndInvalidOutputIsIgnored() async throws {
        let plain = RAGEngine(configuration: configuration())
        let reranked = RAGEngine(configuration: configuration(reranker: ReversingReranker()))
        let broken = RAGEngine(configuration: configuration(reranker: BrokenReranker()))
        for engine in [plain, reranked, broken] {
            _ = try await engine.addDocument(DocumentIngestor.ingestText(zorp, title: "ZORP", id: "zorp"))
        }
        let plainContext = try await plain.context(for: "lists encoded")
        let rerankedContext = try await reranked.context(for: "lists encoded")
        let brokenContext = try await broken.context(for: "lists encoded")
        #expect(rerankedContext.sections.map(\.title) == plainContext.sections.map(\.title).reversed())
        #expect(brokenContext.sections.map(\.title) == plainContext.sections.map(\.title))
    }

    @Test func tokenBudgetOverride() async throws {
        let engine = RAGEngine(configuration: configuration())
        _ = try await engine.addDocument(DocumentIngestor.ingestText(zorp, title: "ZORP", id: "zorp"))
        #expect(try await engine.context(for: "lists", tokenBudget: 20).tokenCount <= 20)
    }

    #if os(macOS)
    @Test func addsAPDFFromDisk() async throws {
        let pdf = FixturePDF.make(pages: [[.heading("Integers"), .body("Integers are written as i<n>.")],
                                          [.heading("Lists"), .body("Lists are written as l<items>e.")]])
        let url = FileManager.default.temporaryDirectory.appending(path: "ondevice-rag-\(UUID().uuidString).pdf")
        #expect(pdf.write(to: url))
        defer { try? FileManager.default.removeItem(at: url) }
        let engine = RAGEngine(configuration: configuration())
        let info = try await engine.addDocument(pdfAt: url, title: "ZORP")
        #expect(info.sectionSource == .headings)
        #expect(try await engine.context(for: "lists").sections.first?.title == "Lists")
    }
    #endif
}
```

Add the 40-task check to `BGESmallEmbedderTests` (inside the gated suite). It uses the research corpus in this repo and is skipped once the package lives elsewhere:

```swift
    @Test(.enabled(if: FileManager.default.fileExists(atPath: FortyTasks.corpusURL.path), "research corpus not in this checkout"))
    func fortyTaskRetrievalCheck() async throws {
        var configuration = RAGConfiguration(embedder: try await BGESmallEmbedder(modelURL: ModelLocation.url))
        configuration.tokenCounter = EstimatingTokenCounter()
        let engine = RAGEngine(configuration: configuration)
        let corpus = try FortyTasks.corpus()
        for manual in corpus.manuals {
            _ = try await engine.addDocument(DocumentIngestor.ingestText(manual.content, title: manual.title, id: manual.id))
        }
        let gold = Dictionary(uniqueKeysWithValues: corpus.sections.map { ($0.task_id, $0.title) })
        for variant in ["main", "stripped"] {
            var hitAt5 = 0, sectionTop3 = 0
            let queries = try FortyTasks.queries().filter { $0.variant == variant }
            for query in queries {
                let hits = try await engine.search(query.query)
                if hits.prefix(5).contains(where: { $0.chunk.sectionTitle == gold[query.task_id] }) { hitAt5 += 1 }
                var titles: [String] = []
                for hit in hits where !titles.contains(hit.chunk.sectionTitle) {
                    titles.append(hit.chunk.sectionTitle)
                    if titles.count == 3 { break }
                }
                if titles.contains(gold[query.task_id]!) { sectionTop3 += 1 }
            }
            print("40-task check (\(variant)): chunk hit@5 \(hitAt5)/40, section top-3 \(sectionTop3)/40")
            // Spike: dense-only 40/40, BM25-only 38–40/40. Hybrid should not be worse than BM25 alone.
            #expect(sectionTop3 >= 38, "\(variant): section top-3 \(sectionTop3)/40")
        }
    }
```

and, at file scope in the same file:

```swift
enum FortyTasks {
    struct Manual: Decodable { let id: String; let title: String; let content: String }
    struct SectionEntry: Decodable { let title: String; let task_id: String }
    struct Corpus: Decodable { let manuals: [Manual]; let sections: [SectionEntry] }
    struct Query: Decodable { let task_id: String; let variant: String; let query: String }

    /// Packages/OnDeviceRAG/Tests/OnDeviceRAGTests/<this file> → repo root → evaluation/rag
    static let ragFolder = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "evaluation/rag")
    static var corpusURL: URL { ragFolder.appending(path: "corpus.json") }

    static func corpus() throws -> Corpus { try JSONDecoder().decode(Corpus.self, from: Data(contentsOf: corpusURL)) }
    static func queries() throws -> [Query] {
        try JSONDecoder().decode([Query].self, from: Data(contentsOf: ragFolder.appending(path: "queries.json")))
    }
}
```

- [ ] **Step 2: Run — expect compile failure** (`cannot find 'RAGEngine'`).

- [ ] **Step 3: Write `Engine/RAGConfiguration.swift`**

```swift
/// Optional second pass that re-orders the top candidates (e.g. with a language model).
/// Must return a reordering of exactly the chunks it was given; anything else is ignored.
public protocol Reranker: Sendable {
    func rerank(_ question: String, _ candidates: [RetrievedChunk]) async throws -> [RetrievedChunk]
}

/// Every tunable setting and pluggable component. Defaults follow design.md §8.
public struct RAGConfiguration: Sendable {
    public var embedder: any Embedder
    public var tokenCounter: any TokenCounter = AppleTokenCounter()
    /// Off by default; milestone 3's benchmark decides whether that changes.
    public var reranker: (any Reranker)? = nil
    public var rerankCount = 10
    public var ingest = IngestSettings()
    public var retrieval = RetrievalSettings()
    public var bm25K1 = 1.2
    public var bm25B = 0.75
    public var tokenBudget = 2000

    public init(embedder: any Embedder) {
        self.embedder = embedder
    }
}
```

- [ ] **Step 4: Write `Engine/RAGEngine.swift`**

```swift
import Foundation

/// On-device retrieval for any language model: add documents, then ask for the context that
/// best answers a question, packed to fit a token budget.
public actor RAGEngine {
    private let configuration: RAGConfiguration
    private var index: InMemoryIndex

    public init(configuration: RAGConfiguration) {
        self.configuration = configuration
        index = InMemoryIndex(dimension: configuration.embedder.dimension, k1: configuration.bm25K1, b: configuration.bm25B)
    }

    public var documents: [DocumentInfo] { index.documents.map(\.info) }

    public func addDocument(pdfAt url: URL, title: String? = nil) async throws -> DocumentInfo {
        try await addDocument(DocumentIngestor.ingestPDF(at: url, title: title, settings: configuration.ingest))
    }

    /// Adds (or replaces, same ID) an already-ingested document; embeds its chunks.
    public func addDocument(_ document: IngestedDocument) async throws -> DocumentInfo {
        let vectors = try await configuration.embedder.embed(document.chunks.map(\.indexedText), as: .passage)
        index.add(document, vectors: vectors)
        return document.info
    }

    public func removeDocument(id: DocumentID) {
        index.remove(id)
    }

    /// Best chunks first (hybrid dense + BM25). `history`: earlier chat messages, oldest first; optional.
    public func search(_ question: String, history: [String] = [], limit: Int? = nil) async throws -> [RetrievedChunk] {
        try await Retriever.retrieve(question: question, history: history, index: index, embedder: configuration.embedder,
                                     settings: configuration.retrieval, limit: limit ?? configuration.retrieval.candidateCount)
    }

    /// Text for the prompt: whole sections (or windows) in rank order, within the token budget.
    public func context(for question: String, history: [String] = [], tokenBudget: Int? = nil) async throws -> RAGContext {
        var candidates = try await search(question, history: history)
        if let reranker = configuration.reranker {
            candidates = await rerank(question, candidates, with: reranker)
        }
        return await ContextPacker.pack(candidates, index: index, tokenBudget: tokenBudget ?? configuration.tokenBudget,
                                        counter: configuration.tokenCounter)
    }

    private func rerank(_ question: String, _ candidates: [RetrievedChunk], with reranker: any Reranker) async -> [RetrievedChunk] {
        let head = Array(candidates.prefix(configuration.rerankCount))
        guard let reordered = try? await reranker.rerank(question, head),
              reordered.count == head.count,
              Set(reordered.map(\.chunk.id)) == Set(head.map(\.chunk.id)) else {
            return candidates
        }
        return reordered + candidates.dropFirst(head.count)
    }
}
```

- [ ] **Step 5: Run all tests — expect PASS**: `swift test --package-path Packages/OnDeviceRAG`. Without the model, the gated suite shows as skipped. With it, record the printed 40-task numbers.
  - If `historyIsOptionalAndUsed` fails because the fake embedder doesn't pull "cache" to the top, check the BM25 order for `"and how does it work?\nTell me about the cache"` first; the history must reach the query text (the unit test in Task 7 covers the joining).
  - If the 40-task check is below 38/40, **don't tune any setting** (they're frozen). Report the per-task misses: which task, the gold section and the top 3 sections retrieved. The researcher decides.

- [ ] **Step 6: Build for iOS too** (catches iOS-only compile errors such as the `UIFont` branch in `PDFTextExtractor`):

Run: `xcodebuild -scheme OnDeviceRAG -destination 'generic/platform=iOS' -quiet build` from `Packages/OnDeviceRAG`.
Expected: `** BUILD SUCCEEDED **` (or no error output with `-quiet`).

- [ ] **Step 7: Commit (when approved)** — `git commit -m "OnDeviceRAG: RAGEngine, configuration and reranker hook"`

---

### Task 10: Milestone wrap-up

**Files:**
- Create: `Packages/OnDeviceRAG/README.md`
- Modify: `Packages/OnDeviceRAG/Docs/design.md` (status line only)

- [ ] **Step 1: Write a short `README.md`**: what the package does (one paragraph); requirements (iOS 27 / macOS 27, the 64 MB bge-small Core AI model exported with `Tools/export_embedder.py`); a usage example; how to run the tests (`ONDEVICE_RAG_MODEL`); limitations (English-only embedder, PDF + text only, no OCR yet, index in memory until milestone 2, no answer generation, model-agnostic: the returned text goes to any model). Usage example:

```swift
import OnDeviceRAG

let embedder = try await BGESmallEmbedder(modelURL: Bundle.main.url(forResource: "bge_small_buckets_float16", withExtension: "aimodel")!)
let engine = RAGEngine(configuration: RAGConfiguration(embedder: embedder))
_ = try await engine.addDocument(pdfAt: pdfURL)
let context = try await engine.context(for: "How do I encode a list?")
let prompt = "Reference material:\n\(context.text)\n\nQuestion: How do I encode a list?"
```

- [ ] **Step 2: Full verification**: `swift test --package-path Packages/OnDeviceRAG` with the model present (every test passes, nothing skipped except where the corpus is missing), plus the iOS build from Task 9 Step 6. Confirm the app is untouched: `git diff --stat main -- "Prototype On-Device"` prints nothing.

- [ ] **Step 3: Set the design status line** to `_Status: milestone 1 implemented and Mac-tested (<date>); milestones 2–5 pending._`, but only if Step 2 passed.

- [ ] **Step 4: Commit (when approved)** — `git commit -m "OnDeviceRAG: milestone 1 README and status"`. Opening the PR to `main` waits until the researcher asks (milestones 1–4 share `feature/ondevice-rag`).

---

## Self-review notes

- **Spec coverage:** §3 components → Tasks 3–9; §4 API → Task 9 (with deviations 3 and 5); §5 ingest → Tasks 3–5 (OCR deferred, deviation 2; math flag in Task 5); §6 embedder → Task 6, index/retriever → Task 7, reranker → protocol + hook in Task 9 (implementation milestone 3); §7 packer + token counting → Task 8; §8 defaults → Global Constraints + `RAGConfiguration`/`IngestSettings`/`RetrievalSettings`; §9 unit tests → every task, including tokenizer parity and embedder parity (Task 6). Index save/load and embedder-mismatch rejection are milestone 2 (persistence), so they're not here.
- **Not in milestone 1:** persistence, `rag-index` CLI, prebuilt indexes, app integration, OCR, FoundationModels reranker, the harder benchmark, on-device (iPhone) verification.
