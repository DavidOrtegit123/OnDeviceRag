# OnDeviceRAG — Milestone 2a Implementation Plan (persistence, prebuilt indexes, `rag-index`)

> **Historical note:** written while this package lived inside a research app's repository (`Packages/OnDeviceRAG/`). Paths such as `Packages/OnDeviceRAG/…`, `evaluation/…` and `Prototype On-Device/…` refer to that repository; the 40-task retrieval check it mentions now lives there too.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Project rule (AGENTS.md → Mandatory Pre-Coding Protocol):** before each task's first code edit, post a short What / Why / How / Alternatives pitch and wait for the researcher's approval. Commit only when the researcher says so. Commit messages carry **no AI attribution lines**.

**Goal:** Documents survive app restarts without re-embedding, apps can ship ready-made indexes, and a Mac command-line tool builds and queries them.

**Architecture:** Each stored document is one JSON record (`DocumentRecord`): info, sections, chunks, its *source* (extracted lines or text, so it can be re-chunked without the original file) and its vectors as Float16 bytes. A `RAGEngine` opened on a storage directory keeps one `<id>.ragdoc` file per document and rebuilds records whose chunk settings or embedder changed. A shipped index is one `.ragindex` file holding many records, loaded read-only. `rag-index` is a thin executable over the same engine API, so prebuilt and on-device results are identical.

**Tech Stack:** Swift 6.4 tools, Swift 6 mode, Swift Testing, Foundation (`JSONEncoder`, `FileManager`), `Float16`.

**Spec:** `Packages/OnDeviceRAG/Docs/design.md` (§4 API, §6 Index, §11 milestone 2). Milestone 1 plan: `Docs/milestone-1-plan.md`.

## Global Constraints

- Everything from milestone 1's Global Constraints still applies (iOS 27 / macOS 27, tools 6.4, no network, only dependency `swift-transformers` 1.3.4, the app is **not** modified, frozen defaults, model-gated tests, `swift test --package-path Packages/OnDeviceRAG`).
- Milestone-1 public API keeps working: `RAGEngine(configuration:)` stays an in-memory engine. One intentional change: `removeDocument(id:)` becomes `throws` (prebuilt documents can't be removed).
- On-disk format version **1**. Vectors: Float16, little-endian (iOS and macOS 27 are Apple silicon only), `chunks.count × dimension` values in chunk order.
- File names: user documents `<folder-safe id>.ragdoc`; shipped indexes `*.ragindex`. Folder-safe id = the id with every character outside `A–Z a–z 0–9 - _` percent-encoded.
- A document that can't be read never stops the engine from opening: it's skipped and described in `RAGEngine.loadIssues`.
- Prebuilt indexes are **never written to** (they live in an app bundle).

## Deviations from design.md (applied to design.md in Task 6)

1. §6 described four files per document in a folder (`manifest.json`, `pages.json`, `chunks.json`, `vectors.f16`). Instead: **one file per document** (atomic writes with `Data.write(options: .atomic)`; no half-written folders) and **one file per shipped index**. Xcode flattens resource folders into the app bundle (seen with the Qwen model, Architecture Rule #8), so a folder-of-folders index would collide; a single file can't be flattened. Vectors are still Float16, stored as base64 inside the JSON (~1 MB for a 1,000-chunk textbook).
2. Instead of `pages.json`, the record keeps the document's **source**: the extracted lines with font facts and the outline (PDF) or the original text. That's what re-chunking needs, and it removes the need for the original PDF.
3. API: `init(configuration:storageDirectory:) async throws`, `addPrebuiltIndex(at:)` takes a `.ragindex` **file**, plus `exportIndex(to:)`, `isPrebuilt(_:)`, `loadIssues`.

## File Structure

```
Packages/OnDeviceRAG/
├── Package.swift                               + executable target rag-index
├── Sources/OnDeviceRAG/
│   ├── Ingest/TextLine.swift                   TextLine, OutlineEntry become Codable
│   ├── Ingest/DocumentIngestor.swift           DocumentSource, IngestedDocument.source, reingest
│   ├── Persistence/DocumentRecord.swift        DocumentRecord, IndexFile, Float16 coding
│   ├── Persistence/DocumentStore.swift         .ragdoc save/load/delete; .ragindex read/write
│   ├── Search/InMemoryIndex.swift              + vectors(of:)
│   └── Engine/RAGEngine.swift                  storage, rebuild rules, prebuilt, export, RAGError
├── Sources/rag-index/main.swift                build / query commands
└── Tests/OnDeviceRAGTests/
    ├── Support/CountingEmbedder.swift          counts passage embeddings (proves "no re-embedding")
    ├── SourceTests.swift
    ├── PersistenceTests.swift
    ├── EnginePersistenceTests.swift
    ├── PrebuiltIndexTests.swift
    └── EngineTests.swift                       one line: `try await engine.removeDocument`
```

---

### Task 1: Keep each document's source and re-chunk from it

**Files:**
- Modify: `Sources/OnDeviceRAG/Ingest/TextLine.swift` (add `Codable` to `TextLine` and `OutlineEntry`)
- Modify: `Sources/OnDeviceRAG/Ingest/DocumentIngestor.swift`
- Test: `Tests/OnDeviceRAGTests/SourceTests.swift`

**Interfaces:**
- Produces: `enum DocumentSource: Sendable, Hashable, Codable { case pdf(lines: [TextLine], outline: [OutlineEntry], pageCount: Int, pagesWithoutText: [Int]); case text(String) }`
- Produces: `IngestedDocument` gains `let source: DocumentSource` (internal) and an internal memberwise init `IngestedDocument(info:sections:chunks:source:)`.
- Produces: `IngestSettings` becomes `Codable`.
- Produces: `DocumentIngestor.reingest(_ source: DocumentSource, title: String, id: DocumentID, settings: IngestSettings) -> IngestedDocument`
- Produces (internal): `DocumentIngestor.ingest(lines:outline:pageCount:pagesWithoutText:title:id:settings:) -> IngestedDocument`

- [ ] **Step 1: Write the failing tests** — `SourceTests.swift`:

```swift
import Foundation
import Testing
@testable import OnDeviceRAG

struct SourceTests {
    private let longText = "## Rules\n" + (1...12).map { "Rule \($0) says the encoder writes every value as plain text with a fixed tag." }.joined(separator: " ")

    @Test func textSourceIsKeptAndReingestsIdentically() {
        let original = DocumentIngestor.ingestText(longText, title: "Doc", id: "d")
        #expect(original.source == .text(longText))
        let again = DocumentIngestor.reingest(original.source, title: "Doc", id: "d", settings: IngestSettings())
        #expect(again.sections == original.sections)
        #expect(again.chunks == original.chunks)
    }

    @Test func reingestWithSmallerChunksRechunks() {
        let original = DocumentIngestor.ingestText(longText, title: "Doc", id: "d")
        var settings = IngestSettings()
        settings.chunkLength = 100
        let smaller = DocumentIngestor.reingest(original.source, title: "Doc", id: "d", settings: settings)
        #expect(smaller.chunks.count > original.chunks.count)
        #expect(smaller.chunks.allSatisfy { $0.text.count <= 100 })
    }

    @Test func sourceRoundTripsThroughJSON() throws {
        let source = DocumentSource.pdf(lines: [TextLine("Title", page: 1, fontSize: 18, isBold: true), TextLine("x = 1", isMonospaced: true)],
                                        outline: [OutlineEntry(title: "Title", page: 1)], pageCount: 2, pagesWithoutText: [2])
        let decoded = try JSONDecoder().decode(DocumentSource.self, from: JSONEncoder().encode(source))
        #expect(decoded == source)
    }

    #if os(macOS)
    @Test func pdfSourceReingestsIdenticallyWithoutTheFile() {
        let pdf = FixturePDF.make(pages: [[.heading("Integers"), .body("Integers are written as i<n>.")],
                                          [.heading("Lists"), .body("Lists are written as l<items>e."), .code("encode([1])")]])
        let original = DocumentIngestor.ingest(pdf: pdf, title: "ZORP", id: "z", settings: IngestSettings())
        guard case .pdf = original.source else { Issue.record("expected a PDF source"); return }
        let again = DocumentIngestor.reingest(original.source, title: "ZORP", id: "z", settings: IngestSettings())
        #expect(again.info == original.info)
        #expect(again.sections == original.sections)
        #expect(again.chunks == original.chunks)
    }
    #endif
}
```

- [ ] **Step 2: Run — expect compile failure** (`cannot find 'DocumentSource'`).

- [ ] **Step 3: Implement.** In `TextLine.swift`: `struct TextLine: Sendable, Hashable, Codable` and `struct OutlineEntry: Sendable, Hashable, Codable`. In `DocumentIngestor.swift`:

```swift
public struct IngestSettings: Sendable, Hashable, Codable {
    public var chunkLength = 400
    public var sectionWindowLength = 1500
    public init() {}
}

/// What a document was built from — kept so it can be re-chunked later without the original file.
enum DocumentSource: Sendable, Hashable, Codable {
    case pdf(lines: [TextLine], outline: [OutlineEntry], pageCount: Int, pagesWithoutText: [Int])
    case text(String)
}

public struct IngestedDocument: Sendable {
    public let info: DocumentInfo
    public let sections: [Section]
    public let chunks: [Chunk]
    let source: DocumentSource
}
```

Then, inside `DocumentIngestor`:
- `ingestText` passes `source: .text(text)` to `assemble`.
- `ingest(pdf:title:id:settings:)` becomes:

```swift
    static func ingest(pdf document: PDFDocument, title: String, id: DocumentID, settings: IngestSettings) -> IngestedDocument {
        let extracted = PDFTextExtractor.lines(from: document)
        return ingest(lines: extracted.lines, outline: PDFTextExtractor.outline(of: document), pageCount: document.pageCount,
                      pagesWithoutText: extracted.pagesWithoutText, title: title, id: id, settings: settings)
    }

    static func ingest(lines: [TextLine], outline: [OutlineEntry], pageCount: Int, pagesWithoutText: [Int],
                       title: String, id: DocumentID, settings: IngestSettings) -> IngestedDocument {
        let detected = SectionDetector.detect(lines: lines, outline: outline, documentTitle: title,
                                              windowLength: settings.sectionWindowLength)
        return assemble(sections: detected.sections, source: detected.source, title: title, id: id,
                        pageCount: pageCount, pagesWithoutText: pagesWithoutText,
                        mathHeavyPages: mathHeavyPages(lines), settings: settings,
                        documentSource: .pdf(lines: lines, outline: outline, pageCount: pageCount, pagesWithoutText: pagesWithoutText))
    }

    /// Rebuilds sections and chunks from a stored source (e.g. after chunk settings changed).
    static func reingest(_ source: DocumentSource, title: String, id: DocumentID, settings: IngestSettings) -> IngestedDocument {
        switch source {
        case let .pdf(lines, outline, pageCount, pagesWithoutText):
            ingest(lines: lines, outline: outline, pageCount: pageCount, pagesWithoutText: pagesWithoutText,
                   title: title, id: id, settings: settings)
        case let .text(text):
            ingestText(text, title: title, id: id, settings: settings)
        }
    }
```
- `assemble` gains a final parameter `documentSource: DocumentSource` and returns `IngestedDocument(info: info, sections: sections, chunks: chunks, source: documentSource)`. (The existing `source: SectionSource` parameter keeps its name.)

- [ ] **Step 4: Run all tests — expect PASS** (48 tests).
- [ ] **Step 5: Commit (when approved)** — `git commit -m "OnDeviceRAG: keep each document's source and re-chunk from it"`

---

### Task 2: Document records and the on-disk formats

**Files:**
- Create: `Sources/OnDeviceRAG/Persistence/DocumentRecord.swift`
- Create: `Sources/OnDeviceRAG/Persistence/DocumentStore.swift`
- Test: `Tests/OnDeviceRAGTests/PersistenceTests.swift`

**Interfaces:**
- Consumes: `IngestedDocument`, `DocumentSource`, `IngestSettings` (Task 1).
- Produces:

```swift
struct DocumentRecord: Codable, Sendable {
    static let currentFormatVersion = 1
    var formatVersion: Int
    var embedder: String
    var dimension: Int
    var settings: IngestSettings
    var addedAt: Date
    var info: DocumentInfo
    var sections: [Section]
    var chunks: [Chunk]
    var source: DocumentSource
    var vectors: Data
    init(document: IngestedDocument, vectors: [[Float]], embedder: String, dimension: Int, settings: IngestSettings, addedAt: Date)
    var document: IngestedDocument { get }
    func decodedVectors() -> [[Float]]?          // nil if size doesn't match chunks × dimension
}
struct IndexFile: Codable, Sendable { var formatVersion: Int; var documents: [DocumentRecord] }
enum DocumentStore {
    static let documentExtension = "ragdoc"
    static func fileURL(for id: DocumentID, in directory: URL) -> URL
    static func save(_ record: DocumentRecord, in directory: URL) throws
    static func delete(_ id: DocumentID, in directory: URL) throws
    static func loadAll(in directory: URL) -> (records: [DocumentRecord], issues: [String])
    static func writeIndex(_ records: [DocumentRecord], to url: URL) throws
    static func readIndex(at url: URL) throws -> [DocumentRecord]
}
public enum RAGError: Error, Equatable {
    case prebuiltDocument(DocumentID)
    case unsupportedFormat(version: Int, file: String)
}
```

- [ ] **Step 1: Write the failing tests** — `PersistenceTests.swift`:

```swift
import Foundation
import Testing
@testable import OnDeviceRAG

struct PersistenceTests {
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: "ondevice-rag-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func record(_ id: String, addedAt: Date = Date()) async throws -> DocumentRecord {
        let document = DocumentIngestor.ingestText("## A\nAlpha text.\n\n## B\nBeta text.", title: "Doc \(id)", id: id)
        let vectors = try await FakeEmbedder().embed(document.chunks.map(\.indexedText), as: .passage)
        return DocumentRecord(document: document, vectors: vectors, embedder: "fake", dimension: 64,
                              settings: IngestSettings(), addedAt: addedAt)
    }

    @Test func vectorsSurviveFloat16() async throws {
        let original = try await FakeEmbedder().embed(["alpha beta gamma", "delta"], as: .passage)
        let document = DocumentIngestor.ingestText("## A\nOne.\n\n## B\nTwo.", title: "D", id: "d")
        let stored = DocumentRecord(document: document, vectors: original, embedder: "fake", dimension: 64,
                                    settings: IngestSettings(), addedAt: Date())
        let decoded = try #require(stored.decodedVectors())
        #expect(stored.vectors.count == 2 * 64 * 2)
        // Cosine, not a raw dot product: Float16 rounding can shorten a vector slightly (~0.03%)
        // without changing its direction. (Real bge vectors are Float16 already, so lose nothing.)
        for (a, b) in zip(original, decoded) {
            let dot = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            let normB = b.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
            #expect(dot / normB >= 0.9999)
        }
    }

    @Test func wrongSizedVectorsDecodeToNil() async throws {
        var stored = try await record("d")
        stored.vectors = stored.vectors.dropLast(2)
        #expect(stored.decodedVectors() == nil)
    }

    @Test func saveLoadDeleteRoundTrip() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let older = try await record("b/odd id", addedAt: Date(timeIntervalSince1970: 100))
        let newer = try await record("a", addedAt: Date(timeIntervalSince1970: 200))
        try DocumentStore.save(newer, in: directory)
        try DocumentStore.save(older, in: directory)
        #expect(DocumentStore.fileURL(for: "b/odd id", in: directory).lastPathComponent == "b%2Fodd%20id.ragdoc")

        let loaded = DocumentStore.loadAll(in: directory)
        #expect(loaded.issues.isEmpty)
        #expect(loaded.records.map(\.info.id) == ["b/odd id", "a"]) // oldest first
        #expect(loaded.records[0].document.chunks == older.document.chunks)
        #expect(loaded.records[0].document.source == older.document.source)

        try DocumentStore.delete("b/odd id", in: directory)
        try DocumentStore.delete("missing", in: directory) // no error
        #expect(DocumentStore.loadAll(in: directory).records.map(\.info.id) == ["a"])
    }

    @Test func unreadableFilesBecomeIssuesNotErrors() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try DocumentStore.save(try await record("good"), in: directory)
        try Data("not json".utf8).write(to: directory.appending(path: "garbage.ragdoc"))
        try Data(#"{"formatVersion": 99}"#.utf8).write(to: directory.appending(path: "future.ragdoc"))
        let loaded = DocumentStore.loadAll(in: directory)
        #expect(loaded.records.map(\.info.id) == ["good"])
        #expect(loaded.issues.count == 2)
        #expect(loaded.issues.contains { $0.contains("future.ragdoc") && $0.contains("99") })
    }

    @Test func indexFileRoundTripAndVersionCheck() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(path: "shipped.ragindex")
        try DocumentStore.writeIndex([try await record("x"), try await record("y")], to: url)
        #expect(try DocumentStore.readIndex(at: url).map(\.info.id) == ["x", "y"])

        let future = directory.appending(path: "future.ragindex")
        try Data(#"{"formatVersion": 2, "documents": []}"#.utf8).write(to: future)
        #expect(throws: RAGError.unsupportedFormat(version: 2, file: "future.ragindex")) { try DocumentStore.readIndex(at: future) }
    }
}
```

- [ ] **Step 2: Run — expect compile failure** (`cannot find 'DocumentRecord'`).

- [ ] **Step 3: Write `Persistence/DocumentRecord.swift`**

```swift
import Foundation

public enum RAGError: Error, Equatable {
    /// Prebuilt (shipped) documents are read-only.
    case prebuiltDocument(DocumentID)
    case unsupportedFormat(version: Int, file: String)
}

/// One document as stored on disk: everything needed to search it, plus its source for re-chunking.
struct DocumentRecord: Codable, Sendable {
    static let currentFormatVersion = 1

    var formatVersion: Int
    var embedder: String
    var dimension: Int
    var settings: IngestSettings
    var addedAt: Date
    var info: DocumentInfo
    var sections: [Section]
    var chunks: [Chunk]
    var source: DocumentSource
    /// Float16, little-endian, `chunks.count × dimension` values in chunk order (base64 in JSON).
    var vectors: Data

    init(document: IngestedDocument, vectors: [[Float]], embedder: String, dimension: Int, settings: IngestSettings, addedAt: Date) {
        formatVersion = Self.currentFormatVersion
        self.embedder = embedder
        self.dimension = dimension
        self.settings = settings
        self.addedAt = addedAt
        info = document.info
        sections = document.sections
        chunks = document.chunks
        source = document.source
        self.vectors = Self.encode(vectors)
    }

    var document: IngestedDocument {
        IngestedDocument(info: info, sections: sections, chunks: chunks, source: source)
    }

    func decodedVectors() -> [[Float]]? {
        let scalarCount = chunks.count * dimension
        guard dimension > 0, vectors.count == scalarCount * MemoryLayout<Float16>.size else { return nil }
        let halves: [Float16] = vectors.withUnsafeBytes { raw in
            (0..<scalarCount).map { raw.loadUnaligned(fromByteOffset: $0 * MemoryLayout<Float16>.size, as: Float16.self) }
        }
        return stride(from: 0, to: scalarCount, by: dimension).map { start in
            halves[start..<(start + dimension)].map(Float.init)
        }
    }

    static func encode(_ vectors: [[Float]]) -> Data {
        let halves = vectors.flatMap { $0 }.map(Float16.init)
        return halves.withUnsafeBytes { Data($0) }
    }
}

/// A shipped, read-only bundle of documents (what `rag-index build` writes).
struct IndexFile: Codable, Sendable {
    var formatVersion: Int
    var documents: [DocumentRecord]
}

/// Reads only the version number, so a newer format is reported instead of failing to decode.
struct FormatProbe: Decodable {
    var formatVersion: Int
}
```

- [ ] **Step 4: Write `Persistence/DocumentStore.swift`**

```swift
import Foundation

/// File I/O for document records: one `.ragdoc` per user document in a storage directory,
/// and single-file `.ragindex` bundles for shipped documents.
enum DocumentStore {
    static let documentExtension = "ragdoc"

    static func fileURL(for id: DocumentID, in directory: URL) -> URL {
        let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
        let name = id.addingPercentEncoding(withAllowedCharacters: allowed) ?? UUID().uuidString
        return directory.appending(path: "\(name).\(documentExtension)")
    }

    /// Atomic: a crash mid-write leaves the previous file intact.
    static func save(_ record: DocumentRecord, in directory: URL) throws {
        try encoder().encode(record).write(to: fileURL(for: record.info.id, in: directory), options: .atomic)
    }

    static func delete(_ id: DocumentID, in directory: URL) throws {
        let url = fileURL(for: id, in: directory)
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    /// Every readable record, oldest first; unreadable files are described in `issues`.
    static func loadAll(in directory: URL) -> (records: [DocumentRecord], issues: [String]) {
        let files = ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == documentExtension }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var records: [DocumentRecord] = []
        var issues: [String] = []
        for file in files {
            do {
                let data = try Data(contentsOf: file)
                let version = try JSONDecoder().decode(FormatProbe.self, from: data).formatVersion
                guard version == DocumentRecord.currentFormatVersion else {
                    throw RAGError.unsupportedFormat(version: version, file: file.lastPathComponent)
                }
                records.append(try JSONDecoder().decode(DocumentRecord.self, from: data))
            } catch RAGError.unsupportedFormat(let version, let name) {
                issues.append("\(name): unsupported format version \(version) (this package reads version \(DocumentRecord.currentFormatVersion))")
            } catch {
                issues.append("\(file.lastPathComponent): unreadable (\(error.localizedDescription))")
            }
        }
        records.sort { $0.addedAt != $1.addedAt ? $0.addedAt < $1.addedAt : $0.info.id < $1.info.id }
        return (records, issues)
    }

    static func writeIndex(_ records: [DocumentRecord], to url: URL) throws {
        let file = IndexFile(formatVersion: DocumentRecord.currentFormatVersion, documents: records)
        try encoder().encode(file).write(to: url, options: .atomic)
    }

    static func readIndex(at url: URL) throws -> [DocumentRecord] {
        let data = try Data(contentsOf: url)
        let version = try JSONDecoder().decode(FormatProbe.self, from: data).formatVersion
        guard version == DocumentRecord.currentFormatVersion else {
            throw RAGError.unsupportedFormat(version: version, file: url.lastPathComponent)
        }
        return try JSONDecoder().decode(IndexFile.self, from: data).documents
    }

    /// Sorted keys so the same input always produces the same bytes (reproducible prebuilt indexes).
    private static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}
```

- [ ] **Step 5: Run all tests — expect PASS.** If `saveLoadDeleteRoundTrip`'s file-name expectation fails, print the actual name: `addingPercentEncoding` must encode `/` as `%2F` and the space as `%20`.
- [ ] **Step 6: Commit (when approved)** — `git commit -m "OnDeviceRAG: document records and on-disk formats"`

---

### Task 3: A persistent `RAGEngine`

**Files:**
- Modify: `Sources/OnDeviceRAG/Engine/RAGEngine.swift`
- Modify: `Sources/OnDeviceRAG/Search/InMemoryIndex.swift` (add `vectors(of:)`)
- Modify: `Tests/OnDeviceRAGTests/EngineTests.swift` (`await engine.removeDocument(id: "zorp")` → `try await engine.removeDocument(id: "zorp")`)
- Create: `Tests/OnDeviceRAGTests/Support/CountingEmbedder.swift`
- Test: `Tests/OnDeviceRAGTests/EnginePersistenceTests.swift`

**Interfaces:**
- Consumes: `DocumentRecord`, `DocumentStore`, `RAGError` (Task 2); `DocumentIngestor.reingest` (Task 1).
- Produces: `public init(configuration: RAGConfiguration, storageDirectory: URL) async throws`; `public private(set) var loadIssues: [String]`; `public func removeDocument(id: DocumentID) throws`.
- Produces: `InMemoryIndex.vectors(of id: DocumentID) -> [[Float]]`.
- Produces (internal, used by Task 4): `private func prepare(_ record: DocumentRecord) async throws -> (document: IngestedDocument, vectors: [[Float]], changed: Bool)`.

Rules when opening a stored record (in this order):
1. If its `settings` differ from `configuration.ingest` → re-chunk from its source (`reingest`), then re-embed.
2. Else if its `embedder` identifier or `dimension` differs, or its vectors don't decode → re-embed.
3. Else use it as stored (no embedding at all).
A rebuilt record is saved back (user storage only). `addDocument` embeds, then saves, then indexes — so a failed save leaves memory and disk consistent.

- [ ] **Step 1: Write `Support/CountingEmbedder.swift`**

```swift
@testable import OnDeviceRAG

actor EmbedCounter {
    private(set) var passages = 0
    func add(_ count: Int) { passages += count }
}

/// FakeEmbedder that counts how many passages it embedded and has a configurable identifier.
struct CountingEmbedder: Embedder {
    var identifier = "fake-bag-of-words"
    let dimension = 64
    let counter: EmbedCounter

    func embed(_ texts: [String], as kind: EmbeddingKind) async throws -> [[Float]] {
        if kind == .passage { await counter.add(texts.count) }
        return try await FakeEmbedder().embed(texts, as: kind)
    }
}
```

- [ ] **Step 2: Write the failing tests** — `EnginePersistenceTests.swift`:

```swift
import Foundation
import Testing
@testable import OnDeviceRAG

private let zorp = "## Integers\nIntegers are encoded as i<n>e.\n\n## Lists\nLists are encoded as l<items>e."
private let cache = "## Cache\nThe cache stores loaded values."

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appending(path: "ondevice-rag-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func engine(at directory: URL, identifier: String = "fake-bag-of-words", chunkLength: Int = 400,
                    counter: EmbedCounter) async throws -> RAGEngine {
    var configuration = RAGConfiguration(embedder: CountingEmbedder(identifier: identifier, counter: counter))
    configuration.tokenCounter = EstimatingTokenCounter()
    configuration.ingest.chunkLength = chunkLength
    return try await RAGEngine(configuration: configuration, storageDirectory: directory)
}

struct EnginePersistenceTests {
    @Test func reopenUsesStoredVectorsWithoutReembedding() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = try await engine(at: directory, counter: EmbedCounter())
        _ = try await first.addDocument(DocumentIngestor.ingestText(zorp, title: "ZORP", id: "zorp"))
        _ = try await first.addDocument(DocumentIngestor.ingestText(cache, title: "Cache SDK", id: "cache"))
        let before = try await first.search("how are lists encoded").map(\.chunk.id)

        let counter = EmbedCounter()
        let reopened = try await engine(at: directory, counter: counter)
        #expect(await reopened.documents.map(\.id) == ["zorp", "cache"])
        #expect(await counter.passages == 0)
        #expect(try await reopened.search("how are lists encoded").map(\.chunk.id) == before)
        #expect(await reopened.loadIssues.isEmpty)
    }

    @Test func changedEmbedderReembedsOnOpenAndSaves() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        _ = try await engine(at: directory, counter: EmbedCounter()).addDocument(DocumentIngestor.ingestText(zorp, title: "ZORP", id: "zorp"))

        let counter = EmbedCounter()
        _ = try await engine(at: directory, identifier: "other-model", counter: counter)
        #expect(await counter.passages == 2)

        let again = EmbedCounter()
        _ = try await engine(at: directory, identifier: "other-model", counter: again)
        #expect(await again.passages == 0) // the rebuilt record was saved
    }

    @Test func changedChunkSettingsRechunkFromSource() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let long = "## Rules\n" + (1...12).map { "Rule \($0) says the encoder writes every value as plain text with a fixed tag." }.joined(separator: " ")
        let original = try await engine(at: directory, counter: EmbedCounter()).addDocument(DocumentIngestor.ingestText(long, title: "R", id: "r"))

        let reopened = try await engine(at: directory, chunkLength: 100, counter: EmbedCounter())
        let info = try #require(await reopened.documents.first)
        #expect(info.chunkCount > original.chunkCount)
    }

    #if os(macOS)
    @Test func pdfIsRechunkedWithoutTheOriginalFile() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let pdfURL = directory.appending(path: "zorp.pdf")
        let body = (1...8).map { "Rule \($0) says the encoder writes every value as plain text." }
        _ = FixturePDF.make(pages: [[.heading("Integers")] + body.map { .body($0) }, [.heading("Lists"), .body("Lists are written as l<items>e.")]]).write(to: pdfURL)
        let original = try await engine(at: directory, counter: EmbedCounter()).addDocument(pdfAt: pdfURL, title: "ZORP")
        try FileManager.default.removeItem(at: pdfURL)

        let reopened = try await engine(at: directory, chunkLength: 120, counter: EmbedCounter())
        let info = try #require(await reopened.documents.first)
        #expect(info.sectionSource == .headings)
        #expect(info.chunkCount > original.chunkCount)
    }
    #endif

    @Test func removeDeletesTheFile() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = try await engine(at: directory, counter: EmbedCounter())
        _ = try await first.addDocument(DocumentIngestor.ingestText(zorp, title: "ZORP", id: "zorp"))
        try await first.removeDocument(id: "zorp")
        #expect(try await engine(at: directory, counter: EmbedCounter()).documents.isEmpty)
    }

    @Test func unreadableFileIsReportedAndOthersStillLoad() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        _ = try await engine(at: directory, counter: EmbedCounter()).addDocument(DocumentIngestor.ingestText(zorp, title: "ZORP", id: "zorp"))
        try Data("broken".utf8).write(to: directory.appending(path: "broken.ragdoc"))
        let reopened = try await engine(at: directory, counter: EmbedCounter())
        #expect(await reopened.documents.map(\.id) == ["zorp"])
        #expect(await reopened.loadIssues.count == 1)
    }

    @Test func storageDirectoryIsCreatedIfMissing() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "ondevice-rag-\(UUID().uuidString)/nested")
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let created = try await engine(at: directory, counter: EmbedCounter())
        _ = try await created.addDocument(DocumentIngestor.ingestText(cache, title: "Cache", id: "cache"))
        #expect(FileManager.default.fileExists(atPath: DocumentStore.fileURL(for: "cache", in: directory).path))
    }
}
```

- [ ] **Step 3: Run — expect compile failure** (`extra argument 'storageDirectory'`).

- [ ] **Step 4: Add to `InMemoryIndex`**

```swift
    /// The stored vectors of one document's chunks, in chunk order.
    func vectors(of id: DocumentID) -> [[Float]] {
        chunks.indices.filter { chunks[$0].documentID == id }.map { position in
            Array(vectors[(position * dimension)..<((position + 1) * dimension)])
        }
    }
```

- [ ] **Step 5: Rewrite `Engine/RAGEngine.swift`** (search, context and rerank are unchanged from milestone 1)

```swift
import Foundation

/// On-device retrieval for any language model: add documents, then ask for the context that
/// best answers a question, packed to fit a token budget. With a storage directory, documents
/// are saved and reloaded without re-embedding.
public actor RAGEngine {
    private let configuration: RAGConfiguration
    private let storageDirectory: URL?
    private var index: InMemoryIndex
    private var addedAt: [DocumentID: Date] = [:]
    private var prebuiltIDs: Set<DocumentID> = []

    /// Documents that couldn't be loaded (unreadable file, unsupported format, ID clash), one line each.
    public private(set) var loadIssues: [String] = []

    /// In-memory engine: nothing is saved.
    public init(configuration: RAGConfiguration) {
        self.init(configuration: configuration, storage: nil)
    }

    /// Persistent engine: loads every document saved in `storageDirectory` (created if missing),
    /// re-chunking or re-embedding those built with other settings or another embedder.
    public init(configuration: RAGConfiguration, storageDirectory: URL) async throws {
        self.init(configuration: configuration, storage: storageDirectory)
        try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
        try await loadStoredDocuments()
    }

    private init(configuration: RAGConfiguration, storage: URL?) {
        self.configuration = configuration
        storageDirectory = storage
        index = InMemoryIndex(dimension: configuration.embedder.dimension, k1: configuration.bm25K1, b: configuration.bm25B)
    }

    public var documents: [DocumentInfo] { index.documents.map(\.info) }

    public func addDocument(pdfAt url: URL, title: String? = nil) async throws -> DocumentInfo {
        try await addDocument(DocumentIngestor.ingestPDF(at: url, title: title, settings: configuration.ingest))
    }

    /// Adds (or replaces, same ID) a document; embeds its chunks and, with storage, saves it.
    public func addDocument(_ document: IngestedDocument) async throws -> DocumentInfo {
        guard !prebuiltIDs.contains(document.info.id) else { throw RAGError.prebuiltDocument(document.info.id) }
        let vectors = try await configuration.embedder.embed(document.chunks.map(\.indexedText), as: .passage)
        let date = addedAt[document.info.id] ?? Date()
        if let storageDirectory {
            try DocumentStore.save(record(document, vectors, date), in: storageDirectory)
        }
        index.add(document, vectors: vectors)
        addedAt[document.info.id] = date
        return document.info
    }

    public func removeDocument(id: DocumentID) throws {
        guard !prebuiltIDs.contains(id) else { throw RAGError.prebuiltDocument(id) }
        if let storageDirectory {
            try DocumentStore.delete(id, in: storageDirectory)
        }
        index.remove(id)
        addedAt[id] = nil
    }

    // search(_:history:limit:), context(for:history:tokenBudget:) and rerank(_:_:with:) — unchanged.

    // MARK: - Loading

    private func loadStoredDocuments() async throws {
        guard let storageDirectory else { return }
        let loaded = DocumentStore.loadAll(in: storageDirectory)
        loadIssues += loaded.issues
        for stored in loaded.records {
            let prepared = try await prepare(stored)
            if prepared.changed {
                try DocumentStore.save(record(prepared.document, prepared.vectors, stored.addedAt), in: storageDirectory)
            }
            index.add(prepared.document, vectors: prepared.vectors)
            addedAt[prepared.document.info.id] = stored.addedAt
        }
    }

    /// Brings a stored record up to the current chunk settings and embedder.
    private func prepare(_ stored: DocumentRecord) async throws -> (document: IngestedDocument, vectors: [[Float]], changed: Bool) {
        var document = stored.document
        var vectors = stored.decodedVectors()
        var changed = false
        if stored.settings != configuration.ingest {
            document = DocumentIngestor.reingest(stored.source, title: stored.info.title, id: stored.info.id, settings: configuration.ingest)
            vectors = nil
        }
        if stored.embedder != configuration.embedder.identifier || stored.dimension != configuration.embedder.dimension {
            vectors = nil
        }
        if vectors == nil {
            vectors = try await configuration.embedder.embed(document.chunks.map(\.indexedText), as: .passage)
            changed = true
        }
        return (document, vectors ?? [], changed)
    }

    private func record(_ document: IngestedDocument, _ vectors: [[Float]], _ date: Date) -> DocumentRecord {
        DocumentRecord(document: document, vectors: vectors, embedder: configuration.embedder.identifier,
                       dimension: configuration.embedder.dimension, settings: configuration.ingest, addedAt: date)
    }
}
```

Keep the milestone-1 `search`, `context` and `rerank` methods exactly as they are, in the place marked above.

- [ ] **Step 6: Update `EngineTests.addSearchContextRemove`**: `await engine.removeDocument(id: "zorp")` → `try await engine.removeDocument(id: "zorp")`.
- [ ] **Step 7: Run all tests — expect PASS.**
- [ ] **Step 8: Commit (when approved)** — `git commit -m "OnDeviceRAG: persistent engine with rebuild rules"`

---

### Task 4: Prebuilt (shipped) indexes

**Files:**
- Modify: `Sources/OnDeviceRAG/Engine/RAGEngine.swift`
- Test: `Tests/OnDeviceRAGTests/PrebuiltIndexTests.swift`

**Interfaces:**
- Consumes: `prepare(_:)`, `record(_:_:_:)`, `DocumentStore.writeIndex/readIndex`, `InMemoryIndex.vectors(of:)`.
- Produces: `public func exportIndex(to url: URL) throws`; `@discardableResult public func addPrebuiltIndex(at url: URL) async throws -> [DocumentInfo]`; `public func isPrebuilt(_ id: DocumentID) -> Bool`.

Rules: `exportIndex` writes every **non-prebuilt** document (in `documents` order) to one `.ragindex` file. `addPrebuiltIndex` loads each record through the same `prepare` rules, but **never writes** (rebuilt vectors stay in memory). A record whose ID is already in the engine is skipped and reported in `loadIssues`. Prebuilt documents can't be removed or replaced (`RAGError.prebuiltDocument`) and are never saved into the storage directory.

- [ ] **Step 1: Write the failing tests** — `PrebuiltIndexTests.swift`:

```swift
import Foundation
import Testing
@testable import OnDeviceRAG

private let zorp = "## Integers\nIntegers are encoded as i<n>e.\n\n## Lists\nLists are encoded as l<items>e."
private let cache = "## Cache\nThe cache stores loaded values."

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appending(path: "ondevice-rag-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func configuration(identifier: String = "fake-bag-of-words", counter: EmbedCounter = EmbedCounter()) -> RAGConfiguration {
    var configuration = RAGConfiguration(embedder: CountingEmbedder(identifier: identifier, counter: counter))
    configuration.tokenCounter = EstimatingTokenCounter()
    return configuration
}

/// Builds a .ragindex with the ZORP and cache documents; returns its URL.
private func buildIndex(in directory: URL) async throws -> URL {
    let builder = RAGEngine(configuration: configuration())
    _ = try await builder.addDocument(DocumentIngestor.ingestText(zorp, title: "ZORP", id: "zorp"))
    _ = try await builder.addDocument(DocumentIngestor.ingestText(cache, title: "Cache SDK", id: "cache"))
    let url = directory.appending(path: "shipped.ragindex")
    try await builder.exportIndex(to: url)
    return url
}

struct PrebuiltIndexTests {
    @Test func exportedIndexLoadsWithoutReembedding() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try await buildIndex(in: directory)

        let counter = EmbedCounter()
        let engine = RAGEngine(configuration: configuration(counter: counter))
        let infos = try await engine.addPrebuiltIndex(at: url)
        #expect(infos.map(\.id) == ["zorp", "cache"])
        #expect(await counter.passages == 0)
        #expect(await engine.isPrebuilt("zorp"))
        #expect(try await engine.search("how are lists encoded").first?.chunk.sectionTitle == "Lists")
    }

    @Test func prebuiltDocumentsAreReadOnly() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let engine = RAGEngine(configuration: configuration())
        try await engine.addPrebuiltIndex(at: try await buildIndex(in: directory))
        await #expect(throws: RAGError.prebuiltDocument("zorp")) { try await engine.removeDocument(id: "zorp") }
        await #expect(throws: RAGError.prebuiltDocument("zorp")) {
            _ = try await engine.addDocument(DocumentIngestor.ingestText("## X\nY.", title: "X", id: "zorp"))
        }
    }

    @Test func otherEmbedderReembedsInMemoryAndLeavesTheFileUntouched() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try await buildIndex(in: directory)
        let bytesBefore = try Data(contentsOf: url)

        let counter = EmbedCounter()
        let engine = RAGEngine(configuration: configuration(identifier: "other-model", counter: counter))
        try await engine.addPrebuiltIndex(at: url)
        #expect(await counter.passages == 3)
        #expect(try Data(contentsOf: url) == bytesBefore)
    }

    @Test func prebuiltIsNotSavedIntoStorageAndUserDocumentsAre() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try await buildIndex(in: directory)
        let storage = directory.appending(path: "storage")
        let engine = try await RAGEngine(configuration: configuration(), storageDirectory: storage)
        try await engine.addPrebuiltIndex(at: url)
        _ = try await engine.addDocument(DocumentIngestor.ingestText("## Notes\nMy own notes.", title: "Notes", id: "notes"))
        let files = try FileManager.default.contentsOfDirectory(atPath: storage.path)
        #expect(files == ["notes.ragdoc"])
    }

    @Test func clashingIDIsSkippedAndReported() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let engine = RAGEngine(configuration: configuration())
        _ = try await engine.addDocument(DocumentIngestor.ingestText("## Mine\nUser version.", title: "Mine", id: "zorp"))
        let infos = try await engine.addPrebuiltIndex(at: try await buildIndex(in: directory))
        #expect(infos.map(\.id) == ["cache"])
        #expect(await engine.isPrebuilt("zorp") == false)
        #expect(await engine.loadIssues.count == 1)
    }

    @Test func exportIsReproducible() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = try Data(contentsOf: try await buildIndex(in: directory))
        let second = try Data(contentsOf: try await buildIndex(in: directory))
        // addedAt differs between builds, so compare everything except it.
        func withoutDates(_ data: Data) throws -> [IngestedDocumentSummary] {
            try JSONDecoder().decode(IndexFile.self, from: data).documents.map(IngestedDocumentSummary.init)
        }
        #expect(try withoutDates(first) == withoutDates(second))
    }
}

private struct IngestedDocumentSummary: Equatable {
    let info: DocumentInfo
    let chunks: [Chunk]
    let vectors: Data
    init(_ record: DocumentRecord) { info = record.info; chunks = record.chunks; vectors = record.vectors }
}
```

- [ ] **Step 2: Run — expect compile failure** (`value of type 'RAGEngine' has no member 'exportIndex'`).

- [ ] **Step 3: Add to `RAGEngine`**

```swift
    // MARK: - Prebuilt indexes

    public func isPrebuilt(_ id: DocumentID) -> Bool {
        prebuiltIDs.contains(id)
    }

    /// Writes every non-prebuilt document into one read-only index file (what `rag-index build` produces).
    public func exportIndex(to url: URL) throws {
        let records = index.documents
            .filter { !prebuiltIDs.contains($0.info.id) }
            .map { record($0, index.vectors(of: $0.info.id), addedAt[$0.info.id] ?? Date()) }
        try DocumentStore.writeIndex(records, to: url)
    }

    /// Loads a shipped `.ragindex` file. Its documents are searchable but read-only, and are never
    /// written anywhere — if they were built with another embedder or other chunk settings, they're
    /// rebuilt in memory on every load (rebuild the index with `rag-index` to avoid that).
    @discardableResult
    public func addPrebuiltIndex(at url: URL) async throws -> [DocumentInfo] {
        var added: [DocumentInfo] = []
        for stored in try DocumentStore.readIndex(at: url) {
            guard index.document(stored.info.id) == nil else {
                loadIssues.append("\(url.lastPathComponent): skipped '\(stored.info.id)' — a document with that ID is already loaded")
                continue
            }
            let prepared = try await prepare(stored)
            index.add(prepared.document, vectors: prepared.vectors)
            addedAt[stored.info.id] = stored.addedAt
            prebuiltIDs.insert(stored.info.id)
            added.append(prepared.document.info)
        }
        return added
    }
```

- [ ] **Step 4: Run all tests — expect PASS.**
- [ ] **Step 5: Commit (when approved)** — `git commit -m "OnDeviceRAG: prebuilt read-only indexes and index export"`

---

### Task 5: `rag-index` command-line tool

**Files:**
- Modify: `Package.swift` (add `.executableTarget(name: "rag-index", dependencies: ["OnDeviceRAG"])` and product `.executable(name: "rag-index", targets: ["rag-index"])`)
- Create: `Sources/rag-index/main.swift`

**Interfaces:**
- Consumes: the public API only (`BGESmallEmbedder`, `RAGConfiguration`, `RAGEngine`, `DocumentIngestor`, `exportIndex`, `addPrebuiltIndex`, `context`).

Commands:
```
rag-index build --model <bge.aimodel> --output <file.ragindex> <input.pdf|.md|.txt>...
rag-index query --model <bge.aimodel> --index <file.ragindex> [--budget <tokens>] <question...>
```
Document IDs are the input file names without extension, so rebuilding gives the same IDs. Text inputs use `ingestText` (`## ` sections). Uses the default `AppleTokenCounter` (exact on a Mac with Apple Intelligence on).

- [ ] **Step 1: Add the target and product to `Package.swift`**

```swift
    products: [
        .library(name: "OnDeviceRAG", targets: ["OnDeviceRAG"]),
        .executable(name: "rag-index", targets: ["rag-index"]),
    ],
```
and in `targets`:
```swift
        .executableTarget(name: "rag-index", dependencies: ["OnDeviceRAG"]),
```

- [ ] **Step 2: Write `Sources/rag-index/main.swift`**

```swift
import Foundation
import OnDeviceRAG

// Builds and queries OnDeviceRAG indexes on a Mac, with the same code path the apps use.

let usage = """
usage:
  rag-index build --model <bge.aimodel> --output <file.ragindex> <input.pdf|.md|.txt>...
  rag-index query --model <bge.aimodel> --index <file.ragindex> [--budget <tokens>] <question...>
"""

struct Options {
    var model: URL?
    var output: URL?
    var index: URL?
    var budget = 2000
    var positional: [String] = []
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

func parse(_ arguments: ArraySlice<String>) -> Options {
    var options = Options()
    var remaining = arguments
    while let argument = remaining.popFirst() {
        switch argument {
        case "--model", "--output", "--index", "--budget":
            guard let value = remaining.popFirst() else { fail("missing value for \(argument)\n\n\(usage)") }
            switch argument {
            case "--model": options.model = URL(fileURLWithPath: value)
            case "--output": options.output = URL(fileURLWithPath: value)
            case "--index": options.index = URL(fileURLWithPath: value)
            default:
                guard let budget = Int(value), budget > 0 else { fail("--budget needs a positive number") }
                options.budget = budget
            }
        default:
            options.positional.append(argument)
        }
    }
    return options
}

let arguments = CommandLine.arguments.dropFirst()
guard let command = arguments.first else { fail(usage) }
let options = parse(arguments.dropFirst())
guard let modelURL = options.model else { fail("--model is required\n\n\(usage)") }

let clock = ContinuousClock()
let loadStart = clock.now
let engine = RAGEngine(configuration: RAGConfiguration(embedder: try await BGESmallEmbedder(modelURL: modelURL)))
print("model loaded in \(clock.now - loadStart)")

switch command {
case "build":
    guard let output = options.output, !options.positional.isEmpty else { fail(usage) }
    for path in options.positional {
        let url = URL(fileURLWithPath: path)
        let id = url.deletingPathExtension().lastPathComponent
        let start = clock.now
        let info: DocumentInfo
        if url.pathExtension.lowercased() == "pdf" {
            info = try await engine.addDocument(DocumentIngestor.ingestPDF(at: url, id: id))
        } else {
            let text = try String(contentsOf: url, encoding: .utf8)
            info = try await engine.addDocument(DocumentIngestor.ingestText(text, title: id, id: id))
        }
        var line = "\(id): \(info.sectionCount) sections (\(info.sectionSource.rawValue)), \(info.chunkCount) chunks, \(info.pageCount) pages, \(clock.now - start)"
        if !info.pagesWithoutText.isEmpty { line += ", pages without text: \(info.pagesWithoutText)" }
        if !info.mathHeavyPages.isEmpty { line += ", math-heavy pages: \(info.mathHeavyPages)" }
        print(line)
    }
    try await engine.exportIndex(to: output)
    let size = (try? FileManager.default.attributesOfItem(atPath: output.path)[.size] as? Int) ?? 0
    print("wrote \(output.path) (\(size / 1024) KB)")

case "query":
    guard let indexURL = options.index, !options.positional.isEmpty else { fail(usage) }
    try await engine.addPrebuiltIndex(at: indexURL)
    let question = options.positional.joined(separator: " ")
    let start = clock.now
    let context = try await engine.context(for: question, tokenBudget: options.budget)
    print("query took \(clock.now - start); \(context.tokenCount) tokens\(context.tokenCountIsEstimate ? " (estimate)" : "")")
    for section in context.sections {
        print("- \(section.title) (\(section.documentTitle), \(section.pages.label))\(section.isWholeSection ? "" : " [window]")")
    }
    print("\n\(context.text)")

default:
    fail(usage)
}
```

- [ ] **Step 3: Build and run the tests** — `swift build --package-path Packages/OnDeviceRAG` (no errors), `swift test --package-path Packages/OnDeviceRAG` (all pass).

- [ ] **Step 4: Manual check with the real model** (from the repo root):

```bash
M=~/Developer/coreai-models/exports/bge_small_buckets_float16.aimodel
swift run -c release --package-path Packages/OnDeviceRAG rag-index build --model $M --output /tmp/manuals.ragindex evaluation/rag/manuals/*.md
swift run -c release --package-path Packages/OnDeviceRAG rag-index query --model $M --index /tmp/manuals.ragindex "How do I encode a list in ZORP-7?"
```
Expected: `build` prints 4 lines (10 sections each, `provided`) and the file size; `query` lists the ZORP-7 spec section first and prints its text. Report the timings (model load, per-manual build, query). Then run `build` once more to a second file and confirm the two files have the same size and `rag-index query` prints identical sections and text for both (only `addedAt` differs inside).

- [ ] **Step 5: iOS build** — `xcodebuild -scheme OnDeviceRAG -destination 'generic/platform=iOS' -derivedDataPath /tmp/odrag-ios build` from `Packages/OnDeviceRAG` → `BUILD SUCCEEDED` (the executable target is macOS-only in practice and must not break the library's iOS build).
- [ ] **Step 6: Commit (when approved)** — `git commit -m "OnDeviceRAG: rag-index command-line tool"`

---

### Task 6: Documentation

**Files:**
- Modify: `Packages/OnDeviceRAG/README.md` (persistence, prebuilt indexes, `rag-index`; drop the "index lives in memory" limitation)
- Modify: `Packages/OnDeviceRAG/Docs/design.md` (§4 API, §6 Index — the three deviations above, §11 mark 2a done)

- [ ] **Step 1: README** — add a "Saving and shipping indexes" section:

```swift
// Saved between launches (re-embedding only when settings or the embedder change):
let engine = try await RAGEngine(configuration: configuration, storageDirectory: appSupportURL.appending(path: "RAG"))
// Documents shipped with the app, built on a Mac with `rag-index build`:
try await engine.addPrebuiltIndex(at: Bundle.main.url(forResource: "textbook", withExtension: "ragindex")!)
```
plus the two `rag-index` commands, and note `loadIssues`.
- [ ] **Step 2: design.md** — §4: the new init, `addPrebuiltIndex(at:)` taking a file, `exportIndex(to:)`, `isPrebuilt(_:)`, `loadIssues`, `removeDocument` throws. §6 Index: replace the four-file layout with the single-file formats and the rebuild rules, with the Xcode-flattening reason. §11: "2a done (persistence, prebuilt indexes, `rag-index`)"; 2b OCR and 2c app integration next.
- [ ] **Step 3: Full verification** — `swift test` (all pass, model-gated tests included), iOS build, `git diff --stat main -- "Prototype On-Device" evaluation` empty.
- [ ] **Step 4: Commit (when approved)** — `git commit -m "OnDeviceRAG: milestone 2a docs"`

---

## Self-review notes

- **Spec coverage (design §6/§11, milestone 2 part a):** on-disk index → Tasks 2–3; rebuild on embedder/settings change without the PDF → Tasks 1, 3; prebuilt indexes → Task 4; `rag-index` CLI, same code path → Task 5; docs → Task 6. OCR → plan 2b; app integration → plan 2c.
- **Not covered here:** concurrent writers to one storage directory (one engine per directory is assumed — documented in the README), index compaction/size limits, encryption at rest (files inherit the app's data protection).
