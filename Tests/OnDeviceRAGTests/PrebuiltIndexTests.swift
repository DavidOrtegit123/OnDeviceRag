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
