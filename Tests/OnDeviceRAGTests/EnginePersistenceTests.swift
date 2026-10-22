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
        let pdf = FixturePDF.make(pages: [[.heading("Integers")] + body.map { .body($0) },
                                          [.heading("Lists"), .body("Lists are written as l<items>e.")]])
        #expect(pdf.write(to: pdfURL))
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
        let parent = FileManager.default.temporaryDirectory.appending(path: "ondevice-rag-\(UUID().uuidString)")
        let directory = parent.appending(path: "nested")
        defer { try? FileManager.default.removeItem(at: parent) }
        let created = try await engine(at: directory, counter: EmbedCounter())
        _ = try await created.addDocument(DocumentIngestor.ingestText(cache, title: "Cache", id: "cache"))
        #expect(FileManager.default.fileExists(atPath: DocumentStore.fileURL(for: "cache", in: directory).path))
    }
}
