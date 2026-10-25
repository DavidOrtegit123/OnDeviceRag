import Foundation
import Testing
@testable import OnDeviceRAG

private func configuration() -> RAGConfiguration {
    var configuration = RAGConfiguration(embedder: FakeEmbedder())
    configuration.tokenCounter = EstimatingTokenCounter()
    return configuration
}

private let cache = "## Cache\nThe cache stores loaded values."

struct RenameTests {
    @Test func renameChangesTheTitleInDocumentsAndContext() async throws {
        let engine = RAGEngine(configuration: configuration())
        _ = try await engine.addDocument(DocumentIngestor.ingestText(cache, title: "Old name", id: "c"))
        let info = try await engine.renameDocument(id: "c", to: "New name")
        #expect(info.title == "New name")
        #expect(await engine.documents.map(\.title) == ["New name"])
        let text = try await engine.context(for: "cache").text
        #expect(text.contains("(New name, p. 1)"))
        #expect(!text.contains("Old name"))
    }

    @Test func renameIsSavedAndKeepsTheDate() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "rename-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = try await RAGEngine(configuration: configuration(), storageDirectory: directory)
        _ = try await first.addDocument(DocumentIngestor.ingestText(cache, title: "Old name", id: "c"))
        let before = DocumentStore.loadAll(in: directory).records.first?.addedAt
        try await first.renameDocument(id: "c", to: "New name")
        let reopened = try await RAGEngine(configuration: configuration(), storageDirectory: directory)
        #expect(await reopened.documents.map(\.title) == ["New name"])
        #expect(before != nil)
        #expect(DocumentStore.loadAll(in: directory).records.first?.addedAt == before)
    }

    @Test func renamingUnknownOrPrebuiltDocumentsThrows() async throws {
        let builder = RAGEngine(configuration: configuration())
        _ = try await builder.addDocument(DocumentIngestor.ingestText(cache, title: "Shipped", id: "s"))
        let url = FileManager.default.temporaryDirectory.appending(path: "rename-\(UUID().uuidString).ragindex")
        defer { try? FileManager.default.removeItem(at: url) }
        try await builder.exportIndex(to: url)
        let engine = RAGEngine(configuration: configuration())
        try await engine.addPrebuiltIndex(at: url)
        await #expect(throws: RAGError.prebuiltDocument("s")) { try await engine.renameDocument(id: "s", to: "X") }
        await #expect(throws: RAGError.unknownDocument("nope")) { try await engine.renameDocument(id: "nope", to: "X") }
    }
}
