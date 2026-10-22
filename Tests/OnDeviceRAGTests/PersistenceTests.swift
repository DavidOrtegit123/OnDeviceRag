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
        #expect(loaded.records.first?.document.chunks == older.document.chunks)
        #expect(loaded.records.first?.document.source == older.document.source)

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
