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

    /// The research project's 40 specs (4 manuals) and 80 questions; the gold section is known.
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
            var hitAt5 = 0
            var sectionTop3 = 0
            var misses: [String] = []
            let queries = try FortyTasks.queries().filter { $0.variant == variant }
            for query in queries {
                let goldTitle = try #require(gold[query.task_id])
                let hits = try await engine.search(query.query)
                if hits.prefix(5).contains(where: { $0.chunk.sectionTitle == goldTitle }) { hitAt5 += 1 }
                var titles: [String] = []
                for hit in hits where !titles.contains(hit.chunk.sectionTitle) {
                    titles.append(hit.chunk.sectionTitle)
                    if titles.count == 3 { break }
                }
                if titles.contains(goldTitle) {
                    sectionTop3 += 1
                } else {
                    misses.append("\(query.task_id): gold '\(goldTitle)', got \(titles)")
                }
            }
            print("40-task check (\(variant)): chunk hit@5 \(hitAt5)/\(queries.count), section top-3 \(sectionTop3)/\(queries.count)")
            misses.forEach { print("  miss \($0)") }
            // Spike: dense-only 40/40, BM25-only 38–40/40. Hybrid should not be worse than BM25 alone.
            #expect(sectionTop3 >= 38, "\(variant): section top-3 \(sectionTop3)/\(queries.count)")
        }
    }
}

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
