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

        try await engine.removeDocument(id: "zorp")
        let remaining = try await engine.search("lists")
        #expect(remaining.allSatisfy { $0.chunk.documentID == "cache" })
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
        let plainTitles = try await plain.context(for: "lists encoded").sections.map(\.title)
        let rerankedTitles = try await reranked.context(for: "lists encoded").sections.map(\.title)
        let brokenTitles = try await broken.context(for: "lists encoded").sections.map(\.title)
        #expect(plainTitles.count == 2)
        #expect(rerankedTitles == Array(plainTitles.reversed()))
        #expect(brokenTitles == plainTitles)
    }

    @Test func tokenBudgetOverride() async throws {
        let engine = RAGEngine(configuration: configuration())
        _ = try await engine.addDocument(DocumentIngestor.ingestText(zorp, title: "ZORP", id: "zorp"))
        let context = try await engine.context(for: "lists", tokenBudget: 20)
        #expect(context.tokenCount <= 20)
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
        let context = try await engine.context(for: "lists")
        #expect(context.sections.first?.title == "Lists")
    }
    #endif
}
