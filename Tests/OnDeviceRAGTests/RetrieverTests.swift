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

    @Test func questionComesFirstThenRecentUserTurnsNewestFirst() {
        let history: [ChatTurn] = [.user("old"), .assistant("an answer"), .user("middle"), .assistant("another"), .user("newest")]
        #expect(Retriever.queryText(question: "Q?", history: history, searchTurns: 2) == "Q?\nnewest\nmiddle")
        #expect(Retriever.queryText(question: "Q?", history: [.assistant("only answers")], searchTurns: 2) == "Q?")
        #expect(Retriever.queryText(question: "Q?", history: [], searchTurns: 2) == "Q?")
    }

    @Test func retrievesTheMatchingChunkFirstWithBothRanks() async throws {
        let index = try await makeIndex()
        let results = try await Retriever.retrieve(question: "how are lists encoded", history: [], searchTurns: 2, index: index,
                                                   embedder: FakeEmbedder(), settings: RetrievalSettings(), limit: 20)
        #expect(results.first?.chunk.sectionTitle == "Lists")
        #expect(results.first?.documentTitle == "ZORP")
        #expect(results.first?.denseRank != nil)
        #expect(results.first?.keywordRank != nil)
        #expect(results.count == index.chunks.count) // 3 chunks < limit
    }

    @Test func limitCapsResults() async throws {
        let index = try await makeIndex()
        let results = try await Retriever.retrieve(question: "cache", history: [], searchTurns: 2, index: index,
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
        let results = try await Retriever.retrieve(question: "anything", history: [], searchTurns: 2, index: InMemoryIndex(dimension: 64, k1: 1.2, b: 0.75),
                                                   embedder: FakeEmbedder(), settings: RetrievalSettings(), limit: 20)
        #expect(results.isEmpty)
    }
}
