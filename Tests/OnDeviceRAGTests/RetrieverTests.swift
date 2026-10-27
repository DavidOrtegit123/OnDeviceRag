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

    private func index(_ sections: [String]) async throws -> InMemoryIndex {
        let doc = DocumentIngestor.ingestText(sections.joined(separator: "\n\n"), title: "Manual", id: "manual")
        var index = InMemoryIndex(dimension: 64, k1: 1.2, b: 0.75)
        index.add(doc, vectors: try await FakeEmbedder().embed(doc.chunks.map(\.indexedText), as: .passage))
        return index
    }

    @Test func aNewTopicIsNotPushedOutByEarlierQuestions() async throws {
        let index = try await index([
            "## Specifications\nOne controller supports up to 64 sensors.",
            "## Lamp schedule\nThe dusk cutoff decides when lamps turn off late in the evening; too much light harms plants.",
            "## Lamp timeout\nLamps that do not light in the evening are switched off and plants get no extra light.",
            "## Status LEDs\nA blinking LED on the control panel means an update is waiting; a steady LED means normal.",
            "## LED panel\nThe control panel has a blinking LED for each module and a steady LED for power.",
        ])
        let history: [ChatTurn] = [
            .user("What does a blinking LED on the control panel mean?"),
            .user("My plants get too much light late in the evening, which setting decides when the lamps turn off?"),
        ]
        let results = try await Retriever.retrieve(question: "How many sensors can one controller support?", history: history,
                                                   searchTurns: 2, index: index, embedder: FakeEmbedder(),
                                                   settings: RetrievalSettings(), limit: 2)
        #expect(results.map(\.chunk.sectionTitle).contains("Specifications"))
    }

    @Test func theQuestionAloneCountsTwiceAsMuchAsTheHistorySearch() {
        #expect(Retriever.questionWeight == 2)
        #expect(Retriever.historyWeight == 1)
    }

    @Test func aFollowUpStillFindsTheEarlierTopic() async throws {
        let index = try await index([
            "## Ventilation timeout\nThe ventilation vent actuator timeout is 312 seconds.",
            // The longest section, so the question alone ranks it last.
            "## Heating timeout\nAfter a request, the heating boiler must report a rising supply temperature, and the timeout is 410 seconds.",
            "## Lighting timeout\nThe lighting lamp driver timeout is 95 seconds.",
            "## Irrigation timeout\nThe irrigation zone valve timeout is 240 seconds.",
            "## Dosing timeout\nThe CO2 dosing valve timeout is 180 seconds.",
            "## Shading timeout\nThe shading screen motor timeout is 150 seconds.",
        ])
        func heatingRank(_ history: [ChatTurn]) async throws -> Int? {
            try await Retriever.retrieve(question: "How long is its timeout?", history: history, searchTurns: 2,
                                         index: index, embedder: FakeEmbedder(), settings: RetrievalSettings(), limit: 6)
                .firstIndex { $0.chunk.sectionTitle == "Heating timeout" }
        }
        let alone = try await heatingRank([])
        let followUp = try await heatingRank([.user("Tell me about the heating boiler.")])
        #expect(alone == 5)  // last of six without history
        // The question still counts twice, so history lifts the earlier topic without forcing it first.
        #expect(followUp.map { $0 < 5 } == true)
    }

    @Test func emptyIndexReturnsNothing() async throws {
        let results = try await Retriever.retrieve(question: "anything", history: [], searchTurns: 2, index: InMemoryIndex(dimension: 64, k1: 1.2, b: 0.75),
                                                   embedder: FakeEmbedder(), settings: RetrievalSettings(), limit: 20)
        #expect(results.isEmpty)
    }
}
