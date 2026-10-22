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
        // The empty preamble section is dropped, so A = 0, B = 1.
        let context = await ContextPacker.pack(candidates(idx, [(1, 0), (0, 0)]), index: idx, tokenBudget: 2000, counter: counter)
        #expect(context.text == "## B (Doc, p. 1)\nBeta text.\n\n## A (Doc, p. 1)\nAlpha text.")
        #expect(context.sections.map(\.title) == ["B", "A"])
        #expect(context.sections.allSatisfy { $0.isWholeSection })
        #expect(context.tokenCountIsEstimate)
        #expect(context.tokenCount == Int((Double(context.text.count) / 3).rounded(.up)))
    }

    @Test func sameSectionIsUsedOnce() async throws {
        let sentences = (1...6).map { "Sentence number \($0) is here." }.joined(separator: " ")
        let idx = try await index("## A\n\(sentences)")
        let context = await ContextPacker.pack(candidates(idx, idx.chunks.map { ($0.sectionIndex, $0.position) }),
                                               index: idx, tokenBudget: 2000, counter: counter)
        #expect(context.sections.count == 1)
    }

    @Test func oversizedSectionBecomesAWindowAroundTheMatch() async throws {
        // 12 sentences of ~200 characters → several ≤400-character chunks; the budget fits ~2 of them.
        let long = (1...12).map { "Sentence \($0) " + String(repeating: "filler ", count: 26) + "end." }.joined(separator: " ")
        let idx = try await index("## Big\n\(long)")
        #expect(idx.chunks.count >= 5)
        let context = await ContextPacker.pack(candidates(idx, [(0, 3)]), index: idx, tokenBudget: 300, counter: counter)
        #expect(context.sections.count == 1)
        #expect(context.sections.first?.isWholeSection == false)
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
