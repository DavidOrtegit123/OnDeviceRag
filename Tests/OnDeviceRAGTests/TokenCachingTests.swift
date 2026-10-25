import Foundation
import Testing
@testable import OnDeviceRAG

private let twoSections = "## Integers\nIntegers are encoded as i<n>e.\n\n## Lists\nLists are encoded as l<items>e."

private func index(_ text: String) async throws -> InMemoryIndex {
    let doc = DocumentIngestor.ingestText(text, title: "Doc", id: "d")
    var index = InMemoryIndex(dimension: 64, k1: 1.2, b: 0.75)
    index.add(doc, vectors: try await FakeEmbedder().embed(doc.chunks.map(\.indexedText), as: .passage))
    return index
}

private func allCandidates(_ index: InMemoryIndex) -> [RetrievedChunk] {
    index.chunks.map { RetrievedChunk(chunk: $0, documentTitle: "Doc", fusedScore: 0, denseRank: nil, keywordRank: nil) }
}

private func engine(counter: TallyCounter) -> RAGEngine {
    var configuration = RAGConfiguration(embedder: FakeEmbedder())
    configuration.tokenCounter = counter
    return RAGEngine(configuration: configuration)
}

struct TokenCachingTests {
    @Test func cacheCallsTheCounterOncePerText() async {
        let tally = TallyCounter()
        let cache = CachingTokenCounter(tally)
        let first = await cache.count("Lists are encoded as l<items>e.")
        let second = await cache.count("Lists are encoded as l<items>e.")
        _ = await cache.count("Something else.")
        #expect(first == second)
        #expect(await tally.calls == 2)
    }

    @Test func estimatesAreNotCached() async {
        // An estimate may become exact later (Apple Intelligence switched on), so it's never remembered.
        let tally = TallyCounter(exact: false)
        let cache = CachingTokenCounter(tally)
        _ = await cache.count("Same text.")
        _ = await cache.count("Same text.")
        #expect(await tally.calls == 2)
    }

    @Test func packingIsIdenticalWithAndWithoutTheCache() async throws {
        let long = (1...12).map { "Sentence \($0) " + String(repeating: "filler ", count: 26) + "end." }.joined(separator: " ")
        let text = twoSections + "\n\n## Big\n" + long
        let idx = try await index(text)
        let candidates = allCandidates(idx)
        let cache = CachingTokenCounter(TallyCounter())
        for budget in [20, 60, 150, 400, 2000] {
            let plain = await ContextPacker.pack(candidates, index: idx, tokenBudget: budget, counter: TallyCounter())
            // Twice through the same cache: cold, then warm.
            for _ in 0..<2 {
                let cached = await ContextPacker.pack(candidates, index: idx, tokenBudget: budget, counter: cache)
                #expect(cached.text == plain.text, "budget \(budget)")
                #expect(cached.tokenCount == plain.tokenCount, "budget \(budget)")
                #expect(cached.sections == plain.sections, "budget \(budget)")
            }
        }
    }

    @Test func separatorIsCountedOncePerPack() async throws {
        let idx = try await index(twoSections + "\n\n## Maps\nMaps are encoded as d<pairs>e.")
        let tally = TallyCounter()
        let context = await ContextPacker.pack(allCandidates(idx), index: idx, tokenBudget: 2000, counter: tally)
        #expect(context.sections.count == 3)
        #expect(await tally.texts.filter { $0 == ContextPacker.blockSeparator }.count == 1)
    }

    @Test func afterWarmUpANewQuestionCountsOnlyTheJoinedText() async throws {
        let tally = TallyCounter()
        let rag = engine(counter: tally)
        _ = try await rag.addDocument(DocumentIngestor.ingestText(twoSections, title: "ZORP", id: "zorp"))
        await rag.waitForTokenWarmUp()
        await tally.reset()

        let first = try await rag.context(for: "how are lists encoded")
        #expect(first.sections.count == 2)
        #expect(await tally.calls == 1)   // only the joined text of the two sections
        _ = try await rag.context(for: "how are lists encoded")
        #expect(await tally.calls == 1)   // the repeat is fully cached
    }

    @Test func prebuiltAndStoredDocumentsAreWarmedToo() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "ondevice-rag-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        var configuration = RAGConfiguration(embedder: FakeEmbedder())
        configuration.tokenCounter = EstimatingTokenCounter()
        let storage = directory.appending(path: "storage")
        let builder = try await RAGEngine(configuration: configuration, storageDirectory: storage)
        _ = try await builder.addDocument(DocumentIngestor.ingestText(twoSections, title: "ZORP", id: "zorp"))
        let indexURL = directory.appending(path: "shipped.ragindex")
        try await builder.exportIndex(to: indexURL)

        // Stored documents, loaded at init.
        let storedTally = TallyCounter()
        configuration.tokenCounter = storedTally
        let reopened = try await RAGEngine(configuration: configuration, storageDirectory: storage)
        await reopened.waitForTokenWarmUp()
        #expect(await storedTally.calls == 3)   // separator + 2 sections

        // Prebuilt documents.
        let prebuiltTally = TallyCounter()
        let shipped = engine(counter: prebuiltTally)
        try await shipped.addPrebuiltIndex(at: indexURL)
        await shipped.waitForTokenWarmUp()
        #expect(await prebuiltTally.calls == 3)
    }
}
