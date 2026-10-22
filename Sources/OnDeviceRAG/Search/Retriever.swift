public struct RetrievalSettings: Sendable, Hashable {
    public var fusionConstant = 60.0
    public var candidateCount = 20
    public var historyMessages = 2
    public init() {}
}

/// Dense (embedding) + BM25 rankings over every chunk, merged with Reciprocal Rank Fusion.
enum Retriever {
    /// The question, then up to `historyMessages` recent messages, newest first — so when the
    /// embedder truncates at 512 tokens, only history is cut.
    static func queryText(question: String, history: [String], historyMessages: Int) -> String {
        ([question] + history.suffix(historyMessages).reversed()).joined(separator: "\n")
    }

    static func retrieve(question: String, history: [String], index: InMemoryIndex, embedder: any Embedder,
                         settings: RetrievalSettings, limit: Int) async throws -> [RetrievedChunk] {
        guard !index.chunks.isEmpty else { return [] }
        let query = queryText(question: question, history: history, historyMessages: settings.historyMessages)
        let queryVector = try await embedder.embed([query], as: .query)[0]
        let dense = index.denseOrder(for: queryVector)
        let keyword = index.keywordOrder(for: query)
        let denseRanks = Dictionary(uniqueKeysWithValues: dense.enumerated().map { ($1, $0 + 1) })
        let keywordRanks = Dictionary(uniqueKeysWithValues: keyword.enumerated().map { ($1, $0 + 1) })
        let titles = Dictionary(uniqueKeysWithValues: index.documents.map { ($0.info.id, $0.info.title) })

        return RankFusion.fuse([dense, keyword], count: index.chunks.count, constant: settings.fusionConstant)
            .prefix(limit)
            .map { entry in
                let chunk = index.chunks[entry.index]
                return RetrievedChunk(chunk: chunk, documentTitle: titles[chunk.documentID] ?? "",
                                      fusedScore: entry.score, denseRank: denseRanks[entry.index],
                                      keywordRank: keywordRanks[entry.index])
            }
    }
}
