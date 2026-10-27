public struct RetrievalSettings: Sendable, Hashable {
    public var fusionConstant = 60.0
    public var candidateCount = 20
    public init() {}
}

/// Dense (embedding) + BM25 rankings over every chunk, merged with Reciprocal Rank Fusion.
///
/// With searchable history, the question is searched twice — alone and together with the recent
/// user turns — and all four rankings are fused, the question-alone ones with twice the weight.
/// On a topic change the question-alone rankings keep the new topic near the top; on a follow-up
/// ("how long is its timeout?") the history search lifts the earlier topic's section. Without
/// history it is a single search. Weights were chosen on a 63-page test manual: equal weights
/// still let two earlier questions push a new topic's answer out of the context.
enum Retriever {
    static let questionWeight = 2.0
    static let historyWeight = 1.0

    /// The question, then up to `searchTurns` recent *user* turns, newest first — so when the
    /// embedder truncates at 512 tokens, only history is cut. Assistant turns are never searched.
    static func queryText(question: String, history: [ChatTurn], searchTurns: Int) -> String {
        let recent = history.filter { $0.role == .user }.suffix(max(searchTurns, 0)).reversed().map(\.text)
        return ([question] + recent).joined(separator: "\n")
    }

    static func retrieve(question: String, history: [ChatTurn], searchTurns: Int, index: InMemoryIndex,
                         embedder: any Embedder, settings: RetrievalSettings, limit: Int) async throws -> [RetrievedChunk] {
        guard !index.chunks.isEmpty else { return [] }
        let withHistory = queryText(question: question, history: history, searchTurns: searchTurns)
        let queries = withHistory == question ? [question] : [question, withHistory]
        let vectors = try await embedder.embed(queries, as: .query)
        var orders: [[Int]] = []
        var weights: [Double] = []
        for (position, (query, vector)) in zip(queries, vectors).enumerated() {
            orders.append(index.denseOrder(for: vector))
            orders.append(index.keywordOrder(for: query))
            let weight = position == 0 ? questionWeight : historyWeight
            weights += [weight, weight]
        }
        // The reported ranks are those of the question-alone search.
        let denseRanks = Dictionary(uniqueKeysWithValues: orders[0].enumerated().map { ($1, $0 + 1) })
        let keywordRanks = Dictionary(uniqueKeysWithValues: orders[1].enumerated().map { ($1, $0 + 1) })
        let titles = Dictionary(uniqueKeysWithValues: index.documents.map { ($0.info.id, $0.info.title) })

        // A single search keeps weight 1, so scores without history are unchanged.
        let fusionWeights = queries.count == 1 ? nil : weights
        return RankFusion.fuse(orders, weights: fusionWeights, count: index.chunks.count, constant: settings.fusionConstant)
            .prefix(limit)
            .map { entry in
                let chunk = index.chunks[entry.index]
                return RetrievedChunk(chunk: chunk, documentTitle: titles[chunk.documentID] ?? "",
                                      fusedScore: entry.score, denseRank: denseRanks[entry.index],
                                      keywordRank: keywordRanks[entry.index])
            }
    }
}
