/// Optional second pass that re-orders the top candidates (e.g. with a language model).
/// Must return a reordering of exactly the chunks it was given; anything else is ignored.
public protocol Reranker: Sendable {
    func rerank(_ question: String, _ candidates: [RetrievedChunk]) async throws -> [RetrievedChunk]
}

/// Every tunable setting and pluggable component. Defaults follow design.md §8.
public struct RAGConfiguration: Sendable {
    public var embedder: any Embedder
    public var tokenCounter: any TokenCounter = AppleTokenCounter()
    /// Off by default; milestone 3's benchmark decides whether that changes.
    public var reranker: (any Reranker)? = nil
    public var rerankCount = 10
    public var ingest = IngestSettings()
    public var retrieval = RetrievalSettings()
    public var bm25K1 = 1.2
    public var bm25B = 0.75
    public var tokenBudget = 2000

    public init(embedder: any Embedder) {
        self.embedder = embedder
    }
}
