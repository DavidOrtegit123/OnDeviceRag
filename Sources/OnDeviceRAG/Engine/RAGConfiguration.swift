/// Optional second pass that re-orders the top candidates (e.g. with a language model).
/// Must return a reordering of exactly the chunks it was given; anything else is ignored.
public protocol Reranker: Sendable {
    func rerank(_ question: String, _ candidates: [RetrievedChunk]) async throws -> [RetrievedChunk]
}

/// Conversation history — both uses are off by default, so passing history changes nothing
/// until an app turns them on.
public struct HistorySettings: Sendable, Hashable {
    /// Add the last `searchTurns` user turns to the search query (follow-up questions).
    public var useForSearch = false
    /// Return the recent turns that fit `tokenBudget` in `RAGContext.history` (for the prompt).
    public var includeInContext = false
    public var searchTurns = 2
    public var maximumTurns = 6
    /// Separate from the sections' `tokenBudget`.
    public var tokenBudget = 600
    public init() {}
}

/// Every tunable setting and pluggable component. Defaults follow design.md §8.
public struct RAGConfiguration: Sendable {
    public var embedder: any Embedder
    public var tokenCounter: any TokenCounter = AppleTokenCounter()
    /// Off by default; milestone 3's benchmark decides whether that changes.
    public var reranker: (any Reranker)? = nil
    public var rerankCount = 10
    public var ingest = IngestSettings()
    /// Read PDF pages without a text layer (scans) with Vision OCR (~0.2 s per page on a Mac).
    /// Not part of `IngestSettings`: re-chunking a stored document can't re-run OCR.
    public var ocrScannedPages = true
    public var retrieval = RetrievalSettings()
    public var history = HistorySettings()
    public var bm25K1 = 1.2
    public var bm25B = 0.75
    public var tokenBudget = 2000

    public init(embedder: any Embedder) {
        self.embedder = embedder
    }
}
