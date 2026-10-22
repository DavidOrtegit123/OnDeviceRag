import Foundation

/// On-device retrieval for any language model: add documents, then ask for the context that
/// best answers a question, packed to fit a token budget.
public actor RAGEngine {
    private let configuration: RAGConfiguration
    private var index: InMemoryIndex

    public init(configuration: RAGConfiguration) {
        self.configuration = configuration
        index = InMemoryIndex(dimension: configuration.embedder.dimension, k1: configuration.bm25K1, b: configuration.bm25B)
    }

    public var documents: [DocumentInfo] { index.documents.map(\.info) }

    public func addDocument(pdfAt url: URL, title: String? = nil) async throws -> DocumentInfo {
        try await addDocument(DocumentIngestor.ingestPDF(at: url, title: title, settings: configuration.ingest))
    }

    /// Adds (or replaces, same ID) an already-ingested document; embeds its chunks.
    public func addDocument(_ document: IngestedDocument) async throws -> DocumentInfo {
        let vectors = try await configuration.embedder.embed(document.chunks.map(\.indexedText), as: .passage)
        index.add(document, vectors: vectors)
        return document.info
    }

    public func removeDocument(id: DocumentID) {
        index.remove(id)
    }

    /// Best chunks first (hybrid dense + BM25). `history`: earlier chat messages, oldest first; optional.
    public func search(_ question: String, history: [String] = [], limit: Int? = nil) async throws -> [RetrievedChunk] {
        try await Retriever.retrieve(question: question, history: history, index: index, embedder: configuration.embedder,
                                     settings: configuration.retrieval, limit: limit ?? configuration.retrieval.candidateCount)
    }

    /// Text for the prompt: whole sections (or windows) in rank order, within the token budget.
    public func context(for question: String, history: [String] = [], tokenBudget: Int? = nil) async throws -> RAGContext {
        var candidates = try await search(question, history: history)
        if let reranker = configuration.reranker {
            candidates = await rerank(question, candidates, with: reranker)
        }
        return await ContextPacker.pack(candidates, index: index, tokenBudget: tokenBudget ?? configuration.tokenBudget,
                                        counter: configuration.tokenCounter)
    }

    /// Reorders the first `rerankCount` candidates; any error or non-reordering output keeps the original order.
    private func rerank(_ question: String, _ candidates: [RetrievedChunk], with reranker: any Reranker) async -> [RetrievedChunk] {
        let head = Array(candidates.prefix(configuration.rerankCount))
        guard let reordered = try? await reranker.rerank(question, head),
              reordered.count == head.count,
              Set(reordered.map(\.chunk.id)) == Set(head.map(\.chunk.id)) else {
            return candidates
        }
        return reordered + candidates.dropFirst(head.count)
    }
}
