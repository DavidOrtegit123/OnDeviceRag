import Foundation

/// On-device retrieval for any language model: add documents, then ask for the context that
/// best answers a question, packed to fit a token budget. With a storage directory, documents
/// are saved and reloaded without re-embedding.
public actor RAGEngine {
    private let configuration: RAGConfiguration
    private let storageDirectory: URL?
    private var index: InMemoryIndex
    private var addedAt: [DocumentID: Date] = [:]
    private var prebuiltIDs: Set<DocumentID> = []

    /// Documents that couldn't be loaded (unreadable file, unsupported format, ID clash), one line each.
    public private(set) var loadIssues: [String] = []

    /// In-memory engine: nothing is saved.
    public init(configuration: RAGConfiguration) {
        self.init(configuration: configuration, storage: nil)
    }

    /// Persistent engine: loads every document saved in `storageDirectory` (created if missing),
    /// re-chunking or re-embedding those built with other settings or another embedder.
    public init(configuration: RAGConfiguration, storageDirectory: URL) async throws {
        self.init(configuration: configuration, storage: storageDirectory)
        try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
        try await loadStoredDocuments()
    }

    private init(configuration: RAGConfiguration, storage: URL?) {
        self.configuration = configuration
        storageDirectory = storage
        index = InMemoryIndex(dimension: configuration.embedder.dimension, k1: configuration.bm25K1, b: configuration.bm25B)
    }

    public var documents: [DocumentInfo] { index.documents.map(\.info) }

    public func addDocument(pdfAt url: URL, title: String? = nil) async throws -> DocumentInfo {
        try await addDocument(DocumentIngestor.ingestPDF(at: url, title: title, settings: configuration.ingest))
    }

    /// Adds (or replaces, same ID) a document; embeds its chunks and, with storage, saves it.
    public func addDocument(_ document: IngestedDocument) async throws -> DocumentInfo {
        guard !prebuiltIDs.contains(document.info.id) else { throw RAGError.prebuiltDocument(document.info.id) }
        let vectors = try await configuration.embedder.embed(document.chunks.map(\.indexedText), as: .passage)
        let date = addedAt[document.info.id] ?? Date()
        if let storageDirectory {
            try DocumentStore.save(record(document, vectors, date), in: storageDirectory)
        }
        index.add(document, vectors: vectors)
        addedAt[document.info.id] = date
        return document.info
    }

    public func removeDocument(id: DocumentID) throws {
        guard !prebuiltIDs.contains(id) else { throw RAGError.prebuiltDocument(id) }
        if let storageDirectory {
            try DocumentStore.delete(id, in: storageDirectory)
        }
        index.remove(id)
        addedAt[id] = nil
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

    // MARK: - Loading

    private func loadStoredDocuments() async throws {
        guard let storageDirectory else { return }
        let loaded = DocumentStore.loadAll(in: storageDirectory)
        loadIssues += loaded.issues
        for stored in loaded.records {
            let prepared = try await prepare(stored)
            if prepared.changed {
                try DocumentStore.save(record(prepared.document, prepared.vectors, stored.addedAt), in: storageDirectory)
            }
            index.add(prepared.document, vectors: prepared.vectors)
            addedAt[prepared.document.info.id] = stored.addedAt
        }
    }

    /// Brings a stored record up to the current chunk settings and embedder:
    /// settings changed → re-chunk from its source and re-embed; embedder changed or vectors
    /// damaged → re-embed; otherwise use it as stored.
    private func prepare(_ stored: DocumentRecord) async throws -> (document: IngestedDocument, vectors: [[Float]], changed: Bool) {
        var document = stored.document
        var vectors = stored.decodedVectors()
        if stored.settings != configuration.ingest {
            document = DocumentIngestor.reingest(stored.source, title: stored.info.title, id: stored.info.id, settings: configuration.ingest)
            vectors = nil
        }
        if stored.embedder != configuration.embedder.identifier || stored.dimension != configuration.embedder.dimension {
            vectors = nil
        }
        if let vectors {
            return (document, vectors, false)
        }
        let embedded = try await configuration.embedder.embed(document.chunks.map(\.indexedText), as: .passage)
        return (document, embedded, true)
    }

    private func record(_ document: IngestedDocument, _ vectors: [[Float]], _ date: Date) -> DocumentRecord {
        DocumentRecord(document: document, vectors: vectors, embedder: configuration.embedder.identifier,
                       dimension: configuration.embedder.dimension, settings: configuration.ingest, addedAt: date)
    }
}
