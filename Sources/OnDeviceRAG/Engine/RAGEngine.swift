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
    /// The configured counter, with exact counts remembered (see `CachingTokenCounter`).
    private let tokenCounter: CachingTokenCounter
    /// Background pre-counting of section blocks; each run waits for the previous one.
    private var warmUp: Task<Void, Never>?

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
        tokenCounter = CachingTokenCounter(configuration.tokenCounter)
        index = InMemoryIndex(dimension: configuration.embedder.dimension, k1: configuration.bm25K1, b: configuration.bm25B)
    }

    public var documents: [DocumentInfo] { index.documents.map(\.info) }

    public func addDocument(pdfAt url: URL, title: String? = nil) async throws -> DocumentInfo {
        try await addDocument(DocumentIngestor.ingestPDF(at: url, title: title, settings: configuration.ingest,
                                                         ocr: configuration.ocrScannedPages))
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
        warmTokenCounts(for: document)
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

    /// Gives a document a new title: re-chunks it from its stored source (the title heads text
    /// documents' first section and every packed block), re-embeds it and, with storage, saves it.
    /// Keeps its added-at date.
    @discardableResult
    public func renameDocument(id: DocumentID, to title: String) async throws -> DocumentInfo {
        guard !prebuiltIDs.contains(id) else { throw RAGError.prebuiltDocument(id) }
        guard let current = index.document(id) else { throw RAGError.unknownDocument(id) }
        return try await addDocument(DocumentIngestor.reingest(current.source, title: title, id: id,
                                                               settings: configuration.ingest))
    }

    /// Best chunks first (hybrid dense + BM25). `history`: earlier turns, oldest first — used only
    /// when `configuration.history.useForSearch` is on.
    public func search(_ question: String, history: [ChatTurn] = [], limit: Int? = nil) async throws -> [RetrievedChunk] {
        let settings = configuration.history
        return try await Retriever.retrieve(question: question, history: settings.useForSearch ? history : [],
                                            searchTurns: settings.searchTurns, index: index, embedder: configuration.embedder,
                                            settings: configuration.retrieval, limit: limit ?? configuration.retrieval.candidateCount)
    }

    /// Text for the prompt: whole sections (or windows) in rank order, within the token budget.
    /// `history` (oldest first) is searched when `history.useForSearch` is on and returned, trimmed
    /// to `history.tokenBudget`, when `history.includeInContext` is on.
    public func context(for question: String, history: [ChatTurn] = [], tokenBudget: Int? = nil) async throws -> RAGContext {
        var candidates = try await search(question, history: history)
        if let reranker = configuration.reranker {
            candidates = await rerank(question, candidates, with: reranker)
        }
        var context = await ContextPacker.pack(candidates, index: index, tokenBudget: tokenBudget ?? configuration.tokenBudget,
                                               counter: tokenCounter)
        let settings = configuration.history
        if settings.includeInContext {
            let window = await HistoryWindow.select(history, maximumTurns: settings.maximumTurns,
                                                    tokenBudget: settings.tokenBudget, counter: tokenCounter)
            context.history = window.turns
            context.historyTokenCount = window.tokenCount
        }
        return context
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

    // MARK: - Prebuilt indexes

    public func isPrebuilt(_ id: DocumentID) -> Bool {
        prebuiltIDs.contains(id)
    }

    /// Writes every non-prebuilt document into one read-only index file (what `rag-index build` produces).
    public func exportIndex(to url: URL) throws {
        let records = index.documents
            .filter { !prebuiltIDs.contains($0.info.id) }
            .map { record($0, index.vectors(of: $0.info.id), addedAt[$0.info.id] ?? Date()) }
        try DocumentStore.writeIndex(records, to: url)
    }

    /// Loads a shipped `.ragindex` file. Its documents are searchable but read-only, and are never
    /// written anywhere — if they were built with another embedder or other chunk settings, they're
    /// rebuilt in memory on every load (rebuild the index with `rag-index` to avoid that).
    @discardableResult
    public func addPrebuiltIndex(at url: URL) async throws -> [DocumentInfo] {
        var added: [DocumentInfo] = []
        for stored in try DocumentStore.readIndex(at: url) {
            guard index.document(stored.info.id) == nil else {
                loadIssues.append("\(url.lastPathComponent): skipped '\(stored.info.id)' — a document with that ID is already loaded")
                continue
            }
            let prepared = try await prepare(stored)
            index.add(prepared.document, vectors: prepared.vectors)
            addedAt[stored.info.id] = stored.addedAt
            prebuiltIDs.insert(stored.info.id)
            added.append(prepared.document.info)
            warmTokenCounts(for: prepared.document)
        }
        return added
    }

    // MARK: - Token counts

    /// Counts the document's whole-section blocks (and the block separator) in the background, so
    /// the first question about it doesn't pay one slow exact count per section. Questions asked
    /// meanwhile still work; they just count what isn't cached yet.
    private func warmTokenCounts(for document: IngestedDocument) {
        let texts = [ContextPacker.blockSeparator]
            + document.sections.map { ContextPacker.wholeSectionText($0, documentTitle: document.info.title) }
        let counter = tokenCounter
        let previous = warmUp
        warmUp = Task {
            await previous?.value
            await counter.warm(texts)
        }
    }

    /// Waits until all background pre-counting has finished (for tests and benchmarks).
    func waitForTokenWarmUp() async {
        while let task = warmUp {
            await task.value
            if warmUp == task { warmUp = nil }
        }
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
            warmTokenCounts(for: prepared.document)
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
