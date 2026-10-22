import Accelerate

/// All chunks of all documents, their vectors (one flat array, `dimension` floats per chunk)
/// and a BM25 index over their indexed text. Milestone 2 adds saving/loading.
struct InMemoryIndex: Sendable {
    let dimension: Int
    let k1: Double
    let b: Double

    private(set) var documents: [IngestedDocument] = []
    private(set) var chunks: [Chunk] = []
    private var vectors: [Float] = []
    private var keywordIndex: BM25Index

    init(dimension: Int, k1: Double, b: Double) {
        self.dimension = dimension
        self.k1 = k1
        self.b = b
        keywordIndex = BM25Index(documents: [], k1: k1, b: b)
    }

    /// Adds (or replaces, same ID) a document. `vectors` must be in its chunk order, `dimension` floats each.
    mutating func add(_ document: IngestedDocument, vectors documentVectors: [[Float]]) {
        precondition(documentVectors.count == document.chunks.count, "one vector per chunk")
        precondition(documentVectors.allSatisfy { $0.count == dimension }, "vector dimension mismatch")
        remove(document.info.id)
        documents.append(document)
        chunks += document.chunks
        vectors += documentVectors.flatMap { $0 }
        rebuildKeywordIndex()
    }

    mutating func remove(_ id: DocumentID) {
        guard documents.contains(where: { $0.info.id == id }) else { return }
        var keptVectors: [Float] = []
        for (position, chunk) in chunks.enumerated() where chunk.documentID != id {
            keptVectors += vectors[(position * dimension)..<((position + 1) * dimension)]
        }
        vectors = keptVectors
        chunks.removeAll { $0.documentID == id }
        documents.removeAll { $0.info.id == id }
        rebuildKeywordIndex()
    }

    func document(_ id: DocumentID) -> IngestedDocument? {
        documents.first { $0.info.id == id }
    }

    /// Chunk indices by cosine similarity (vectors are normalized, so a dot product), best first;
    /// ties go to the earlier chunk.
    func denseOrder(for query: [Float]) -> [Int] {
        guard !chunks.isEmpty else { return [] }
        precondition(query.count == dimension, "query dimension mismatch")
        var scores = [Float](repeating: 0, count: chunks.count)
        vDSP_mmul(vectors, 1, query, 1, &scores, 1, vDSP_Length(chunks.count), 1, vDSP_Length(dimension))
        return chunks.indices.sorted { scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1 }
    }

    /// Chunk indices with a positive BM25 score, best first; ties go to the earlier chunk.
    func keywordOrder(for query: String) -> [Int] {
        let scores = keywordIndex.scores(for: query)
        return chunks.indices
            .filter { scores[$0] > 0 }
            .sorted { scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1 }
    }

    private mutating func rebuildKeywordIndex() {
        keywordIndex = BM25Index(documents: chunks.map(\.indexedText), k1: k1, b: b)
    }
}
