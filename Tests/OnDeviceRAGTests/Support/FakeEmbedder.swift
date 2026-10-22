import Foundation
@testable import OnDeviceRAG

/// Deterministic bag-of-words embedder: each BM25 token adds 1 to a hashed dimension,
/// then the vector is L2-normalized. Texts sharing words get high cosine.
struct FakeEmbedder: Embedder {
    let identifier = "fake-bag-of-words"
    let dimension = 64

    func embed(_ texts: [String], as kind: EmbeddingKind) async throws -> [[Float]] {
        texts.map { text in
            var vector = [Float](repeating: 0, count: dimension)
            for token in BM25Index.tokenize(text) {
                let hash = token.unicodeScalars.reduce(UInt32(2166136261)) { ($0 ^ $1.value) &* 16777619 }
                vector[Int(hash % UInt32(dimension))] += 1
            }
            let norm = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
            return norm > 0 ? vector.map { $0 / norm } : vector
        }
    }
}
