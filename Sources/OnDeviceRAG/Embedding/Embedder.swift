public enum EmbeddingKind: Sendable {
    /// A search question (bge adds its query instruction).
    case query
    /// A chunk of a document.
    case passage
}

/// Turns texts into L2-normalized vectors. Swap in another implementation to use a different model.
public protocol Embedder: Sendable {
    /// Stable ID of the model + export (stored in indexes from milestone 2 on).
    var identifier: String { get }
    var dimension: Int { get }
    func embed(_ texts: [String], as kind: EmbeddingKind) async throws -> [[Float]]
}
