@testable import OnDeviceRAG

actor EmbedCounter {
    private(set) var passages = 0
    func add(_ count: Int) { passages += count }
}

/// FakeEmbedder that counts how many passages it embedded and has a configurable identifier.
struct CountingEmbedder: Embedder {
    var identifier = "fake-bag-of-words"
    let dimension = 64
    let counter: EmbedCounter

    func embed(_ texts: [String], as kind: EmbeddingKind) async throws -> [[Float]] {
        if kind == .passage { await counter.add(texts.count) }
        return try await FakeEmbedder().embed(texts, as: kind)
    }
}
