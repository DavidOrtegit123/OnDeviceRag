import Foundation
import Testing
@testable import OnDeviceRAG

private struct ReferenceItem: Decodable {
    let kind: String
    let text: String
    let ids: [Int32]
    let vector: [Float]
}

private func loadReference() throws -> [ReferenceItem] {
    let url = try #require(Bundle.module.url(forResource: "bge_reference", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode([ReferenceItem].self, from: Data(contentsOf: url))
}

@Suite(.enabled(if: ModelLocation.isAvailable, "bge-small model not found; set ONDEVICE_RAG_MODEL"), .serialized)
struct BGESmallEmbedderTests {
    @Test func tokenIDsMatchPython() async throws {
        let embedder = try await BGESmallEmbedder(modelURL: ModelLocation.url)
        for item in try loadReference() {
            let ids = embedder.tokenIDs(for: item.text, as: item.kind == "query" ? .query : .passage)
            #expect(ids == item.ids, "token IDs differ for: \(item.text.prefix(40))")
        }
    }

    @Test func vectorsMatchPyTorch() async throws {
        let embedder = try await BGESmallEmbedder(modelURL: ModelLocation.url)
        #expect(embedder.dimension == 384)
        for item in try loadReference() {
            let vector = try await embedder.embed([item.text], as: item.kind == "query" ? .query : .passage)[0]
            let cosine = zip(vector, item.vector).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            #expect(cosine >= 0.999, "cosine \(cosine) for: \(item.text.prefix(40))")
        }
    }
}
