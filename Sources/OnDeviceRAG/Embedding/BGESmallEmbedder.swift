import CoreAI
import Foundation
import Tokenizers

public enum EmbedderError: Error {
    case missingFunction(String)
    case missingOutput
    case missingTokenizer
}

/// bge-small-en-v1.5 (33M parameters, MIT) running through Core AI. The export has one
/// function per fixed input length (`embed_128/256/512`); texts are padded to the nearest one,
/// because flexible lengths cost ~2.5 s per new length on iPhone. Pooling and L2 normalization
/// are inside the model.
public actor BGESmallEmbedder: Embedder {
    public nonisolated let identifier = "bge-small-en-v1.5/float16/fixed-128-256-512"
    public nonisolated let dimension = 384

    static let queryPrefix = "Represent this sentence for searching relevant passages: "
    static let lengths = [128, 256, 512]
    static let separatorID: Int32 = 102  // [SEP]
    static let paddingID: Int32 = 0      // [PAD]

    private nonisolated let tokenizer: any Tokenizer
    private let functions: [Int: InferenceFunction]

    public init(modelURL: URL) async throws {
        guard let folder = Bundle.module.url(forResource: "bge-small-en-v1.5", withExtension: nil) else {
            throw EmbedderError.missingTokenizer
        }
        tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        let model = try await AIModel(contentsOf: modelURL)
        var loaded: [Int: InferenceFunction] = [:]
        for length in Self.lengths {
            guard let function = try model.loadFunction(named: "embed_\(length)") else {
                throw EmbedderError.missingFunction("embed_\(length)")
            }
            loaded[length] = function
        }
        functions = loaded
    }

    /// Token IDs with [CLS]/[SEP], truncated to 512 like Python's `truncation=True, max_length=512`.
    nonisolated func tokenIDs(for text: String, as kind: EmbeddingKind) -> [Int32] {
        let full = kind == .query ? Self.queryPrefix + text : text
        var ids = tokenizer.encode(text: full).map(Int32.init)
        if ids.count > 512 {
            ids = Array(ids.prefix(511)) + [Self.separatorID]
        }
        return ids
    }

    public func embed(_ texts: [String], as kind: EmbeddingKind) async throws -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        for text in texts {
            vectors.append(try await embed(ids: tokenIDs(for: text, as: kind)))
        }
        return vectors
    }

    private func embed(ids: [Int32]) async throws -> [Float] {
        let length = Self.lengths.first { $0 >= ids.count } ?? 512
        let padding = length - ids.count
        let paddedIDs = ids + Array(repeating: Self.paddingID, count: padding)
        let mask = Array(repeating: Int32(1), count: ids.count) + Array(repeating: Int32(0), count: padding)
        guard let function = functions[length] else { throw EmbedderError.missingFunction("embed_\(length)") }
        var outputs = try await function.run(inputs: [
            "input_ids": NDArray(scalars: paddedIDs, shape: [1, length]),
            "attention_mask": NDArray(scalars: mask, shape: [1, length]),
        ])
        guard let output = outputs.remove("embedding")?.ndArray else { throw EmbedderError.missingOutput }
        let count = output.shape.reduce(1, *)
        var result: [Float] = []
        if output.scalarType == .float16 {
            output.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
                result = UnsafeBufferPointer(start: pointer, count: count).map(Float.init)
            }
        } else {
            output.view(as: Float.self).withUnsafePointer { pointer, _, _ in
                result = Array(UnsafeBufferPointer(start: pointer, count: count))
            }
        }
        return result
    }
}
