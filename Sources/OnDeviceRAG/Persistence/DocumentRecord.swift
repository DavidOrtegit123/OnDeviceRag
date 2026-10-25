import Foundation

public enum RAGError: Error, Equatable {
    /// Prebuilt (shipped) documents are read-only.
    case prebuiltDocument(DocumentID)
    /// No document with this ID is loaded.
    case unknownDocument(DocumentID)
    case unsupportedFormat(version: Int, file: String)
}

/// One document as stored on disk: everything needed to search it, plus its source for re-chunking.
struct DocumentRecord: Codable, Sendable {
    static let currentFormatVersion = 1

    var formatVersion: Int
    var embedder: String
    var dimension: Int
    var settings: IngestSettings
    var addedAt: Date
    var info: DocumentInfo
    var sections: [Section]
    var chunks: [Chunk]
    var source: DocumentSource
    /// Float16, little-endian, `chunks.count × dimension` values in chunk order (base64 in JSON).
    var vectors: Data

    init(document: IngestedDocument, vectors: [[Float]], embedder: String, dimension: Int, settings: IngestSettings, addedAt: Date) {
        formatVersion = Self.currentFormatVersion
        self.embedder = embedder
        self.dimension = dimension
        self.settings = settings
        self.addedAt = addedAt
        info = document.info
        sections = document.sections
        chunks = document.chunks
        source = document.source
        self.vectors = Self.encode(vectors)
    }

    var document: IngestedDocument {
        IngestedDocument(info: info, sections: sections, chunks: chunks, source: source)
    }

    /// nil when the stored bytes don't match `chunks.count × dimension` Float16 values.
    func decodedVectors() -> [[Float]]? {
        let scalarCount = chunks.count * dimension
        let size = MemoryLayout<Float16>.size
        guard dimension > 0, vectors.count == scalarCount * size else { return nil }
        let halves: [Float16] = vectors.withUnsafeBytes { raw in
            (0..<scalarCount).map { raw.loadUnaligned(fromByteOffset: $0 * size, as: Float16.self) }
        }
        return stride(from: 0, to: scalarCount, by: dimension).map { start in
            halves[start..<(start + dimension)].map(Float.init)
        }
    }

    static func encode(_ vectors: [[Float]]) -> Data {
        let halves = vectors.flatMap { $0 }.map(Float16.init)
        return halves.withUnsafeBytes { Data($0) }
    }
}

/// A shipped, read-only bundle of documents (what `rag-index build` writes).
struct IndexFile: Codable, Sendable {
    var formatVersion: Int
    var documents: [DocumentRecord]
}

/// Reads only the version number, so a newer format is reported instead of failing to decode.
struct FormatProbe: Decodable {
    var formatVersion: Int
}
