import Foundation

/// File I/O for document records: one `.ragdoc` per user document in a storage directory,
/// and single-file `.ragindex` bundles for shipped documents.
enum DocumentStore {
    static let documentExtension = "ragdoc"

    static func fileURL(for id: DocumentID, in directory: URL) -> URL {
        let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
        let name = id.addingPercentEncoding(withAllowedCharacters: allowed) ?? UUID().uuidString
        return directory.appending(path: "\(name).\(documentExtension)")
    }

    /// Atomic: a crash mid-write leaves the previous file intact.
    static func save(_ record: DocumentRecord, in directory: URL) throws {
        try encoder().encode(record).write(to: fileURL(for: record.info.id, in: directory), options: .atomic)
    }

    static func delete(_ id: DocumentID, in directory: URL) throws {
        let url = fileURL(for: id, in: directory)
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    /// Every readable record, oldest first; unreadable files are described in `issues`.
    static func loadAll(in directory: URL) -> (records: [DocumentRecord], issues: [String]) {
        let files = ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == documentExtension }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var records: [DocumentRecord] = []
        var issues: [String] = []
        for file in files {
            do {
                let data = try Data(contentsOf: file)
                try checkVersion(of: data, file: file)
                records.append(try JSONDecoder().decode(DocumentRecord.self, from: data))
            } catch RAGError.unsupportedFormat(let version, let name) {
                issues.append("\(name): unsupported format version \(version) (this package reads version \(DocumentRecord.currentFormatVersion))")
            } catch {
                issues.append("\(file.lastPathComponent): unreadable (\(error.localizedDescription))")
            }
        }
        records.sort { $0.addedAt != $1.addedAt ? $0.addedAt < $1.addedAt : $0.info.id < $1.info.id }
        return (records, issues)
    }

    static func writeIndex(_ records: [DocumentRecord], to url: URL) throws {
        let file = IndexFile(formatVersion: DocumentRecord.currentFormatVersion, documents: records)
        try encoder().encode(file).write(to: url, options: .atomic)
    }

    static func readIndex(at url: URL) throws -> [DocumentRecord] {
        let data = try Data(contentsOf: url)
        try checkVersion(of: data, file: url)
        return try JSONDecoder().decode(IndexFile.self, from: data).documents
    }

    private static func checkVersion(of data: Data, file: URL) throws {
        let version = try JSONDecoder().decode(FormatProbe.self, from: data).formatVersion
        guard version == DocumentRecord.currentFormatVersion else {
            throw RAGError.unsupportedFormat(version: version, file: file.lastPathComponent)
        }
    }

    /// Sorted keys so the same input always produces the same bytes (reproducible prebuilt indexes).
    private static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}
