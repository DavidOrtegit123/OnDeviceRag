import Foundation

/// Where the (git-ignored) bge-small Core AI model lives on this Mac.
enum ModelLocation {
    static let url: URL = {
        if let path = ProcessInfo.processInfo.environment["ONDEVICE_RAG_MODEL"] {
            return URL(fileURLWithPath: path)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Developer/coreai-models/exports/bge_small_buckets_float16.aimodel")
    }()

    static var isAvailable: Bool { FileManager.default.fileExists(atPath: url.path) }
}
