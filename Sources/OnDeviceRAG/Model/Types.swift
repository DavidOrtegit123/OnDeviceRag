import Foundation

public typealias DocumentID = String

/// 1-based, inclusive page numbers.
public struct PageRange: Sendable, Hashable, Codable {
    public let first: Int
    public let last: Int

    public init(first: Int, last: Int) {
        self.first = min(first, last)
        self.last = max(first, last)
    }

    public func union(_ other: PageRange) -> PageRange {
        PageRange(first: min(first, other.first), last: max(last, other.last))
    }

    /// "p. 3" or "pp. 3–6".
    public var label: String {
        first == last ? "p. \(first)" : "pp. \(first)–\(last)"
    }
}

/// How a document's sections were found.
public enum SectionSource: String, Sendable, Codable {
    /// The PDF's table of contents.
    case outline
    /// Large or bold short lines.
    case headings
    /// Fixed ~1,500-character windows (no structure found).
    case windows
    /// `## ` headings in text given by the app.
    case provided
}

public struct DocumentInfo: Sendable, Hashable, Codable {
    public let id: DocumentID
    public let title: String
    public let pageCount: Int
    public let sectionCount: Int
    public let chunkCount: Int
    public let sectionSource: SectionSource
    /// Pages with < 50 non-whitespace characters (probably scanned; OCR comes in milestone 2).
    public let pagesWithoutText: [Int]
    /// Pages where math symbols are > 5% of non-whitespace characters (extraction may be garbled).
    public let mathHeavyPages: [Int]
}

public struct Section: Sendable, Hashable, Codable {
    public let documentID: DocumentID
    public let index: Int
    public let title: String
    /// Full section text: prose paragraphs and code blocks separated by blank lines.
    public let text: String
    public let pages: PageRange
}

public struct Chunk: Sendable, Hashable, Codable {
    public let documentID: DocumentID
    public let sectionIndex: Int
    public let sectionTitle: String
    /// 0-based position within its section.
    public let position: Int
    public let text: String
    public let pages: PageRange
    public let isCode: Bool

    public var id: String { "\(documentID)#\(sectionIndex).\(position)" }

    /// What gets embedded and keyword-searched: the heading gives a short chunk context.
    public var indexedText: String { "\(sectionTitle): \(text)" }
}

public struct RetrievedChunk: Sendable, Hashable {
    public let chunk: Chunk
    public let documentTitle: String
    public let fusedScore: Double
    /// 1-based rank in the dense (embedding) ranking.
    public let denseRank: Int?
    /// 1-based rank in the BM25 ranking; nil if the chunk shares no word with the query.
    public let keywordRank: Int?
}

public struct ContextSection: Sendable, Hashable {
    public let documentID: DocumentID
    public let documentTitle: String
    public let sectionIndex: Int
    public let title: String
    public let pages: PageRange
    /// False when only a window of chunks around the match fit the budget.
    public let isWholeSection: Bool
}

public struct RAGContext: Sendable, Hashable {
    /// Ready to insert under "Reference material:".
    public let text: String
    public let sections: [ContextSection]
    public let tokenCount: Int
    public let tokenCountIsEstimate: Bool
}

public struct TokenCount: Sendable, Hashable {
    public let value: Int
    public let isEstimate: Bool

    public init(value: Int, isEstimate: Bool) {
        self.value = value
        self.isEstimate = isEstimate
    }
}
