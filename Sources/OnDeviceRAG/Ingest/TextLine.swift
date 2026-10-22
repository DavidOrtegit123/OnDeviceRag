/// One line of extracted text with the font facts section detection needs.
struct TextLine: Sendable, Hashable, Codable {
    var text: String
    /// 1-based page number.
    var page: Int
    var fontSize: Double
    var isBold: Bool
    var isMonospaced: Bool

    init(_ text: String, page: Int = 1, fontSize: Double = 11, isBold: Bool = false, isMonospaced: Bool = false) {
        self.text = text
        self.page = page
        self.fontSize = fontSize
        self.isBold = isBold
        self.isMonospaced = isMonospaced
    }

    var isBlank: Bool { text.allSatisfy(\.isWhitespace) }
}

/// One table-of-contents entry from a PDF outline.
struct OutlineEntry: Sendable, Hashable, Codable {
    var title: String
    /// 1-based page number.
    var page: Int
}

/// A titled run of lines, before chunking.
struct DetectedSection: Sendable, Hashable {
    var title: String
    var lines: [TextLine]
}
