/// One line of extracted text with the font facts section detection needs.
struct TextLine: Sendable, Hashable, Codable {
    var text: String
    /// 1-based page number.
    var page: Int
    var fontSize: Double
    var isBold: Bool
    var isMonospaced: Bool
    /// Read with OCR from a picture of the page, not from the PDF's text layer.
    var isOCR: Bool

    init(_ text: String, page: Int = 1, fontSize: Double = 11, isBold: Bool = false, isMonospaced: Bool = false,
         isOCR: Bool = false) {
        self.text = text
        self.page = page
        self.fontSize = fontSize
        self.isBold = isBold
        self.isMonospaced = isMonospaced
        self.isOCR = isOCR
    }

    var isBlank: Bool { text.allSatisfy(\.isWhitespace) }
}

extension TextLine {
    enum CodingKeys: String, CodingKey { case text, page, fontSize, isBold, isMonospaced, isOCR }

    /// Lines saved before OCR existed (milestone 2a files) have no `isOCR` key.
    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(try container.decode(String.self, forKey: .text),
                  page: try container.decode(Int.self, forKey: .page),
                  fontSize: try container.decode(Double.self, forKey: .fontSize),
                  isBold: try container.decode(Bool.self, forKey: .isBold),
                  isMonospaced: try container.decode(Bool.self, forKey: .isMonospaced),
                  isOCR: try container.decodeIfPresent(Bool.self, forKey: .isOCR) ?? false)
    }
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
