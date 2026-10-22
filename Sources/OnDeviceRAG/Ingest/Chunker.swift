import Foundation
import NaturalLanguage

/// Turns a detected section into its full text and ≤ `maximumLength`-character chunks.
/// Prose is packed from whole sentences; each run of monospaced lines becomes one code chunk.
enum Chunker {
    /// A prose run (with each character's page) or a code block.
    private enum Block {
        case prose(text: String, pages: [Int])   // pages[i] = page of character i
        case code(text: String, pages: PageRange)
    }

    static func build(_ section: DetectedSection, documentID: DocumentID, sectionIndex: Int, maximumLength: Int) -> (section: Section, chunks: [Chunk])? {
        var pieces: [(text: String, pages: PageRange, isCode: Bool)] = []
        var sectionParts: [String] = []

        for block in makeBlocks(section.lines) {
            switch block {
            case let .code(text, pages):
                pieces.append((text, pages, true))
                sectionParts.append(text)
            case let .prose(text, pages):
                sectionParts.append(text)
                pieces += packSentences(text: text, pages: pages, maximumLength: maximumLength).map { ($0.text, $0.pages, false) }
            }
        }
        guard !pieces.isEmpty else { return nil }

        let chunks = pieces.enumerated().map { position, piece in
            Chunk(documentID: documentID, sectionIndex: sectionIndex, sectionTitle: section.title,
                  position: position, text: piece.text, pages: piece.pages, isCode: piece.isCode)
        }
        let pages = chunks.dropFirst().reduce(chunks[0].pages) { $0.union($1.pages) }
        let sectionValue = Section(documentID: documentID, index: sectionIndex, title: section.title,
                                   text: sectionParts.joined(separator: "\n\n"), pages: pages)
        return (sectionValue, chunks)
    }

    // MARK: - Blocks

    /// Wrapped lines are joined into prose (a trailing "-" before a lowercase letter is
    /// removed; otherwise a space); blank lines become paragraph breaks ("\n\n").
    private static func makeBlocks(_ lines: [TextLine]) -> [Block] {
        var blocks: [Block] = []
        var prose = ""
        var prosePages: [Int] = []
        var code: [TextLine] = []

        func closeProse() {
            if let start = prose.firstIndex(where: { !$0.isWhitespace }) {
                let trimmed = prose.trimmingCharacters(in: .whitespacesAndNewlines)
                let offset = prose.distance(from: prose.startIndex, to: start)
                blocks.append(.prose(text: trimmed, pages: Array(prosePages[offset..<(offset + trimmed.count)])))
            }
            prose = ""
            prosePages = []
        }
        func closeCode() {
            guard !code.isEmpty else { return }
            let pages = code.map(\.page)
            blocks.append(.code(text: code.map(\.text).joined(separator: "\n"),
                                pages: PageRange(first: pages.min()!, last: pages.max()!)))
            code = []
        }
        func appendProse(_ piece: String, page: Int) {
            prose += piece
            prosePages += Array(repeating: page, count: piece.count)
        }

        for line in lines {
            if line.isMonospaced && !line.isBlank {
                closeProse()
                code.append(line)
                continue
            }
            closeCode()
            if line.isBlank {
                if !prose.isEmpty && !prose.hasSuffix("\n\n") {
                    appendProse("\n\n", page: prosePages.last ?? line.page)
                }
                continue
            }
            let text = line.text.trimmingCharacters(in: .whitespaces)
            if prose.isEmpty || prose.hasSuffix("\n\n") {
                appendProse(text, page: line.page)
            } else if prose.hasSuffix("-"), text.first?.isLowercase == true {
                prose.removeLast()
                prosePages.removeLast()
                appendProse(text, page: line.page)
            } else {
                appendProse(" " + text, page: line.page)
            }
        }
        closeProse()
        closeCode()
        return blocks
    }

    // MARK: - Sentence packing

    private static func packSentences(text: String, pages: [Int], maximumLength: Int) -> [(text: String, pages: PageRange)] {
        var sentences: [(text: String, pages: PageRange)] = []
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let start = text.distance(from: text.startIndex, to: range.lowerBound)
            let raw = String(text[range])
            let leading = raw.prefix { $0.isWhitespace }.count
            let sentence = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !sentence.isEmpty else { return true }
            for piece in cutAtSpaces(sentence, maximumLength: maximumLength, offset: start + leading) {
                let slice = pages[piece.offset..<(piece.offset + piece.text.count)]
                sentences.append((piece.text, PageRange(first: slice.min()!, last: slice.max()!)))
            }
            return true
        }

        var chunks: [(text: String, pages: PageRange)] = []
        for sentence in sentences {
            if let last = chunks.last, last.text.count + 1 + sentence.text.count <= maximumLength {
                chunks[chunks.count - 1] = (last.text + " " + sentence.text, last.pages.union(sentence.pages))
            } else {
                chunks.append(sentence)
            }
        }
        return chunks
    }

    /// Splits text longer than the limit at the last space before it. `offset` is the text's
    /// character offset in the prose run (for page lookup).
    private static func cutAtSpaces(_ text: String, maximumLength: Int, offset: Int) -> [(text: String, offset: Int)] {
        var result: [(text: String, offset: Int)] = []
        var remaining = Substring(text)
        var position = offset
        while remaining.count > maximumLength {
            let window = remaining.prefix(maximumLength + 1)
            let cut = window.lastIndex(of: " ") ?? remaining.index(remaining.startIndex, offsetBy: maximumLength)
            let head = remaining[..<cut]
            result.append((String(head), position))
            let rest = remaining[cut...]
            let skipped = rest.prefix { $0 == " " }.count
            position += head.count + skipped
            remaining = rest.dropFirst(skipped)
        }
        if !remaining.isEmpty {
            result.append((String(remaining), position))
        }
        return result
    }
}
