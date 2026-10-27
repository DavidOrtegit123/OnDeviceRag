import Foundation
import PDFKit

public struct IngestSettings: Sendable, Hashable, Codable {
    public var chunkLength = 400
    public var sectionWindowLength = 1500
    /// Version of the reading rules (headings, page headers/footers…). A stored document read with
    /// other rules differs from the current settings, so it's re-read from its source once on open.
    /// 1 = up to 0.2.2; 2 = 0.2.3 layout rules.
    public internal(set) var readingRules = IngestSettings.currentReadingRules
    static let currentReadingRules = 2

    public init() {}

    private enum CodingKeys: String, CodingKey { case chunkLength, sectionWindowLength, readingRules }

    /// Settings saved before `readingRules` existed decode as version 1.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        chunkLength = try container.decode(Int.self, forKey: .chunkLength)
        sectionWindowLength = try container.decode(Int.self, forKey: .sectionWindowLength)
        readingRules = try container.decodeIfPresent(Int.self, forKey: .readingRules) ?? 1
    }
}

/// What a document was built from — kept so it can be re-chunked later without the original file.
enum DocumentSource: Sendable, Hashable, Codable {
    case pdf(lines: [TextLine], outline: [OutlineEntry], pageCount: Int, pagesWithoutText: [Int])
    case text(String)
}

public struct IngestedDocument: Sendable {
    public let info: DocumentInfo
    public let sections: [Section]
    public let chunks: [Chunk]
    let source: DocumentSource
}

public enum IngestError: Error, Equatable {
    case unreadablePDF(URL)
}

/// Turns a PDF (or plain text) into sections and chunks. No embedding happens here.
public enum DocumentIngestor {
    static let mathSymbols: Set<Character> = Set("=+−×÷^√∑∏∫≤≥≠≈∞πθλμσΔ∂±⇒⇔∨∧∈∉⊂⊆∪∩²³")
    static let mathHeavyShare = 0.05
    /// Monospace marks code only when it's the exception: in a document at least this monospaced
    /// (e.g. a plain-text file printed to PDF) every line is treated as ordinary text.
    static let mostlyMonospacedShare = 0.8
    /// A first or last page line recurring on at least this share of pages is a running header/footer.
    static let runningLineShare = 0.5

    /// Reads a PDF; pages without a text layer (scans) are read with OCR unless `ocr` is false.
    public static func ingestPDF(at url: URL, title: String? = nil, id: DocumentID = UUID().uuidString,
                                 settings: IngestSettings = IngestSettings(), ocr: Bool = true) async throws -> IngestedDocument {
        guard let document = PDFDocument(url: url) else { throw IngestError.unreadablePDF(url) }
        let pdfTitle = document.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String
        let fallbackTitle = pdfTitle.flatMap { $0.isEmpty ? nil : $0 } ?? url.deletingPathExtension().lastPathComponent
        return try await ingest(pdf: document, title: title ?? fallbackTitle, id: id, settings: settings, ocr: ocr)
    }

    /// Plain text: `## ` lines start sections (text before the first one is titled `title`);
    /// lines between ``` fences are code.
    public static func ingestText(_ text: String, title: String, id: DocumentID = UUID().uuidString,
                                  settings: IngestSettings = IngestSettings()) -> IngestedDocument {
        var sections = [DetectedSection(title: title, lines: [])]
        var inCode = false
        for raw in text.components(separatedBy: "\n") {
            if raw.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                inCode.toggle()
                continue
            }
            if !inCode, raw.hasPrefix("## ") {
                sections.append(DetectedSection(title: raw.dropFirst(3).trimmingCharacters(in: .whitespaces), lines: []))
                continue
            }
            sections[sections.count - 1].lines.append(TextLine(raw, isMonospaced: inCode))
        }
        return assemble(sections: sections, source: .provided, title: title, id: id, pageCount: 1,
                        pagesWithoutText: [], ocrPages: [], mathHeavyPages: [], settings: settings, documentSource: .text(text))
    }

    static func ingest(pdf document: PDFDocument, title: String, id: DocumentID, settings: IngestSettings,
                       ocr: Bool = true) async throws -> IngestedDocument {
        var extracted = PDFTextExtractor.lines(from: document)
        if ocr, !extracted.pagesWithoutText.isEmpty {
            extracted = try await PageOCR.fill(extracted, in: document)
        }
        return ingest(lines: extracted.lines, outline: PDFTextExtractor.outline(of: document), pageCount: document.pageCount,
                      pagesWithoutText: extracted.pagesWithoutText, title: title, id: id, settings: settings)
    }

    /// Lines → sections → chunks; shared by fresh PDFs and stored PDF sources.
    static func ingest(lines: [TextLine], outline: [OutlineEntry], pageCount: Int, pagesWithoutText: [Int],
                       title: String, id: DocumentID, settings: IngestSettings) -> IngestedDocument {
        // The source keeps the original lines; these rules are re-applied on every re-chunk.
        let bodyLines = removingRunningHeadersAndFooters(lines)
        let readableLines = isMostlyMonospaced(bodyLines)
            ? bodyLines.map { line in var prose = line; prose.isMonospaced = false; return prose }
            : bodyLines
        let detected = SectionDetector.detect(lines: readableLines, outline: outline, documentTitle: title,
                                              windowLength: settings.sectionWindowLength)
        return assemble(sections: detected.sections, source: detected.source, title: title, id: id,
                        pageCount: pageCount, pagesWithoutText: pagesWithoutText,
                        ocrPages: Set(lines.filter(\.isOCR).map(\.page)).sorted(),
                        mathHeavyPages: mathHeavyPages(lines), settings: settings,
                        documentSource: .pdf(lines: lines, outline: outline, pageCount: pageCount, pagesWithoutText: pagesWithoutText))
    }

    /// Rebuilds sections and chunks from a stored source (e.g. after chunk settings changed).
    static func reingest(_ source: DocumentSource, title: String, id: DocumentID, settings: IngestSettings) -> IngestedDocument {
        switch source {
        case let .pdf(lines, outline, pageCount, pagesWithoutText):
            ingest(lines: lines, outline: outline, pageCount: pageCount, pagesWithoutText: pagesWithoutText,
                   title: title, id: id, settings: settings)
        case let .text(text):
            ingestText(text, title: title, id: id, settings: settings)
        }
    }

    /// Sections without any chunk are dropped, so `Section.index` is always its position in `sections`.
    private static func assemble(sections detected: [DetectedSection], source: SectionSource, title: String, id: DocumentID,
                                 pageCount: Int, pagesWithoutText: [Int], ocrPages: [Int], mathHeavyPages: [Int],
                                 settings: IngestSettings, documentSource: DocumentSource) -> IngestedDocument {
        var sections: [Section] = []
        var chunks: [Chunk] = []
        for candidate in detected {
            guard let built = Chunker.build(candidate, documentID: id, sectionIndex: sections.count,
                                            maximumLength: settings.chunkLength) else { continue }
            sections.append(built.section)
            chunks += built.chunks
        }
        let info = DocumentInfo(id: id, title: title, pageCount: pageCount, sectionCount: sections.count,
                                chunkCount: chunks.count, sectionSource: source,
                                pagesWithoutText: pagesWithoutText, ocrPages: ocrPages, mathHeavyPages: mathHeavyPages)
        return IngestedDocument(info: info, sections: sections, chunks: chunks, source: documentSource)
    }

    /// Running headers and footers: a page's first or last non-blank line, no larger than the body
    /// text, whose text (digit runs read as "#", so "page 2" matches "page 61") and font size recur
    /// as a first or last line on at least `runningLineShare` of the pages with text, and on at least
    /// 3 of them, is removed. The size limit keeps per-page headings ("Exercise 1", "Exercise 2").
    static func removingRunningHeadersAndFooters(_ lines: [TextLine]) -> [TextLine] {
        struct Key: Hashable { var text: String; var fontSize: Double }
        func key(_ line: TextLine) -> Key {
            let text = line.text.replacingOccurrences(of: "[0-9]+", with: "#", options: .regularExpression)
            return Key(text: SectionDetector.normalize(text), fontSize: line.fontSize)
        }
        var edges: [Int: Set<Int>] = [:]  // page → indices of its first and last non-blank line
        for (index, line) in lines.enumerated() where !line.isBlank {
            if edges[line.page] == nil { edges[line.page] = [index] }
        }
        for (index, line) in lines.enumerated().reversed() where !line.isBlank {
            if edges[line.page]?.count == 1, let first = edges[line.page]?.first, first != index {
                edges[line.page]?.insert(index)
            }
        }
        let pagesWithText = edges.count
        guard pagesWithText >= 3,
              let bodySize = SectionDetector.weightedMedianFontSize(lines.filter { !$0.isBlank && !$0.isMonospaced })
        else { return lines }
        edges = edges.mapValues { $0.filter { lines[$0].fontSize <= bodySize } }
        var pagesPerKey: [Key: Set<Int>] = [:]
        for (page, indices) in edges {
            for index in indices { pagesPerKey[key(lines[index]), default: []].insert(page) }
        }
        let needed = max(3, Int((Double(pagesWithText) * runningLineShare).rounded(.up)))
        let running = Set(edges.values.flatMap { $0 }.filter { pagesPerKey[key(lines[$0])]!.count >= needed })
        guard !running.isEmpty else { return lines }
        return lines.enumerated().filter { !running.contains($0.offset) }.map(\.element)
    }

    private static func isMostlyMonospaced(_ lines: [TextLine]) -> Bool {
        let total = PDFTextExtractor.visibleCharacterCount(lines)
        guard total > 0 else { return false }
        let monospaced = PDFTextExtractor.visibleCharacterCount(lines.filter(\.isMonospaced))
        return Double(monospaced) / Double(total) >= mostlyMonospacedShare
    }

    private static func mathHeavyPages(_ lines: [TextLine]) -> [Int] {
        let byPage = Dictionary(grouping: lines, by: \.page)
        return byPage.keys.sorted().filter { page in
            let characters = byPage[page, default: []].flatMap { $0.text.filter { !$0.isWhitespace } }
            guard !characters.isEmpty else { return false }
            let math = characters.filter { mathSymbols.contains($0) }.count
            return Double(math) / Double(characters.count) > mathHeavyShare
        }
    }
}
