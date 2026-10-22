import Foundation
import PDFKit

public struct IngestSettings: Sendable, Hashable {
    public var chunkLength = 400
    public var sectionWindowLength = 1500
    public init() {}
}

public struct IngestedDocument: Sendable {
    public let info: DocumentInfo
    public let sections: [Section]
    public let chunks: [Chunk]
}

public enum IngestError: Error, Equatable {
    case unreadablePDF(URL)
}

/// Turns a PDF (or plain text) into sections and chunks. No embedding happens here.
public enum DocumentIngestor {
    static let mathSymbols: Set<Character> = Set("=+−×÷^√∑∏∫≤≥≠≈∞πθλμσΔ∂±⇒⇔∨∧∈∉⊂⊆∪∩²³")
    static let mathHeavyShare = 0.05

    public static func ingestPDF(at url: URL, title: String? = nil, id: DocumentID = UUID().uuidString,
                                 settings: IngestSettings = IngestSettings()) throws -> IngestedDocument {
        guard let document = PDFDocument(url: url) else { throw IngestError.unreadablePDF(url) }
        let pdfTitle = document.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String
        let fallbackTitle = pdfTitle.flatMap { $0.isEmpty ? nil : $0 } ?? url.deletingPathExtension().lastPathComponent
        return ingest(pdf: document, title: title ?? fallbackTitle, id: id, settings: settings)
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
                        pagesWithoutText: [], mathHeavyPages: [], settings: settings)
    }

    static func ingest(pdf document: PDFDocument, title: String, id: DocumentID, settings: IngestSettings) -> IngestedDocument {
        let extracted = PDFTextExtractor.lines(from: document)
        let detected = SectionDetector.detect(lines: extracted.lines, outline: PDFTextExtractor.outline(of: document),
                                              documentTitle: title, windowLength: settings.sectionWindowLength)
        return assemble(sections: detected.sections, source: detected.source, title: title, id: id,
                        pageCount: document.pageCount, pagesWithoutText: extracted.pagesWithoutText,
                        mathHeavyPages: mathHeavyPages(extracted.lines), settings: settings)
    }

    /// Sections without any chunk are dropped, so `Section.index` is always its position in `sections`.
    private static func assemble(sections detected: [DetectedSection], source: SectionSource, title: String, id: DocumentID,
                                 pageCount: Int, pagesWithoutText: [Int], mathHeavyPages: [Int],
                                 settings: IngestSettings) -> IngestedDocument {
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
                                pagesWithoutText: pagesWithoutText, mathHeavyPages: mathHeavyPages)
        return IngestedDocument(info: info, sections: sections, chunks: chunks)
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
