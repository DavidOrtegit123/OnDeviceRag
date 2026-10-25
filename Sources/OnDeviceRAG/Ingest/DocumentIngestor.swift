import Foundation
import PDFKit

public struct IngestSettings: Sendable, Hashable, Codable {
    public var chunkLength = 400
    public var sectionWindowLength = 1500
    public init() {}
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
                        pagesWithoutText: [], ocrPages: [], mathHeavyPages: [], settings: settings, documentSource: .text(text))
    }

    static func ingest(pdf document: PDFDocument, title: String, id: DocumentID, settings: IngestSettings) -> IngestedDocument {
        let extracted = PDFTextExtractor.lines(from: document)
        return ingest(lines: extracted.lines, outline: PDFTextExtractor.outline(of: document), pageCount: document.pageCount,
                      pagesWithoutText: extracted.pagesWithoutText, title: title, id: id, settings: settings)
    }

    /// Lines → sections → chunks; shared by fresh PDFs and stored PDF sources.
    static func ingest(lines: [TextLine], outline: [OutlineEntry], pageCount: Int, pagesWithoutText: [Int],
                       title: String, id: DocumentID, settings: IngestSettings) -> IngestedDocument {
        let detected = SectionDetector.detect(lines: lines, outline: outline, documentTitle: title,
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
