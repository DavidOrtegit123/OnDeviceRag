#if os(macOS)
import Foundation
import PDFKit
import Testing
@testable import OnDeviceRAG

struct PDFIngestTests {
    @Test func extractsLinesWithFontFacts() throws {
        let pdf = FixturePDF.make(pages: [[.heading("Integers"), .body("Integers are written as i<n>."), .code("    encode(5)")]])
        let result = PDFTextExtractor.lines(from: pdf)
        let texts = result.lines.map { $0.text.trimmingCharacters(in: .whitespaces) }
        #expect(texts == ["Integers", "Integers are written as i<n>.", "encode(5)"])
        try #require(result.lines.count == 3)
        #expect(result.lines[0].fontSize > result.lines[1].fontSize)
        #expect(result.lines[0].isBold)
        #expect(result.lines[2].isMonospaced)
        #expect(!result.lines[1].isMonospaced)
    }

    @Test func headingsBecomeSectionsAndCodeStaysOneChunk() async throws {
        let pdf = FixturePDF.make(pages: [
            [.heading("Integers"), .body("Integers are written as i<n>."), .body("Negative numbers keep their sign.")],
            [.heading("Lists"), .body("Lists are written as l<items>e."), .code("encode([1, 2])"), .code("# -> li1ei2ee")],
        ])
        let document = try await DocumentIngestor.ingest(pdf: pdf, title: "ZORP", id: "z", settings: IngestSettings())
        #expect(document.info.sectionSource == .headings)
        #expect(document.sections.map(\.title) == ["Integers", "Lists"])
        #expect(document.sections.last?.pages == PageRange(first: 2, last: 2))
        let code = document.chunks.filter(\.isCode)
        #expect(code.count == 1)
        #expect(code.first?.text.contains("encode([1, 2])\n") == true)
        #expect(document.info.pageCount == 2)
        #expect(document.info.chunkCount == document.chunks.count)
    }

    @Test func outlineWinsOverHeadings() async throws {
        let pdf = FixturePDF.make(pages: [[.heading("Alpha"), .body("First body sentence here.")],
                                          [.heading("Beta"), .body("Second body sentence here.")]],
                                  outline: [(title: "Alpha", page: 1), (title: "Beta", page: 2)])
        let document = try await DocumentIngestor.ingest(pdf: pdf, title: "Doc", id: "d", settings: IngestSettings())
        #expect(document.info.sectionSource == .outline)
        #expect(document.sections.map(\.title) == ["Alpha", "Beta"])
        #expect(document.chunks.map(\.text) == ["First body sentence here.", "Second body sentence here."])
    }

    @Test func emptyPageIsReportedAsWithoutText() async throws {
        // Page 1 needs ≥ 50 non-whitespace characters to count as a text page.
        let pdf = FixturePDF.make(pages: [[.body("Some real text on the first page, clearly long enough to count as text.")], []])
        let document = try await DocumentIngestor.ingest(pdf: pdf, title: "Doc", id: "d", settings: IngestSettings())
        #expect(document.info.pagesWithoutText == [2])
    }

    @Test func mathHeavyPageIsFlagged() async throws {
        let pdf = FixturePDF.make(pages: [[.body("x² + 2x − 3 = 0 ⇒ x = −3 ∨ x = 1, √9 = 3, ∑ k ≤ n")],
                                          [.body("A page of plain words without any formulas at all here.")]])
        let document = try await DocumentIngestor.ingest(pdf: pdf, title: "Doc", id: "d", settings: IngestSettings())
        #expect(document.info.mathHeavyPages == [1])
    }

    @Test func textIngestUsesHashHeadingsAndFences() {
        let text = """
        Intro line.

        ## Integers
        Integers are i<n>.

        ## Lists
        Lists are l<items>e.
        ```
        encode([1])
        ```
        """
        let document = DocumentIngestor.ingestText(text, title: "ZORP", id: "z")
        #expect(document.info.sectionSource == .provided)
        #expect(document.sections.map(\.title) == ["ZORP", "Integers", "Lists"])
        #expect(document.chunks.filter(\.isCode).map(\.text) == ["encode([1])"])
    }

    /// A PDF printed from a plain-text file is entirely monospaced: the font says nothing about code.
    private let plainTextRules = (1...12).map { "Rule \($0) says the encoder writes every value as plain text before sending it." }

    @Test func allMonospacedPDFIsChunkedAsProse() async throws {
        let pdf = FixturePDF.make(pages: [plainTextRules.map(FixturePDF.Line.code)])
        let document = try await DocumentIngestor.ingest(pdf: pdf, title: "Rules", id: "r", settings: IngestSettings())
        #expect(document.chunks.count >= 3)
        #expect(document.chunks.allSatisfy { !$0.isCode && $0.text.count <= 400 })
        #expect(document.chunks.first?.text.hasPrefix("Rule 1 says") == true)
    }

    @Test func allMonospacedDocumentReingestsIdentically() async throws {
        let pdf = FixturePDF.make(pages: [plainTextRules.map(FixturePDF.Line.code)])
        let original = try await DocumentIngestor.ingest(pdf: pdf, title: "Rules", id: "r", settings: IngestSettings())
        guard case let .pdf(lines, _, _, _) = original.source else {
            Issue.record("expected a PDF source")
            return
        }
        #expect(lines.allSatisfy { $0.isMonospaced })   // the source keeps the original font facts
        let again = DocumentIngestor.reingest(original.source, title: "Rules", id: "r", settings: IngestSettings())
        #expect(again.chunks == original.chunks)
    }

    @Test func codeListingInAMostlyProseDocumentStaysCode() async throws {
        let prose = plainTextRules.prefix(6).map(FixturePDF.Line.body)
        let listing = ["def encode(value):", "    return str(value)", "", "def decode(text):", "    return int(text)"]
            .filter { !$0.isEmpty }.map(FixturePDF.Line.code)
        let pdf = FixturePDF.make(pages: [Array(prose) + listing])
        let document = try await DocumentIngestor.ingest(pdf: pdf, title: "Rules", id: "r", settings: IngestSettings())
        let code = document.chunks.filter(\.isCode)
        #expect(code.count == 1)
        #expect(code.first?.text.contains("def decode(text):") == true)
    }

    @Test func unreadableFileThrows() async {
        let url = URL(fileURLWithPath: "/nonexistent/file.pdf")
        await #expect(throws: IngestError.unreadablePDF(url)) { try await DocumentIngestor.ingestPDF(at: url) }
    }
}
#endif
