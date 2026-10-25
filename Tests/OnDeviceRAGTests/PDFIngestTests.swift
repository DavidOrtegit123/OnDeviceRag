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

    @Test func unreadableFileThrows() async {
        let url = URL(fileURLWithPath: "/nonexistent/file.pdf")
        await #expect(throws: IngestError.unreadablePDF(url)) { try await DocumentIngestor.ingestPDF(at: url) }
    }
}
#endif
