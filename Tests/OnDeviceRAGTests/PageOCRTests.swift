#if os(macOS)
import AppKit
import PDFKit
import Testing
@testable import OnDeviceRAG

struct PageOCRTests {
    private let body = ["Lists are encoded as items between markers.", "Nested lists are allowed in this format.",
                        "Every item keeps its original order."]

    @Test func scannedFixtureHasNoTextLayer() {
        let pdf = FixturePDF.makeScanned(pages: [body.map(FixturePDF.Line.body)])
        #expect(PDFTextExtractor.lines(from: pdf).pagesWithoutText == [1])
    }

    @Test func readsTheTextOfAScannedPage() async throws {
        let pdf = FixturePDF.makeScanned(pages: [body.map(FixturePDF.Line.body)])
        let lines = try await PageOCR.lines(of: try #require(pdf.page(at: 0)), pageNumber: 4)
        let texts = lines.filter { !$0.isBlank }.map { $0.text.trimmingCharacters(in: .whitespaces) }
        #expect(texts == body)
        #expect(lines.allSatisfy { $0.isOCR && $0.page == 4 && !$0.isMonospaced })
    }

    @Test func headingIsLargerAndBodyLinesShareOneSize() async throws {
        let pdf = FixturePDF.makeScanned(pages: [[.heading("Chapter Two Lists")] + body.map(FixturePDF.Line.body)])
        let lines = try await PageOCR.lines(of: try #require(pdf.page(at: 0)), pageNumber: 1).filter { !$0.isBlank }
        let heading = try #require(lines.first { $0.text.contains("Chapter") })
        let bodyLines = lines.filter { !$0.text.contains("Chapter") }
        #expect(bodyLines.count == 3)
        #expect(bodyLines.allSatisfy { $0.fontSize == PageOCR.bodyFontSize })
        #expect(heading.fontSize >= 1.3 * PageOCR.bodyFontSize)
    }

    @Test func blankPageGivesNoText() async throws {
        let pdf = FixturePDF.makeScanned(pages: [[]])
        let lines = try await PageOCR.lines(of: try #require(pdf.page(at: 0)), pageNumber: 1)
        #expect(lines.allSatisfy { $0.isBlank })
    }

    @Test func rotatedScanIsReadUpright() async throws {
        let pdf = FixturePDF.makeScanned(pages: [body.map(FixturePDF.Line.body)])
        let page = try #require(pdf.page(at: 0))
        page.rotation = 90
        let texts = try await PageOCR.lines(of: page, pageNumber: 1).filter { !$0.isBlank }.map(\.text)
        #expect(texts.contains { $0.contains("Nested lists are allowed") })
    }

    @Test func fillReplacesOnlyEmptyPagesAndKeepsPageOrder() async throws {
        // Page 1 typed, page 2 scanned, page 3 blank scan.
        let typed = FixturePDF.make(pages: [[.heading("Typed"), .body("This page has a real text layer with enough characters.")]])
        let scanned = FixturePDF.makeScanned(pages: [body.map(FixturePDF.Line.body), []])
        for index in 0..<scanned.pageCount {
            typed.insert(try #require(scanned.page(at: index)), at: typed.pageCount)
        }
        let extracted = PDFTextExtractor.lines(from: typed)
        #expect(extracted.pagesWithoutText == [2, 3])

        let filled = try await PageOCR.fill(extracted, in: typed)
        #expect(filled.pagesWithoutText == [3])
        #expect(filled.lines.map(\.page) == filled.lines.map(\.page).sorted())
        #expect(filled.lines.filter { $0.page == 1 }.allSatisfy { !$0.isOCR })
        #expect(filled.lines.filter { $0.page == 2 }.allSatisfy { $0.isOCR })
        #expect(filled.lines.contains { $0.page == 2 && $0.text.contains("Every item keeps its original order") })
    }
}
#endif
