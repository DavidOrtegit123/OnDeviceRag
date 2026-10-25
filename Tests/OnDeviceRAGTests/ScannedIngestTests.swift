#if os(macOS)
import AppKit
import Foundation
import PDFKit
import Testing
@testable import OnDeviceRAG

private func zorpScan() -> [[FixturePDF.Line]] {
    [[
        FixturePDF.Line(text: "Encoding Lists", font: NSFont(name: "Helvetica-Bold", size: 20)!),
        .body("Lists are encoded as items between markers."),
        .body("Nested lists are allowed in this format."),
        .body("Every item keeps its original order."),
        FixturePDF.Line(text: "Encoding Numbers", font: NSFont(name: "Helvetica-Bold", size: 20)!),
        .body("Numbers are written with a leading marker."),
        .body("Negative numbers keep their minus sign."),
    ]]
}

private func writeTemporary(_ pdf: PDFDocument) throws -> URL {
    let url = FileManager.default.temporaryDirectory.appending(path: "scan-\(UUID().uuidString).pdf")
    try #require(pdf.write(to: url))
    return url
}

struct ScannedIngestTests {
    @Test func scannedPDFBecomesSectionsAndChunks() async throws {
        let document = try await DocumentIngestor.ingest(pdf: FixturePDF.makeScanned(pages: zorpScan()), title: "ZORP", id: "z",
                                                         settings: IngestSettings())
        #expect(document.info.ocrPages == [1])
        #expect(document.info.pagesWithoutText == [])
        #expect(document.info.sectionSource == .headings)
        #expect(document.sections.map(\.title) == ["Encoding Lists", "Encoding Numbers"])
        #expect(document.chunks.contains { $0.text.contains("Every item keeps its original order.") })
    }

    @Test func ocrOffLeavesScannedPagesEmpty() async throws {
        let document = try await DocumentIngestor.ingest(pdf: FixturePDF.makeScanned(pages: zorpScan()), title: "ZORP", id: "z",
                                                         settings: IngestSettings(), ocr: false)
        #expect(document.info.ocrPages == [])
        #expect(document.info.pagesWithoutText == [1])
        #expect(document.chunks.isEmpty)
    }

    @Test func reingestKeepsOCRTextWithoutRunningOCRAgain() async throws {
        let original = try await DocumentIngestor.ingest(pdf: FixturePDF.makeScanned(pages: zorpScan()), title: "ZORP", id: "z",
                                                         settings: IngestSettings())
        // reingest is synchronous and has no PDF: it can only use the stored OCR lines.
        let again = DocumentIngestor.reingest(original.source, title: "ZORP", id: "z", settings: IngestSettings())
        #expect(again.info == original.info)
        #expect(again.chunks == original.chunks)
    }

    @Test func engineFindsTextOnAScannedPage() async throws {
        let url = try writeTemporary(FixturePDF.makeScanned(pages: zorpScan()))
        defer { try? FileManager.default.removeItem(at: url) }
        var configuration = RAGConfiguration(embedder: FakeEmbedder())
        configuration.tokenCounter = EstimatingTokenCounter()
        let engine = RAGEngine(configuration: configuration)
        let info = try await engine.addDocument(pdfAt: url, title: "ZORP")
        #expect(info.ocrPages == [1])
        let top = try await engine.search("negative numbers minus sign").first
        #expect(top?.chunk.sectionTitle == "Encoding Numbers")
    }

    @Test func engineRespectsOCRSwitch() async throws {
        let url = try writeTemporary(FixturePDF.makeScanned(pages: zorpScan()))
        defer { try? FileManager.default.removeItem(at: url) }
        var configuration = RAGConfiguration(embedder: FakeEmbedder())
        configuration.ocrScannedPages = false
        let info = try await RAGEngine(configuration: configuration).addDocument(pdfAt: url, title: "ZORP")
        #expect(info.ocrPages == [])
        #expect(info.pagesWithoutText == [1])
    }
}
#endif
