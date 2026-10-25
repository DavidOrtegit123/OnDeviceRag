import Foundation
import Testing
@testable import OnDeviceRAG

struct SourceTests {
    private let longText = "## Rules\n" + (1...12).map { "Rule \($0) says the encoder writes every value as plain text with a fixed tag." }.joined(separator: " ")

    @Test func textSourceIsKeptAndReingestsIdentically() {
        let original = DocumentIngestor.ingestText(longText, title: "Doc", id: "d")
        #expect(original.source == .text(longText))
        let again = DocumentIngestor.reingest(original.source, title: "Doc", id: "d", settings: IngestSettings())
        #expect(again.sections == original.sections)
        #expect(again.chunks == original.chunks)
    }

    @Test func reingestWithSmallerChunksRechunks() {
        let original = DocumentIngestor.ingestText(longText, title: "Doc", id: "d")
        var settings = IngestSettings()
        settings.chunkLength = 100
        let smaller = DocumentIngestor.reingest(original.source, title: "Doc", id: "d", settings: settings)
        #expect(smaller.chunks.count > original.chunks.count)
        #expect(smaller.chunks.allSatisfy { $0.text.count <= 100 })
    }

    @Test func sourceRoundTripsThroughJSON() throws {
        let source = DocumentSource.pdf(lines: [TextLine("Title", page: 1, fontSize: 18, isBold: true), TextLine("x = 1", isMonospaced: true)],
                                        outline: [OutlineEntry(title: "Title", page: 1)], pageCount: 2, pagesWithoutText: [2])
        let decoded = try JSONDecoder().decode(DocumentSource.self, from: JSONEncoder().encode(source))
        #expect(decoded == source)
    }

    #if os(macOS)
    @Test func pdfSourceReingestsIdenticallyWithoutTheFile() async throws {
        let pdf = FixturePDF.make(pages: [[.heading("Integers"), .body("Integers are written as i<n>.")],
                                          [.heading("Lists"), .body("Lists are written as l<items>e."), .code("encode([1])")]])
        let original = try await DocumentIngestor.ingest(pdf: pdf, title: "ZORP", id: "z", settings: IngestSettings())
        guard case .pdf = original.source else {
            Issue.record("expected a PDF source")
            return
        }
        let again = DocumentIngestor.reingest(original.source, title: "ZORP", id: "z", settings: IngestSettings())
        #expect(again.info == original.info)
        #expect(again.sections == original.sections)
        #expect(again.chunks == original.chunks)
    }
    #endif
}
