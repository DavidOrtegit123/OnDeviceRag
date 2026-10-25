import Foundation
import Testing
@testable import OnDeviceRAG

struct OCRFormatTests {
    @Test func linesSavedBefore2bDecodeAsNotOCR() throws {
        let json = #"{"text":"Lists","page":2,"fontSize":18,"isBold":true,"isMonospaced":false}"#
        let line = try JSONDecoder().decode(TextLine.self, from: Data(json.utf8))
        #expect(line == TextLine("Lists", page: 2, fontSize: 18, isBold: true))
        #expect(line.isOCR == false)
    }

    @Test func infoSavedBefore2bDecodesWithNoOCRPages() throws {
        let info = DocumentIngestor.ingestText("## A\nAlpha.", title: "T", id: "t").info
        var object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(info)) as? [String: Any])
        #expect(object["ocrPages"] != nil)
        object["ocrPages"] = nil
        let decoded = try JSONDecoder().decode(DocumentInfo.self, from: JSONSerialization.data(withJSONObject: object))
        #expect(decoded == info)
    }

    @Test func ocrFlagRoundTrips() throws {
        let line = TextLine("Scanned", page: 3, isOCR: true)
        #expect(try JSONDecoder().decode(TextLine.self, from: JSONEncoder().encode(line)) == line)
    }

    @Test func ocrPagesComeFromTheLines() {
        let lines = [TextLine("Heading", page: 1, fontSize: 18), TextLine("Typed body text on the first page.", page: 1),
                     TextLine("Scanned body text on the second page.", page: 2, isOCR: true),
                     TextLine("More scanned text on the third page.", page: 3, isOCR: true)]
        let document = DocumentIngestor.ingest(lines: lines, outline: [], pageCount: 3, pagesWithoutText: [],
                                               title: "T", id: "t", settings: IngestSettings())
        #expect(document.info.ocrPages == [2, 3])
        #expect(DocumentIngestor.ingestText("## A\nAlpha.", title: "T").info.ocrPages == [])
    }
}
