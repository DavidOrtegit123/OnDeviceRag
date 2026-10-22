import Testing
@testable import OnDeviceRAG

struct TypesTests {
    @Test func pageRangeUnionAndLabel() {
        let range = PageRange(first: 3, last: 3).union(PageRange(first: 5, last: 6))
        #expect(range == PageRange(first: 3, last: 6))
        #expect(range.label == "pp. 3–6")
        #expect(PageRange(first: 2, last: 2).label == "p. 2")
    }

    @Test func chunkIndexedTextIncludesSectionTitle() {
        let chunk = Chunk(documentID: "d", sectionIndex: 0, sectionTitle: "Rules", position: 0,
                          text: "Integers are encoded as i<n>.", pages: PageRange(first: 1, last: 1), isCode: false)
        #expect(chunk.indexedText == "Rules: Integers are encoded as i<n>.")
        #expect(chunk.id == "d#0.0")
    }
}
