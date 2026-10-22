import Testing
@testable import OnDeviceRAG

struct ChunkerTests {
    private func build(_ lines: [TextLine], limit: Int = 400) -> (section: Section, chunks: [Chunk]) {
        Chunker.build(DetectedSection(title: "S", lines: lines), documentID: "d", sectionIndex: 0, maximumLength: limit)!
    }

    @Test func joinsWrappedLinesAndDehyphenates() {
        let result = build([TextLine("The encoder writes inte-"), TextLine("gers as text.")])
        #expect(result.chunks.map(\.text) == ["The encoder writes integers as text."])
    }

    @Test func packsWholeSentencesUpToTheLimit() {
        // Real sentences (capitalized): NLTokenizer doesn't split "word. word".
        let sentences = (1...8).map { "Rule \($0) says the encoder writes every value as plain text with a fixed tag." }
        #expect(sentences.allSatisfy { (67...99).contains($0.count) }) // 2 fit in 200 characters, 3 don't
        let result = build([TextLine(sentences.joined(separator: " "))], limit: 200)
        #expect(result.chunks.allSatisfy { $0.text.count <= 200 })
        #expect(result.chunks.allSatisfy { $0.text.hasSuffix(".") })
        #expect(result.chunks.count == 4)
    }

    @Test func cutsAnOverlongSentenceAtASpace() {
        let long = Array(repeating: "abcdefghi", count: 60).joined(separator: " ") // 599 chars, no period
        let result = build([TextLine(long)])
        #expect(result.chunks.count == 2)
        #expect(result.chunks.allSatisfy { $0.text.count <= 400 && !$0.text.hasPrefix(" ") })
    }

    @Test func keepsCodeBlockWholeWithIndentation() {
        let code = (0..<30).map { TextLine("    value_\($0) = compute(\($0))", isMonospaced: true) }
        let result = build([TextLine("Example:")] + code + [TextLine("That is all.")])
        let codeChunks = result.chunks.filter(\.isCode)
        #expect(codeChunks.count == 1)
        #expect(codeChunks[0].text.count > 400)
        #expect(codeChunks[0].text.hasPrefix("    value_0 = compute(0)\n    value_1"))
        #expect(result.chunks.map(\.isCode) == [false, true, false])
        #expect(result.chunks.map(\.position) == [0, 1, 2])
    }

    @Test func tracksPagesPerChunkAndSection() {
        let result = build([TextLine("First page sentence.", page: 2), TextLine("Second page sentence.", page: 3)], limit: 25)
        #expect(result.chunks.map(\.pages) == [PageRange(first: 2, last: 2), PageRange(first: 3, last: 3)])
        #expect(result.section.pages == PageRange(first: 2, last: 3))
    }

    @Test func sectionTextSeparatesParagraphsAndCode() {
        let result = build([
            TextLine("Para one."), TextLine(""), TextLine("Para two."),
            TextLine("x = 1", isMonospaced: true),
        ])
        #expect(result.section.text == "Para one.\n\nPara two.\n\nx = 1")
    }

    @Test func emptySectionReturnsNil() {
        #expect(Chunker.build(DetectedSection(title: "S", lines: [TextLine("  ")]), documentID: "d", sectionIndex: 0, maximumLength: 400) == nil)
    }
}
