import Testing
@testable import OnDeviceRAG

struct SectionDetectorTests {
    @Test func outlineSplitsAtMatchingHeadingLines() {
        let lines = [
            TextLine("Preface text.", page: 1),
            TextLine("1 Integers", page: 2, fontSize: 16), TextLine("Ints are i<n>.", page: 2),
            TextLine("2  Strings", page: 3, fontSize: 16), TextLine("Strings are s<n>.", page: 3),
        ]
        let outline = [OutlineEntry(title: "1 Integers", page: 2), OutlineEntry(title: "2 Strings", page: 3)]
        let result = SectionDetector.detect(lines: lines, outline: outline, documentTitle: "Doc", windowLength: 1500)
        #expect(result.source == .outline)
        #expect(result.sections.map(\.title) == ["Doc", "1 Integers", "2 Strings"])
        #expect(result.sections[1].lines.map(\.text) == ["Ints are i<n>."])
        #expect(result.sections[2].lines.map(\.text) == ["Strings are s<n>."])
    }

    @Test func outlineEntryWithoutMatchingLineStartsAtItsPage() {
        let lines = [TextLine("A.", page: 1), TextLine("B.", page: 2)]
        let result = SectionDetector.detect(lines: lines, outline: [OutlineEntry(title: "Second", page: 2)],
                                            documentTitle: "Doc", windowLength: 1500)
        #expect(result.sections.map(\.title) == ["Doc", "Second"])
        #expect(result.sections[1].lines.map(\.text) == ["B."])
    }

    @Test func largeFontLinesBecomeHeadingsAndMultiLineTitlesMerge() {
        let body = "This is ordinary body text that is long enough to dominate."
        let lines = [
            TextLine("Chapter One", fontSize: 18), TextLine("The Start", fontSize: 18), TextLine(body),
            TextLine("Chapter Two", fontSize: 18), TextLine(body),
        ]
        let result = SectionDetector.detect(lines: lines, outline: [], documentTitle: "Doc", windowLength: 1500)
        #expect(result.source == .headings)
        #expect(result.sections.map(\.title) == ["Chapter One The Start", "Chapter Two"])
    }

    @Test func boldShortLinesAreHeadingsButBoldSentencesAreNot() {
        let body = "Plain body text that is clearly the most common style here."
        let lines = [
            TextLine("Setup", isBold: true), TextLine(body),
            TextLine("Remember to save.", isBold: true), TextLine(body),
            TextLine("Usage", isBold: true), TextLine(body),
        ]
        let result = SectionDetector.detect(lines: lines, outline: [], documentTitle: "Doc", windowLength: 1500)
        #expect(result.sections.map(\.title) == ["Setup", "Usage"])
        #expect(result.sections[0].lines.map(\.text).contains("Remember to save."))
    }

    @Test func fallsBackToWindowsWithoutStructure() {
        let lines = (0..<10).map { TextLine(String(repeating: "x", count: 99) + "\($0)") } // 100 chars each
        let result = SectionDetector.detect(lines: lines, outline: [], documentTitle: "Doc", windowLength: 300)
        #expect(result.source == .windows)
        #expect(result.sections.map(\.title) == ["Doc (part 1)", "Doc (part 2)", "Doc (part 3)", "Doc (part 4)"])
        #expect(result.sections.map(\.lines.count) == [3, 3, 3, 1])
    }
}
