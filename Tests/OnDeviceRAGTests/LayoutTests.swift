import Testing
@testable import OnDeviceRAG

/// PDF layout rules found with a 63-page test manual (0.2.3): small bold lines, page footers,
/// chapter headings directly above section headings.
struct LayoutTests {
    private let body = "Ordinary body text that is long enough to be the most common style here."

    private func sections(_ lines: [TextLine]) -> [DetectedSection] {
        SectionDetector.detect(lines: lines, outline: [], documentTitle: "Doc", windowLength: 1500).sections
    }

    // MARK: Rule 1 — bold text smaller than the body is not a heading

    @Test func aSmallBoldTableHeaderStaysInsideItsSection() {
        let result = sections([
            TextLine("Specifications", fontSize: 14, isBold: true), TextLine(body),
            TextLine("Property Value", fontSize: 9.5, isBold: true),
            TextLine("Maximum sensors per controller 64", fontSize: 9.5),
            TextLine("Glossary", fontSize: 14, isBold: true), TextLine(body),
        ])
        #expect(result.map(\.title) == ["Specifications", "Glossary"])
        #expect(result[0].lines.map(\.text).contains("Property Value"))
        #expect(result[0].lines.map(\.text).contains("Maximum sensors per controller 64"))
    }

    // MARK: Rule 3 — heading levels

    @Test func aChapterHeadingNamesTheSectionsBelowIt() {
        let result = sections([
            TextLine("4 Irrigation module", fontSize: 18, isBold: true),
            TextLine("4.1 Overview", fontSize: 14, isBold: true), TextLine(body),
            TextLine("4.2 Schedules", fontSize: 14, isBold: true), TextLine(body),
            TextLine("5 Ventilation module", fontSize: 18, isBold: true),
            TextLine("5.1 Overview", fontSize: 14, isBold: true), TextLine(body),
        ])
        #expect(result.map(\.title) == ["4 Irrigation module › 4.1 Overview", "4 Irrigation module › 4.2 Schedules",
                                        "5 Ventilation module › 5.1 Overview"])
        #expect(result.allSatisfy { $0.lines.contains { !$0.isBlank } })
    }

    @Test func aChapterWithItsOwnTextKeepsItAsASection() {
        let result = sections([
            TextLine("4 Irrigation module", fontSize: 18, isBold: true), TextLine(body),
            TextLine("4.1 Overview", fontSize: 14, isBold: true), TextLine(body),
        ])
        #expect(result.map(\.title) == ["4 Irrigation module", "4 Irrigation module › 4.1 Overview"])
    }

    @Test func aWrappedTitleInOneStyleStillMerges() {
        let result = sections([
            TextLine("Verdane GH-4", fontSize: 26, isBold: true), TextLine("Greenhouse Controller", fontSize: 26, isBold: true),
            TextLine(body),
            TextLine("1 Introduction", fontSize: 18, isBold: true), TextLine(body),
        ])
        #expect(result.map(\.title) == ["Verdane GH-4 Greenhouse Controller",
                                        "Verdane GH-4 Greenhouse Controller › 1 Introduction"])
    }

    @Test func anEmptyHeadingThatIsNoParentKeepsItsText() {
        // A subtitle followed by a larger heading: it names nothing, so its words must not vanish.
        let result = sections([
            TextLine("Operator Manual", fontSize: 14), TextLine("1 Introduction", fontSize: 18, isBold: true), TextLine(body),
            TextLine("2 Installation", fontSize: 18, isBold: true), TextLine(body),
        ])
        #expect(result.flatMap(\.lines).map(\.text).contains("Operator Manual") || result.map(\.title).contains("Operator Manual"))
        #expect(result.contains { $0.title.hasSuffix("1 Introduction") })
    }

    // MARK: Rule 2 — repeating page headers and footers

    private func pages(_ count: Int, footer: (Int) -> String?) -> [TextLine] {
        (1...count).flatMap { page -> [TextLine] in
            var lines = [TextLine("Part \(page)", page: page, fontSize: 14, isBold: true),
                         TextLine("Body text of page \(page) that explains something useful.", page: page)]
            if let text = footer(page) { lines.append(TextLine(text, page: page, fontSize: 9)) }
            return lines
        }
    }

    private func ingest(_ lines: [TextLine], pageCount: Int) -> IngestedDocument {
        DocumentIngestor.ingest(lines: lines, outline: [], pageCount: pageCount, pagesWithoutText: [],
                                title: "Doc", id: "doc", settings: IngestSettings())
    }

    @Test func aFooterRepeatedOnMostPagesIsRemovedButKeptInTheSource() {
        let lines = pages(6) { "Verdane Manual - page \($0)" }
        let document = ingest(lines, pageCount: 6)
        #expect(!document.chunks.contains { $0.text.contains("Verdane Manual") })
        #expect(document.chunks.contains { $0.text.contains("Body text of page 4") })
        #expect(document.sections.map(\.title) == (1...6).map { "Part \($0)" })  // "Part #" is a heading, not a header
        guard case let .pdf(stored, _, _, _) = document.source else { Issue.record("not a PDF source"); return }
        #expect(stored.filter { $0.text.contains("Verdane Manual") }.count == 6)
    }

    @Test func aLastLineOnFewPagesIsNotTreatedAsAFooter() {
        let lines = pages(6) { $0 <= 2 ? "See also the appendix on page \($0)" : nil }
        let document = ingest(lines, pageCount: 6)
        #expect(document.chunks.filter { $0.text.contains("See also the appendix") }.count == 2)
    }

    @Test func shortDocumentsKeepTheirLastLines() {
        let lines = pages(2) { "Verdane Manual - page \($0)" }
        let document = ingest(lines, pageCount: 2)
        #expect(document.chunks.contains { $0.text.contains("Verdane Manual") })
    }
}
