import Foundation

/// Finds a document's sections: the PDF outline if there is one, else headings detected
/// from font size or bold short lines, else fixed-size windows.
enum SectionDetector {
    static func detect(lines: [TextLine], outline: [OutlineEntry], documentTitle: String, windowLength: Int) -> (sections: [DetectedSection], source: SectionSource) {
        if !outline.isEmpty {
            return (fromOutline(lines, outline, documentTitle), .outline)
        }
        if let sections = fromHeadings(lines) {
            return (sections, .headings)
        }
        return (windows(lines, documentTitle, windowLength), .windows)
    }

    /// Lowercased, whitespace runs collapsed to one space.
    static func normalize(_ text: String) -> String {
        text.lowercased().split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    // MARK: - Outline

    /// Each entry starts at the first line (at or after the previous start) on its page whose
    /// text matches the entry title — that heading line is dropped from the body — or, if
    /// none matches, at the first line on or after its page.
    private static func fromOutline(_ lines: [TextLine], _ outline: [OutlineEntry], _ documentTitle: String) -> [DetectedSection] {
        var starts: [(index: Int, title: String, skipsHeading: Bool)] = []
        var searchFrom = 0
        for entry in outline {
            let target = normalize(entry.title)
            if let match = lines.indices.first(where: { $0 >= searchFrom && lines[$0].page == entry.page && normalize(lines[$0].text) == target }) {
                starts.append((match, entry.title, true))
                searchFrom = match + 1
            } else {
                let index = lines.indices.first(where: { $0 >= searchFrom && lines[$0].page >= entry.page }) ?? lines.count
                starts.append((index, entry.title, false))
                searchFrom = index
            }
        }

        var sections: [DetectedSection] = []
        let preamble = Array(lines[..<(starts.first?.index ?? lines.count)])
        if preamble.contains(where: { !$0.isBlank }) {
            sections.append(DetectedSection(title: documentTitle, lines: preamble))
        }
        for (position, start) in starts.enumerated() {
            let end = position + 1 < starts.count ? starts[position + 1].index : lines.count
            let bodyStart = min(start.skipsHeading ? start.index + 1 : start.index, end)
            sections.append(DetectedSection(title: start.title, lines: Array(lines[bodyStart..<end])))
        }
        return sections
    }

    // MARK: - Headings

    /// A heading is a non-monospaced line, 1–99 characters, not ending in ".", that is either
    /// ≥ 1.2 × the body font size or bold while body text isn't. Consecutive heading lines
    /// merge into one title. Needs ≥ 2 headings; text before the first joins the first section.
    private static func fromHeadings(_ lines: [TextLine]) -> [DetectedSection]? {
        let body = lines.filter { !$0.isBlank && !$0.isMonospaced }
        guard let bodySize = weightedMedianFontSize(body) else { return nil }
        let bodyIsBold = body.filter { $0.fontSize == bodySize }.allSatisfy(\.isBold)

        func isHeading(_ line: TextLine) -> Bool {
            let text = line.text.trimmingCharacters(in: .whitespaces)
            guard !line.isMonospaced, (1..<100).contains(text.count), !text.hasSuffix(".") else { return false }
            return line.fontSize >= 1.2 * bodySize || (line.isBold && !bodyIsBold)
        }

        var sections: [DetectedSection] = []
        var preamble: [TextLine] = []
        var previousWasHeading = false
        for line in lines {
            if isHeading(line) {
                let title = line.text.trimmingCharacters(in: .whitespaces)
                if previousWasHeading, let last = sections.indices.last {
                    sections[last].title += " " + title
                } else {
                    sections.append(DetectedSection(title: title, lines: []))
                }
                previousWasHeading = true
            } else {
                if !line.isBlank { previousWasHeading = false }
                if sections.isEmpty {
                    preamble.append(line)
                } else {
                    sections[sections.count - 1].lines.append(line)
                }
            }
        }
        guard sections.count >= 2 else { return nil }
        if preamble.contains(where: { !$0.isBlank }) {
            sections[0].lines = preamble + sections[0].lines
        }
        return sections
    }

    /// Median font size where each line counts once per character.
    private static func weightedMedianFontSize(_ lines: [TextLine]) -> Double? {
        let sizes = lines.map { (size: $0.fontSize, weight: $0.text.count) }.sorted { $0.size < $1.size }
        let total = sizes.reduce(0) { $0 + $1.weight }
        guard total > 0 else { return nil }
        var running = 0
        for entry in sizes {
            running += entry.weight
            if running * 2 >= total { return entry.size }
        }
        return sizes.last?.size
    }

    // MARK: - Windows

    /// Consecutive lines grouped until a group reaches `windowLength` characters (lines are never split).
    private static func windows(_ lines: [TextLine], _ documentTitle: String, _ windowLength: Int) -> [DetectedSection] {
        var sections: [DetectedSection] = []
        var current: [TextLine] = []
        var length = 0
        for line in lines {
            current.append(line)
            length += line.text.count
            if length >= windowLength {
                sections.append(DetectedSection(title: "\(documentTitle) (part \(sections.count + 1))", lines: current))
                current = []
                length = 0
            }
        }
        if current.contains(where: { !$0.isBlank }) {
            sections.append(DetectedSection(title: "\(documentTitle) (part \(sections.count + 1))", lines: current))
        }
        return sections
    }
}
