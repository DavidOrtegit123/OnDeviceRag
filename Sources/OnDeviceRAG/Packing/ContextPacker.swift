/// Packs retrieved chunks into prompt text within a token budget, "small-to-big":
/// whole sections when they fit, otherwise a window of chunks around the match.
enum ContextPacker {
    private struct Block {
        var text: String
        var section: ContextSection
    }

    static let blockSeparator = "\n\n"

    static func pack(_ candidates: [RetrievedChunk], index: InMemoryIndex, tokenBudget: Int, counter: any TokenCounter) async -> RAGContext {
        var blocks: [Block] = []
        var used = 0
        var seen: Set<String> = []

        for candidate in candidates {
            let chunk = candidate.chunk
            let key = "\(chunk.documentID)#\(chunk.sectionIndex)"
            guard !seen.contains(key), let document = index.document(chunk.documentID),
                  chunk.sectionIndex < document.sections.count else { continue }
            let section = document.sections[chunk.sectionIndex]
            let separatorCost = blocks.isEmpty ? 0 : await counter.count(blockSeparator).value
            let remaining = tokenBudget - used - separatorCost
            if remaining <= 0 { break }

            let whole = header(section.title, document.info.title, section.pages) + "\n" + section.text
            let wholeCost = await counter.count(whole).value
            if wholeCost <= remaining {
                blocks.append(Block(text: whole, section: contextSection(section, document, section.pages, whole: true)))
                used += separatorCost + wholeCost
                seen.insert(key)
                continue
            }
            let sectionChunks = document.chunks.filter { $0.sectionIndex == chunk.sectionIndex }
            if let window = await window(around: chunk.position, in: sectionChunks, documentTitle: document.info.title,
                                         section: section, remaining: remaining, counter: counter) {
                blocks.append(Block(text: window.text, section: contextSection(section, document, window.pages, whole: false)))
                used += separatorCost + window.cost
                seen.insert(key)
            }
        }

        // Nothing fit: the top chunk, truncated, so a found answer is never dropped entirely.
        if blocks.isEmpty, let top = candidates.first, let document = index.document(top.chunk.documentID),
           top.chunk.sectionIndex < document.sections.count {
            let section = document.sections[top.chunk.sectionIndex]
            let full = header(section.title, document.info.title, top.chunk.pages) + "\n" + top.chunk.text
            let text = await truncate(full, toFit: tokenBudget, counter: counter)
            if !text.isEmpty {
                blocks.append(Block(text: text, section: contextSection(section, document, top.chunk.pages, whole: false)))
            }
        }

        // Token counts aren't always additive: re-count the joined text and drop blocks until it fits.
        while true {
            let text = blocks.map(\.text).joined(separator: blockSeparator)
            let count = text.isEmpty ? TokenCount(value: 0, isEstimate: false) : await counter.count(text)
            if count.value <= tokenBudget || blocks.count <= 1 {
                return RAGContext(text: text, sections: blocks.map(\.section), tokenCount: count.value,
                                  tokenCountIsEstimate: count.isEstimate)
            }
            blocks.removeLast()
        }
    }

    private static func header(_ sectionTitle: String, _ documentTitle: String, _ pages: PageRange) -> String {
        "## \(sectionTitle) (\(documentTitle), \(pages.label))"
    }

    private static func contextSection(_ section: Section, _ document: IngestedDocument, _ pages: PageRange, whole: Bool) -> ContextSection {
        ContextSection(documentID: document.info.id, documentTitle: document.info.title, sectionIndex: section.index,
                       title: section.title, pages: pages, isWholeSection: whole)
    }

    /// The matched chunk, then neighbors after/before alternately, while the block fits.
    private static func window(around position: Int, in chunks: [Chunk], documentTitle: String, section: Section,
                               remaining: Int, counter: any TokenCounter) async -> (text: String, pages: PageRange, cost: Int)? {
        guard let matched = chunks.firstIndex(where: { $0.position == position }) else { return nil }
        func render(_ range: ClosedRange<Int>) -> (text: String, pages: PageRange) {
            let selected = chunks[range]
            let pages = selected.dropFirst().reduce(selected[selected.startIndex].pages) { $0.union($1.pages) }
            return (header(section.title, documentTitle, pages) + "\n" + selected.map(\.text).joined(separator: "\n"), pages)
        }

        var range = matched...matched
        var (text, pages) = render(range)
        var cost = await counter.count(text).value
        guard cost <= remaining else { return nil }

        var growAfter = true
        var blockedAfter = false
        var blockedBefore = false
        while !(blockedAfter && blockedBefore) {
            let candidate: ClosedRange<Int>?
            if growAfter && !blockedAfter {
                candidate = range.upperBound + 1 < chunks.count ? range.lowerBound...(range.upperBound + 1) : nil
                if candidate == nil { blockedAfter = true }
            } else if !growAfter && !blockedBefore {
                candidate = range.lowerBound > 0 ? (range.lowerBound - 1)...range.upperBound : nil
                if candidate == nil { blockedBefore = true }
            } else {
                candidate = nil
            }
            if let candidate {
                let rendered = render(candidate)
                let candidateCost = await counter.count(rendered.text).value
                if candidateCost <= remaining {
                    range = candidate
                    (text, pages, cost) = (rendered.text, rendered.pages, candidateCost)
                } else if growAfter {
                    blockedAfter = true
                } else {
                    blockedBefore = true
                }
            }
            growAfter.toggle()
        }
        return (text, pages, cost)
    }

    /// Longest prefix (cut at a space when possible) whose count fits; binary search on length.
    private static func truncate(_ text: String, toFit budget: Int, counter: any TokenCounter) async -> String {
        if await counter.count(text).value <= budget { return text }
        var low = 0
        var high = text.count
        while low < high {
            let middle = (low + high + 1) / 2
            if await counter.count(String(text.prefix(middle))).value <= budget {
                low = middle
            } else {
                high = middle - 1
            }
        }
        let prefix = text.prefix(low)
        if let space = prefix.lastIndex(of: " "), prefix.distance(from: prefix.startIndex, to: space) > low / 2 {
            return String(prefix[..<space])
        }
        return String(prefix)
    }
}
