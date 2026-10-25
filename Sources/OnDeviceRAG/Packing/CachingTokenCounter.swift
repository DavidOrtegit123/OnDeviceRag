/// Remembers exact token counts by text. Apple's `tokenCount(for:)` costs ~48 ms per call regardless
/// of length (measured on a Mac, 2026-10-25), and packing asks for the same section texts on every
/// question, so caching turns ~50 calls per question into ~1. Estimates aren't cached: they're cheap,
/// and an estimate (Apple Intelligence off) must not outlive the model becoming available.
actor CachingTokenCounter: TokenCounter {
    /// Window and truncation texts are one-offs; clearing past this bounds memory.
    static let maximumEntries = 20_000

    private let counter: any TokenCounter
    private var counts: [String: TokenCount] = [:]

    init(_ counter: any TokenCounter) {
        self.counter = counter
    }

    func count(_ text: String) async -> TokenCount {
        if let cached = counts[text] { return cached }
        let count = await counter.count(text)
        if !count.isEstimate {
            if counts.count >= Self.maximumEntries { counts.removeAll() }
            counts[text] = count
        }
        return count
    }

    /// Counts texts not yet cached, one at a time (so the model isn't flooded with requests).
    func warm(_ texts: [String]) async {
        for text in texts where counts[text] == nil {
            _ = await count(text)
        }
    }
}
