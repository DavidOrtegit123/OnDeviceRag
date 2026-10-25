@testable import OnDeviceRAG

/// A token counter that records every text it's asked to count, like a slow exact counter
/// (Apple's `tokenCount(for:)` costs ~48 ms per call whatever the length).
actor TallyCounter: TokenCounter {
    private(set) var texts: [String] = []
    private let exact: Bool

    init(exact: Bool = true) {
        self.exact = exact
    }

    var calls: Int { texts.count }

    func reset() {
        texts.removeAll()
    }

    func count(_ text: String) async -> TokenCount {
        texts.append(text)
        return TokenCount(value: Int((Double(text.count) / 4).rounded(.up)), isEstimate: !exact)
    }
}
