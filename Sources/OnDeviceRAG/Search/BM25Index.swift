import Foundation

/// Classic BM25 keyword ranking (Lucene-style IDF) with a `[a-z0-9_]+` tokenizer; k1 and b are
/// parameters.
/// Good at exact names such as `zorplib` or `encode_zorp`, which embeddings tend to blur.
struct BM25Index: Sendable {
    let k1: Double
    let b: Double

    private let termCounts: [[String: Int]]
    private let lengths: [Int]
    private let averageLength: Double
    private let documentFrequency: [String: Int]

    init(documents: [String], k1: Double, b: Double) {
        self.k1 = k1
        self.b = b
        let tokenized = documents.map { Self.tokenize($0) }
        termCounts = tokenized.map { tokens in
            tokens.reduce(into: [String: Int]()) { counts, token in counts[token, default: 0] += 1 }
        }
        lengths = tokenized.map(\.count)
        averageLength = lengths.isEmpty ? 0 : Double(lengths.reduce(0, +)) / Double(lengths.count)

        var frequency: [String: Int] = [:]
        for counts in termCounts {
            for term in counts.keys {
                frequency[term, default: 0] += 1
            }
        }
        documentFrequency = frequency
    }

    /// ln(1 + (N − n + 0.5) / (n + 0.5)), always positive.
    func inverseDocumentFrequency(of term: String) -> Double {
        let containing = Double(documentFrequency[term] ?? 0)
        let total = Double(termCounts.count)
        return log(1 + (total - containing + 0.5) / (containing + 0.5))
    }

    /// One score per document, in input order. Repeated query tokens count each time.
    func scores(for query: String) -> [Double] {
        let queryTokens = Self.tokenize(query)
        return termCounts.indices.map { index in
            let relativeLength = averageLength > 0 ? Double(lengths[index]) / averageLength : 0
            return queryTokens.reduce(0.0) { total, term in
                guard let count = termCounts[index][term] else { return total }
                let frequency = Double(count)
                let saturation = frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * relativeLength))
                return total + inverseDocumentFrequency(of: term) * saturation
            }
        }
    }

    /// Lowercases, then keeps maximal runs of a–z, 0–9 and `_` (so `encode_zorp` stays one token).
    static func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        var current = String.UnicodeScalarView()
        for scalar in text.lowercased().unicodeScalars {
            let value = scalar.value
            let isTokenScalar = (97...122).contains(value) || (48...57).contains(value) || value == 95
            if isTokenScalar {
                current.append(scalar)
            } else if !current.isEmpty {
                tokens.append(String(current))
                current = String.UnicodeScalarView()
            }
        }
        if !current.isEmpty {
            tokens.append(String(current))
        }
        return tokens
    }
}
