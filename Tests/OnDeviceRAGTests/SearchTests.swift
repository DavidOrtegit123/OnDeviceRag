import Foundation
import Testing
@testable import OnDeviceRAG

struct SearchTests {
    @Test func tokenizerKeepsUnderscoresAndLowercases() {
        #expect(BM25Index.tokenize("Call encode_ZORP(x), then v2!") == ["call", "encode_zorp", "x", "then", "v2"])
    }

    @Test func bm25RanksExactRareTermFirst() {
        let index = BM25Index(documents: [
            "the cache stores values",
            "zorplib encodes integers",
            "the the the cache",
        ], k1: 1.2, b: 0.75)
        let scores = index.scores(for: "zorplib integers")
        #expect(scores[1] > scores[0])
        #expect(scores[0] == 0)
        #expect(scores[2] == 0)
    }

    @Test func bm25MatchesHandComputedValue() {
        // Two docs; query "a". Doc 0 = "a b" (len 2), doc 1 = "b b" (len 2), avg 2.
        // idf(a) = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln(2)
        // tf part = 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 1)) = 2.2 / 2.2 = 1
        let index = BM25Index(documents: ["a b", "b b"], k1: 1.2, b: 0.75)
        let scores = index.scores(for: "a")
        #expect(abs(scores[0] - log(2.0)) < 1e-12)
        #expect(scores[1] == 0)
    }

    @Test func rrfCombinesRanksAndBreaksTiesByIndex() {
        // Item 2 is 1st in A and 2nd in B; item 0 is 2nd in A only; item 1 is 1st in B only.
        let fused = RankFusion.fuse([[2, 0], [1, 2]], count: 4, constant: 60)
        #expect(fused.map(\.index) == [2, 1, 0, 3])
        #expect(abs(fused[0].score - (1.0 / 61 + 1.0 / 62)) < 1e-12)
        #expect(fused[3].score == 0)
    }
}
