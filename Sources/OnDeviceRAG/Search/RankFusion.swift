/// Reciprocal Rank Fusion: score = Σ over rankings of 1 / (constant + rank), ranks 1-based.
/// Items in no ranking score 0. Ties go to the lower index.
enum RankFusion {
    static func fuse(_ orders: [[Int]], count: Int, constant: Double) -> [(index: Int, score: Double)] {
        var scores = [Double](repeating: 0, count: count)
        for order in orders {
            for (position, index) in order.enumerated() {
                scores[index] += 1 / (constant + Double(position + 1))
            }
        }
        return scores.indices
            .map { (index: $0, score: scores[$0]) }
            .sorted { $0.score != $1.score ? $0.score > $1.score : $0.index < $1.index }
    }
}
