/// Reciprocal Rank Fusion: score = Σ over rankings of weight / (constant + rank), ranks 1-based.
/// Weights default to 1. Items in no ranking score 0. Ties go to the lower index.
enum RankFusion {
    static func fuse(_ orders: [[Int]], weights: [Double]? = nil, count: Int, constant: Double) -> [(index: Int, score: Double)] {
        precondition(weights == nil || weights?.count == orders.count, "one weight per ranking")
        var scores = [Double](repeating: 0, count: count)
        for (ranking, order) in orders.enumerated() {
            let weight = weights?[ranking] ?? 1
            for (position, index) in order.enumerated() {
                scores[index] += weight / (constant + Double(position + 1))
            }
        }
        return scores.indices
            .map { (index: $0, score: scores[$0]) }
            .sorted { $0.score != $1.score ? $0.score > $1.score : $0.index < $1.index }
    }
}
