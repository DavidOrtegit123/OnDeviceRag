/// Chooses the conversation turns that go into a prompt: newest first, at most `maximumTurns`,
/// while they fit `tokenBudget`. The first turn that doesn't fit ends the window, so the result is
/// always an unbroken run of the latest turns. Returned oldest first. Turns are never cut.
enum HistoryWindow {
    static func select(_ turns: [ChatTurn], maximumTurns: Int, tokenBudget: Int,
                       counter: any TokenCounter) async -> (turns: [ChatTurn], tokenCount: Int) {
        guard maximumTurns > 0, tokenBudget > 0 else { return ([], 0) }
        var kept: [ChatTurn] = []
        var used = 0
        var separatorCost: Int?
        for turn in turns.reversed().prefix(maximumTurns) {
            var separator = 0
            if !kept.isEmpty {
                if separatorCost == nil { separatorCost = await counter.count("\n\n").value }
                separator = separatorCost ?? 0
            }
            let cost = await counter.count(turn.line()).value
            guard used + separator + cost <= tokenBudget else { break }
            kept.append(turn)
            used += separator + cost
        }
        return (Array(kept.reversed()), used)
    }
}
