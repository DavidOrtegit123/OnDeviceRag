import Foundation
import Testing
@testable import OnDeviceRAG

struct HistoryTests {
    @Test func formatLabelsEachTurnOldestFirst() {
        let turns: [ChatTurn] = [.user("What is ZORP?"), .assistant("A format."), .user("And lists?")]
        #expect(ChatTurn.format(turns) == "User: What is ZORP?\n\nAssistant: A format.\n\nUser: And lists?")
        #expect(ChatTurn.format(turns, userLabel: "Student", assistantLabel: "Tutor")
                == "Student: What is ZORP?\n\nTutor: A format.\n\nStudent: And lists?")
        #expect(ChatTurn.format([]) == "")
    }

    @Test func historyIsOffByDefault() {
        let settings = RAGConfiguration(embedder: FakeEmbedder()).history
        #expect(!settings.useForSearch)
        #expect(!settings.includeInContext)
        #expect(settings.searchTurns == 2)
        #expect(settings.maximumTurns == 6)
        #expect(settings.tokenBudget == 600)
    }
}

/// One token per character, so budgets in the window tests are exact.
private struct CharacterCounter: TokenCounter {
    func count(_ text: String) async -> TokenCount { TokenCount(value: text.count, isEstimate: false) }
}

extension HistoryTests {
    // "User: a" = 7 tokens, "Assistant: b" = 12, separator "\n\n" = 2.
    private var abc: [ChatTurn] { [.user("a"), .assistant("b"), .user("c")] }

    @Test func keepsTheNewestTurnsWithinMaximumTurns() async {
        let window = await HistoryWindow.select(abc, maximumTurns: 2, tokenBudget: 1000, counter: CharacterCounter())
        #expect(window.turns == [.assistant("b"), .user("c")])
        #expect(window.tokenCount == 12 + 2 + 7)
    }

    @Test func dropsTheOldestTurnsToFitTheBudget() async {
        let window = await HistoryWindow.select(abc, maximumTurns: 6, tokenBudget: 21, counter: CharacterCounter())
        #expect(window.turns == [.assistant("b"), .user("c")])
        #expect(window.tokenCount == 21)
    }

    @Test func stopsAtTheFirstTurnThatDoesNotFit() async {
        let turns: [ChatTurn] = [.user("a"), .assistant(String(repeating: "x", count: 100)), .user("c")]
        let window = await HistoryWindow.select(turns, maximumTurns: 6, tokenBudget: 30, counter: CharacterCounter())
        #expect(window.turns == [.user("c")])  // "User: a" would fit, but the long answer ends the run
    }

    @Test func emptyHistoryOrNoBudgetGivesNothing() async {
        let empty = await HistoryWindow.select([], maximumTurns: 6, tokenBudget: 600, counter: CharacterCounter())
        let noBudget = await HistoryWindow.select(abc, maximumTurns: 6, tokenBudget: 0, counter: CharacterCounter())
        #expect(empty.turns.isEmpty && empty.tokenCount == 0)
        #expect(noBudget.turns.isEmpty && noBudget.tokenCount == 0)
    }

    @Test func contextReturnsHistoryOnlyWhenSwitchedOn() async throws {
        var on = RAGConfiguration(embedder: FakeEmbedder())
        on.tokenCounter = EstimatingTokenCounter()
        on.history.includeInContext = true
        var off = on
        off.history.includeInContext = false
        let document = DocumentIngestor.ingestText("## Cache\nThe cache stores loaded values.", title: "Cache SDK", id: "cache")
        let withHistory = RAGEngine(configuration: on)
        let without = RAGEngine(configuration: off)
        for engine in [withHistory, without] { _ = try await engine.addDocument(document) }
        let turns: [ChatTurn] = [.user("What is the cache?"), .assistant("It stores values.")]

        let returned = try await withHistory.context(for: "cache", history: turns)
        let ignored = try await without.context(for: "cache", history: turns)
        let plain = try await without.context(for: "cache")
        #expect(returned.history == turns)
        #expect(returned.historyTokenCount > 0)
        #expect(returned.text == plain.text)
        #expect(ignored == plain)
    }
}
