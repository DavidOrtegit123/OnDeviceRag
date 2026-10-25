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
