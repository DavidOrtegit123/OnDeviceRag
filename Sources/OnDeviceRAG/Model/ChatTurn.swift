/// One message of a conversation, passed as `history` to `RAGEngine.search` / `context(for:)`.
public struct ChatTurn: Sendable, Hashable, Codable {
    public enum Role: String, Sendable, Codable {
        case user
        case assistant
    }

    public var role: Role
    public var text: String

    public init(role: Role, text: String) {
        self.role = role
        self.text = text
    }

    public static func user(_ text: String) -> ChatTurn { ChatTurn(role: .user, text: text) }
    public static func assistant(_ text: String) -> ChatTurn { ChatTurn(role: .assistant, text: text) }

    /// "User: …" / "Assistant: …" — also what `HistoryWindow` counts for this turn.
    func line(userLabel: String = "User", assistantLabel: String = "Assistant") -> String {
        "\(role == .user ? userLabel : assistantLabel): \(text)"
    }

    /// The turns as labeled lines, oldest first, separated by blank lines (ready for a prompt).
    public static func format(_ turns: [ChatTurn], userLabel: String = "User", assistantLabel: String = "Assistant") -> String {
        turns.map { $0.line(userLabel: userLabel, assistantLabel: assistantLabel) }.joined(separator: "\n\n")
    }
}
