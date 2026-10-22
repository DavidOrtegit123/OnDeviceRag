import Foundation
import FoundationModels

/// Counts tokens for a model's prompt. Supply your own for models other than Apple's.
public protocol TokenCounter: Sendable {
    func count(_ text: String) async -> TokenCount
}

/// Characters ÷ 3, rounded up — deliberately conservative (English averages ~4 characters per token).
public struct EstimatingTokenCounter: TokenCounter {
    public var charactersPerToken = 3.0
    public init() {}

    public func count(_ text: String) async -> TokenCount {
        TokenCount(value: Int((Double(text.count) / charactersPerToken).rounded(.up)), isEstimate: true)
    }
}

/// Apple's on-device model's own count when Apple Intelligence is available; otherwise the estimate.
public struct AppleTokenCounter: TokenCounter {
    public init() {}

    public func count(_ text: String) async -> TokenCount {
        let model = SystemLanguageModel.default
        if case .available = model.availability, let value = try? await model.tokenCount(for: text) {
            return TokenCount(value: value, isEstimate: false)
        }
        return await EstimatingTokenCounter().count(text)
    }
}
