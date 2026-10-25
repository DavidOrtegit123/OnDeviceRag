import CoreGraphics
import PDFKit
import Vision

/// Reads scanned PDF pages (pictures of text, no text layer) with Vision's document recognizer.
/// Measured on a Mac (2026-10-25): ~0.2 s per page; the very first request on a device can take
/// ~45 s while the system prepares its model (once, cached by the OS).
enum PageOCR {
    /// 2 × PDF points ≈ 144 dpi.
    static let renderScale: CGFloat = 2
    /// Longest rendered side in pixels, so poster-sized pages don't exhaust memory.
    static let maximumPixelSide: CGFloat = 4000
    /// Body lines measured 0.86–1.23 × the page's median line height and headings ≥ 1.43 ×, so
    /// lines under 1.3 × are body text (their raw heights vary with glyph shapes).
    static let headingHeightRatio = 1.3
    /// Font size given to OCR body lines; only relative sizes matter to section detection.
    static let bodyFontSize = 11.0

    /// Replaces the lines of every page without usable text by its OCR lines, and recomputes which
    /// pages still have none. A page whose recognition fails or finds nothing keeps what it had.
    static func fill(_ extracted: (lines: [TextLine], pagesWithoutText: [Int]),
                     in document: PDFDocument) async throws -> (lines: [TextLine], pagesWithoutText: [Int]) {
        var byPage = Dictionary(grouping: extracted.lines, by: \.page)
        for pageNumber in extracted.pagesWithoutText {
            try Task.checkCancellation()
            guard let page = document.page(at: pageNumber - 1),
                  let recognized = try? await lines(of: page, pageNumber: pageNumber),
                  recognized.contains(where: { !$0.isBlank }) else { continue }
            byPage[pageNumber] = recognized
        }
        let lines = (0..<document.pageCount).flatMap { byPage[$0 + 1] ?? [] }
        let stillEmpty = extracted.pagesWithoutText.filter {
            PDFTextExtractor.visibleCharacterCount(byPage[$0] ?? []) < PDFTextExtractor.minimumCharactersForTextPage
        }
        return (lines, stillEmpty)
    }

    /// The page's text as lines: paragraphs in reading order, separated by one blank line.
    /// Lines clearly taller than the page's median become larger (heading candidates); the
    /// recognizer's document title is marked bold.
    static func lines(of page: PDFPage, pageNumber: Int) async throws -> [TextLine] {
        guard let image = render(page) else { return [] }
        let observations = try await RecognizeDocumentsRequest().perform(on: image)
        var result: [TextLine] = []
        for observation in observations {
            let document = observation.document
            let title = document.title?.transcript.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let heights = document.paragraphs.flatMap(\.lines).map { Double($0.boundingBox.height) }.sorted()
            let median = heights.isEmpty ? 0 : heights[heights.count / 2]
            for paragraph in document.paragraphs {
                if !result.isEmpty {
                    result.append(TextLine("", page: pageNumber, isOCR: true))
                }
                for line in paragraph.lines {
                    let text = line.transcript
                    let ratio = median > 0 ? Double(line.boundingBox.height) / median : 1
                    let isTitle = !title.isEmpty && text.trimmingCharacters(in: .whitespacesAndNewlines) == title
                    result.append(TextLine(text, page: pageNumber,
                                           fontSize: ratio >= headingHeightRatio ? bodyFontSize * ratio : bodyFontSize,
                                           isBold: isTitle, isOCR: true))
                }
            }
        }
        return result
    }

    /// White background, page drawn upright (`draw(with:to:)` applies the page's rotation).
    static func render(_ page: PDFPage) -> CGImage? {
        let bounds = page.bounds(for: .mediaBox)
        let upright = page.rotation % 180 == 0 ? bounds.size : CGSize(width: bounds.height, height: bounds.width)
        guard upright.width > 0, upright.height > 0 else { return nil }
        let scale = min(renderScale, maximumPixelSide / max(upright.width, upright.height))
        let width = Int((upright.width * scale).rounded())
        let height = Int((upright.height * scale).rounded())
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.scaleBy(x: scale, y: scale)
        page.draw(with: .mediaBox, to: context)
        return context.makeImage()
    }
}
