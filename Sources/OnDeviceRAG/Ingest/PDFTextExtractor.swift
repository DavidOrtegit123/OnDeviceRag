import Foundation
import PDFKit
#if canImport(AppKit)
import AppKit
private typealias PlatformFont = NSFont
#else
import UIKit
private typealias PlatformFont = UIFont
#endif

/// Reads a PDF's text line by line with font size, bold and monospace facts, plus its outline.
enum PDFTextExtractor {
    static let minimumCharactersForTextPage = 50

    static func lines(from document: PDFDocument) -> (lines: [TextLine], pagesWithoutText: [Int]) {
        var lines: [TextLine] = []
        var empty: [Int] = []
        for pageIndex in 0..<document.pageCount {
            let pageNumber = pageIndex + 1
            guard let attributed = document.page(at: pageIndex)?.attributedString else {
                empty.append(pageNumber)
                continue
            }
            let string = attributed.string as NSString
            var pageLines: [TextLine] = []
            string.enumerateSubstrings(in: NSRange(location: 0, length: string.length), options: .byLines) { substring, range, _, _ in
                guard let substring else { return }
                pageLines.append(makeLine(substring, range: range, in: attributed, page: pageNumber))
            }
            let characters = visibleCharacterCount(pageLines)
            if characters < minimumCharactersForTextPage {
                empty.append(pageNumber)
            }
            lines += pageLines
        }
        return (lines, empty)
    }

    /// Non-whitespace characters in the lines (a page with < `minimumCharactersForTextPage` has no usable text).
    static func visibleCharacterCount(_ lines: [TextLine]) -> Int {
        lines.reduce(0) { $0 + $1.text.filter { !$0.isWhitespace }.count }
    }

    /// Top two outline levels, in document order.
    static func outline(of document: PDFDocument) -> [OutlineEntry] {
        guard let root = document.outlineRoot else { return [] }
        var entries: [OutlineEntry] = []
        func walk(_ node: PDFOutline, depth: Int) {
            for childIndex in 0..<node.numberOfChildren {
                guard let child = node.child(at: childIndex) else { continue }
                if let label = child.label?.trimmingCharacters(in: .whitespacesAndNewlines), !label.isEmpty,
                   let page = child.destination?.page {
                    entries.append(OutlineEntry(title: label, page: document.index(for: page) + 1))
                }
                if depth < 2 {
                    walk(child, depth: depth + 1)
                }
            }
        }
        walk(root, depth: 1)
        return entries
    }

    /// Font facts come from the line's first visible character; font names back up the traits.
    private static func makeLine(_ text: String, range: NSRange, in attributed: NSAttributedString, page: Int) -> TextLine {
        guard let firstVisible = text.firstIndex(where: { !$0.isWhitespace }) else {
            return TextLine(text, page: page)
        }
        let location = range.location + text.utf16.distance(from: text.startIndex, to: firstVisible)
        guard let font = attributed.attribute(.font, at: location, effectiveRange: nil) as? PlatformFont else {
            return TextLine(text, page: page)
        }
        let name = font.fontName.lowercased()
        let traits = font.fontDescriptor.symbolicTraits
        #if canImport(AppKit)
        let isBold = traits.contains(.bold)
        let isMonospaced = traits.contains(.monoSpace)
        #else
        let isBold = traits.contains(.traitBold)
        let isMonospaced = traits.contains(.traitMonoSpace)
        #endif
        return TextLine(
            text,
            page: page,
            fontSize: Double(font.pointSize),
            isBold: isBold || name.contains("bold"),
            isMonospaced: isMonospaced || ["mono", "courier", "menlo", "consolas"].contains { name.contains($0) }
        )
    }
}
