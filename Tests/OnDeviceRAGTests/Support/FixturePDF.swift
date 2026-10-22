#if os(macOS)
import AppKit
import CoreText
import PDFKit

/// Builds small PDFs in memory with chosen fonts, one CoreText line per text line.
enum FixturePDF {
    struct Line {
        var text: String
        var font: NSFont
        // Named fonts, not the system UI font: PDFKit can't map the system font's private
        // ".SF…" name back when reading, and reports it as Times New Roman (no bold/mono).
        static func body(_ text: String) -> Line { Line(text: text, font: NSFont(name: "Helvetica", size: 11)!) }
        static func heading(_ text: String) -> Line { Line(text: text, font: NSFont(name: "Helvetica-Bold", size: 18)!) }
        static func code(_ text: String) -> Line { Line(text: text, font: NSFont(name: "Menlo-Regular", size: 10)!) }
    }

    static func make(pages: [[Line]], outline: [(title: String, page: Int)] = []) -> PDFDocument {
        let data = NSMutableData()
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let context = CGContext(consumer: CGDataConsumer(data: data)!, mediaBox: &box, nil)!
        for page in pages {
            context.beginPDFPage(nil)
            var y = 740.0
            for line in page {
                let attributed = NSAttributedString(string: line.text, attributes: [.font: line.font])
                context.textPosition = CGPoint(x: 72, y: y)
                CTLineDraw(CTLineCreateWithAttributedString(attributed), context)
                y -= line.font.pointSize * 1.8
            }
            context.endPDFPage()
        }
        context.closePDF()
        let document = PDFDocument(data: data as Data)!
        if !outline.isEmpty {
            let root = PDFOutline()
            for (position, entry) in outline.enumerated() {
                let item = PDFOutline()
                item.label = entry.title
                item.destination = PDFDestination(page: document.page(at: entry.page - 1)!, at: CGPoint(x: 0, y: 792))
                root.insertChild(item, at: position)
            }
            document.outlineRoot = root
        }
        return document
    }
}
#endif
