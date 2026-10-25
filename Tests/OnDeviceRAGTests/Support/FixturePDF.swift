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
            draw(page, in: context)
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

    /// Like `make`, but each page is a 144-dpi picture of the text with no text layer — a scan.
    static func makeScanned(pages: [[Line]]) -> PDFDocument {
        let data = NSMutableData()
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let context = CGContext(consumer: CGDataConsumer(data: data)!, mediaBox: &box, nil)!
        let scale = 2.0
        for page in pages {
            let width = Int(box.width * scale), height = Int(box.height * scale)
            let bitmap = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                   space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            bitmap.setFillColor(CGColor(gray: 1, alpha: 1))
            bitmap.fill(CGRect(x: 0, y: 0, width: width, height: height))
            bitmap.scaleBy(x: scale, y: scale)
            draw(page, in: bitmap)
            context.beginPDFPage(nil)
            context.draw(bitmap.makeImage()!, in: box)
            context.endPDFPage()
        }
        context.closePDF()
        return PDFDocument(data: data as Data)!
    }

    private static func draw(_ lines: [Line], in context: CGContext) {
        var y = 740.0
        for line in lines {
            let attributed = NSAttributedString(string: line.text, attributes: [.font: line.font])
            context.textPosition = CGPoint(x: 72, y: y)
            CTLineDraw(CTLineCreateWithAttributedString(attributed), context)
            y -= line.font.pointSize * 1.8
        }
    }
}
#endif
