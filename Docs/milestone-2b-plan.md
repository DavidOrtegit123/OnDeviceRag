# OnDeviceRAG — Milestone 2b Implementation Plan (OCR for scanned pages)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Project rule (AGENTS.md → Mandatory Pre-Coding Protocol):** before each task's first code edit, post a short What / Why / How / Alternatives pitch and wait for the researcher's approval. Commit only when the researcher says so. Commit messages carry **no AI attribution lines**.

**Goal:** PDF pages that are pictures of text (scans, photographed worksheets) are read with Vision OCR and become searchable like any other page.

**Architecture:** Ingest stays line-based. After PDFKit extracts the text layer, every page with < 50 visible characters is rendered to an image and read with Vision's `RecognizeDocumentsRequest`. Each recognized line becomes a `TextLine` marked `isOCR`, with a font size derived from its box height, so the existing section detector, chunker and everything after it work unchanged. OCR lines are stored in the document's saved source, so re-chunking never re-runs OCR. Two new fields (`TextLine.isOCR`, `DocumentInfo.ocrPages`) decode with defaults, so 2a files keep loading.

**Tech Stack:** Swift 6.4 tools, Swift 6 mode, Swift Testing, PDFKit, Vision (`RecognizeDocumentsRequest`, iOS 26+/macOS 26+), Core Graphics.

**Spec:** `Packages/OnDeviceRAG/Docs/design.md` §5 Ingest step 1, §11 milestone 2b. Previous plans: `Docs/milestone-1-plan.md`, `Docs/milestone-2a-plan.md`.

## Global Constraints

- Everything from milestones 1 and 2a still applies: iOS 27 / macOS 27, tools 6.4, Swift 6 mode, **no network**, only dependency `swift-transformers` 1.3.4, the research app (`Prototype On-Device/`) and `evaluation/` are **not** modified, model-gated tests skip without the model, tests run with `swift test --package-path Packages/OnDeviceRAG`.
- On-disk format stays **version 1**: files written by 2a must still load, and new fields must decode with defaults when absent (no rebuild, no `loadIssues` entry).
- A page goes to OCR only if its text layer has < `PDFTextExtractor.minimumCharactersForTextPage` (50) visible characters. Pages with a usable text layer are never OCR'd.
- An OCR failure on one page never fails the document: that page keeps what it had and stays in `pagesWithoutText`. Only task cancellation throws.
- OCR runs one page at a time (memory) and checks for cancellation between pages.

## Evidence behind the choices (Mac spike, 2026-10-25, `/tmp/ocrspike`, not kept)

- API (compiled and run): `try await RecognizeDocumentsRequest().perform(on: cgImage)` → `[DocumentObservation]`; `observation.document.paragraphs: [DocumentObservation.Container.Text]`; `paragraph.lines` → line observations with `.transcript` and `.boundingBox.height` (fraction of image height); `document.title?.transcript`. `isTitle` on lines was `false` even for the title line, so the title is matched by text.
- Speed: first request ever on a machine **47 s** (one-time system model setup, cached across processes); afterwards **0.16–0.26 s per US-Letter page** at 2× (1224×1584 px). Rendering 3 ms.
- Rendering: drawing `PDFPage.draw(with: .mediaBox, to:)` into a plain `CGContext` bitmap works on both platforms and applies the page's rotation (a 90°-rotated page rendered 1584×1224 and read correctly).
- Bullet lists come through as paragraphs with their `•` markers.
- Line heights relative to the page's median line: body lines **0.86–1.23**, headings **1.43** (14 pt), **1.57** (18 pt), **1.71** (16 pt) against 11 pt body → threshold **1.3**; below it, a line is body text. (Raw heights would make body lines with tall glyphs look like headings under the detector's 1.2× rule.)
- OCR errors seen: a lone `l` read as `I` (with and without language correction); `x = y + 2` read as `x=y +2`. Recorded as limitations.

## Deviations from design.md (applied to design.md in Task 4)

1. §5 said OCR would use "paragraphs, lists, tables". Lists arrive inside `paragraphs` (verified). **Tables are not handled separately**: whether table text appears in `paragraphs` is unverified — listed as a limitation to check with real scans in milestone 3.
2. OCR is controlled by `RAGConfiguration.ocrScannedPages` (default `true`) and an `ocr:` parameter on `ingestPDF`, **not** by `IngestSettings` — settings are stored per document and a settings change triggers re-chunking from the source, which can't re-run OCR.
3. `DocumentIngestor.ingestPDF(at:…)` becomes `async throws` (Vision is async). `RAGEngine.addDocument(pdfAt:)` was already async, so the engine API doesn't change.
4. `DocumentInfo.pagesWithoutText` now means "still < 50 characters **after** OCR" (blank pages, pictures without text, OCR off or failed); new `DocumentInfo.ocrPages` lists pages whose text came from OCR.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `Sources/OnDeviceRAG/Ingest/TextLine.swift` | modify | `isOCR` flag; decoder that defaults it to `false` |
| `Sources/OnDeviceRAG/Model/Types.swift` | modify | `DocumentInfo.ocrPages`; decoder that defaults it to `[]` |
| `Sources/OnDeviceRAG/Ingest/DocumentIngestor.swift` | modify | derive `ocrPages` from lines; async `ingestPDF` / `ingest(pdf:)` with OCR |
| `Sources/OnDeviceRAG/Ingest/PDFTextExtractor.swift` | modify | shared `visibleCharacterCount(_:)` |
| `Sources/OnDeviceRAG/Ingest/PageOCR.swift` | create | render a page, run Vision, turn lines into `TextLine`s; fill empty pages |
| `Sources/OnDeviceRAG/Engine/RAGConfiguration.swift` | modify | `ocrScannedPages` |
| `Sources/OnDeviceRAG/Engine/RAGEngine.swift` | modify | pass `ocrScannedPages` through |
| `Sources/rag-index/main.swift` | modify | print OCR pages |
| `Tests/OnDeviceRAGTests/Support/FixturePDF.swift` | modify | `makeScanned` (pages as images, no text layer) |
| `Tests/OnDeviceRAGTests/OCRFormatTests.swift` | create | Task 1 tests |
| `Tests/OnDeviceRAGTests/PageOCRTests.swift` | create | Task 2 tests |
| `Tests/OnDeviceRAGTests/ScannedIngestTests.swift` | create | Task 3 tests |
| `Tests/OnDeviceRAGTests/PDFIngestTests.swift`, `SourceTests.swift` | modify | `try await` for the now-async `ingest(pdf:)` |
| `README.md`, `Docs/design.md` | modify | Task 4 |

---

### Task 1: OCR provenance fields that old files still decode

**Files:**
- Modify: `Sources/OnDeviceRAG/Ingest/TextLine.swift`
- Modify: `Sources/OnDeviceRAG/Model/Types.swift` (`DocumentInfo`)
- Modify: `Sources/OnDeviceRAG/Ingest/DocumentIngestor.swift` (`ingestText`, `ingest(lines:…)`, `assemble`)
- Test: `Tests/OnDeviceRAGTests/OCRFormatTests.swift`

**Interfaces:**
- Produces: `TextLine.isOCR: Bool` and `TextLine.init(_:page:fontSize:isBold:isMonospaced:isOCR:)` (`isOCR` defaults to `false`); `public let DocumentInfo.ocrPages: [Int]` (1-based, ascending), placed right after `pagesWithoutText` in the declaration (so the memberwise init order is `…, pagesWithoutText:, ocrPages:, mathHeavyPages:`).

- [ ] **Step 1: Write the failing tests** — `OCRFormatTests.swift`:

```swift
import Foundation
import Testing
@testable import OnDeviceRAG

struct OCRFormatTests {
    @Test func linesSavedBefore2bDecodeAsNotOCR() throws {
        let json = #"{"text":"Lists","page":2,"fontSize":18,"isBold":true,"isMonospaced":false}"#
        let line = try JSONDecoder().decode(TextLine.self, from: Data(json.utf8))
        #expect(line == TextLine("Lists", page: 2, fontSize: 18, isBold: true))
        #expect(line.isOCR == false)
    }

    @Test func infoSavedBefore2bDecodesWithNoOCRPages() throws {
        let info = DocumentIngestor.ingestText("## A\nAlpha.", title: "T", id: "t").info
        var object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(info)) as? [String: Any])
        #expect(object["ocrPages"] != nil)
        object["ocrPages"] = nil
        let decoded = try JSONDecoder().decode(DocumentInfo.self, from: JSONSerialization.data(withJSONObject: object))
        #expect(decoded == info)
    }

    @Test func ocrFlagRoundTrips() throws {
        let line = TextLine("Scanned", page: 3, isOCR: true)
        #expect(try JSONDecoder().decode(TextLine.self, from: JSONEncoder().encode(line)) == line)
    }

    @Test func ocrPagesComeFromTheLines() {
        let lines = [TextLine("Heading", page: 1, fontSize: 18), TextLine("Typed body text on the first page.", page: 1),
                     TextLine("Scanned body text on the second page.", page: 2, isOCR: true),
                     TextLine("More scanned text on the third page.", page: 3, isOCR: true)]
        let document = DocumentIngestor.ingest(lines: lines, outline: [], pageCount: 3, pagesWithoutText: [],
                                               title: "T", id: "t", settings: IngestSettings())
        #expect(document.info.ocrPages == [2, 3])
        #expect(DocumentIngestor.ingestText("## A\nAlpha.", title: "T").info.ocrPages == [])
    }
}
```

- [ ] **Step 2: Run — expect compile failure** (`extra argument 'isOCR' in call`, `value of type 'DocumentInfo' has no member 'ocrPages'`):
  `swift test --package-path Packages/OnDeviceRAG --filter OCRFormatTests`

- [ ] **Step 3: `TextLine`** — in `TextLine.swift`, replace the `TextLine` struct with:

```swift
/// One line of extracted text with the font facts section detection needs.
struct TextLine: Sendable, Hashable, Codable {
    var text: String
    /// 1-based page number.
    var page: Int
    var fontSize: Double
    var isBold: Bool
    var isMonospaced: Bool
    /// Read with OCR from a picture of the page, not from the PDF's text layer.
    var isOCR: Bool

    init(_ text: String, page: Int = 1, fontSize: Double = 11, isBold: Bool = false, isMonospaced: Bool = false,
         isOCR: Bool = false) {
        self.text = text
        self.page = page
        self.fontSize = fontSize
        self.isBold = isBold
        self.isMonospaced = isMonospaced
        self.isOCR = isOCR
    }

    var isBlank: Bool { text.allSatisfy(\.isWhitespace) }
}

extension TextLine {
    enum CodingKeys: String, CodingKey { case text, page, fontSize, isBold, isMonospaced, isOCR }

    /// Lines saved before OCR existed (milestone 2a files) have no `isOCR` key.
    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(try container.decode(String.self, forKey: .text),
                  page: try container.decode(Int.self, forKey: .page),
                  fontSize: try container.decode(Double.self, forKey: .fontSize),
                  isBold: try container.decode(Bool.self, forKey: .isBold),
                  isMonospaced: try container.decode(Bool.self, forKey: .isMonospaced),
                  isOCR: try container.decodeIfPresent(Bool.self, forKey: .isOCR) ?? false)
    }
}
```

(Verified in the spike: a decoder in an extension keeps the synthesized `encode(to:)` and memberwise init.)

- [ ] **Step 4: `DocumentInfo`** — in `Types.swift`, replace the two fields' comments and add `ocrPages` between them:

```swift
    /// Pages with < 50 non-whitespace characters even after OCR (blank, a picture without text,
    /// OCR off or failed).
    public let pagesWithoutText: [Int]
    /// Pages whose text was read with OCR (scanned pages).
    public let ocrPages: [Int]
    /// Pages where math symbols are > 5% of non-whitespace characters (extraction may be garbled).
    public let mathHeavyPages: [Int]
}

extension DocumentInfo {
    enum CodingKeys: String, CodingKey {
        case id, title, pageCount, sectionCount, chunkCount, sectionSource, pagesWithoutText, ocrPages, mathHeavyPages
    }

    /// Infos saved before OCR existed (milestone 2a files) have no `ocrPages` key.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(id: try container.decode(DocumentID.self, forKey: .id),
                  title: try container.decode(String.self, forKey: .title),
                  pageCount: try container.decode(Int.self, forKey: .pageCount),
                  sectionCount: try container.decode(Int.self, forKey: .sectionCount),
                  chunkCount: try container.decode(Int.self, forKey: .chunkCount),
                  sectionSource: try container.decode(SectionSource.self, forKey: .sectionSource),
                  pagesWithoutText: try container.decode([Int].self, forKey: .pagesWithoutText),
                  ocrPages: try container.decodeIfPresent([Int].self, forKey: .ocrPages) ?? [],
                  mathHeavyPages: try container.decode([Int].self, forKey: .mathHeavyPages))
    }
}
```

- [ ] **Step 5: `DocumentIngestor`** — `assemble` takes and stores `ocrPages`:
  - signature: `private static func assemble(sections detected: [DetectedSection], source: SectionSource, title: String, id: DocumentID, pageCount: Int, pagesWithoutText: [Int], ocrPages: [Int], mathHeavyPages: [Int], settings: IngestSettings, documentSource: DocumentSource) -> IngestedDocument`
  - the `DocumentInfo(...)` call inside it: `DocumentInfo(id: id, title: title, pageCount: pageCount, sectionCount: sections.count, chunkCount: chunks.count, sectionSource: source, pagesWithoutText: pagesWithoutText, ocrPages: ocrPages, mathHeavyPages: mathHeavyPages)`
  - `ingestText`'s call: add `ocrPages: [],` after `pagesWithoutText: [],`.
  - `ingest(lines:…)`'s call: add `ocrPages: Set(lines.filter(\.isOCR).map(\.page)).sorted(),` after `pagesWithoutText: pagesWithoutText,`.

- [ ] **Step 6: Run all tests — expect PASS** (`swift test --package-path Packages/OnDeviceRAG`; 72 old + 4 new).
- [ ] **Step 7: Mutation check** — temporarily change both `decodeIfPresent(...) ?? …` to `decode(...)`: `linesSavedBefore2bDecodeAsNotOCR` and `infoSavedBefore2bDecodesWithNoOCRPages` must fail. Restore.
- [ ] **Step 8: Commit (when approved)** — `git commit -m "OnDeviceRAG: OCR provenance fields, compatible with 2a files"`

---

### Task 2: `PageOCR` — read one scanned page

**Files:**
- Create: `Sources/OnDeviceRAG/Ingest/PageOCR.swift`
- Modify: `Sources/OnDeviceRAG/Ingest/PDFTextExtractor.swift` (shared character count)
- Modify: `Tests/OnDeviceRAGTests/Support/FixturePDF.swift` (`makeScanned`)
- Test: `Tests/OnDeviceRAGTests/PageOCRTests.swift`

**Interfaces:**
- Consumes: `TextLine(_:page:fontSize:isBold:isMonospaced:isOCR:)` (Task 1).
- Produces: `PageOCR.lines(of page: PDFPage, pageNumber: Int) async throws -> [TextLine]` (paragraphs separated by one blank `TextLine`, all lines `isOCR`); `PageOCR.fill(_ extracted: (lines: [TextLine], pagesWithoutText: [Int]), in document: PDFDocument) async throws -> (lines: [TextLine], pagesWithoutText: [Int])` (used by Task 3); `PageOCR.bodyFontSize`; `PDFTextExtractor.visibleCharacterCount(_ lines: [TextLine]) -> Int`; test helper `FixturePDF.makeScanned(pages: [[Line]]) -> PDFDocument`.

- [ ] **Step 1: Add `makeScanned` to `FixturePDF`** — replace the body of `FixturePDF` (keep `Line` as is) so both builders share the line drawing:

```swift
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
```

- [ ] **Step 2: Write the failing tests** — `PageOCRTests.swift`. Expected texts avoid lone letters (OCR reads `l` as `I`):

```swift
#if os(macOS)
import AppKit
import PDFKit
import Testing
@testable import OnDeviceRAG

struct PageOCRTests {
    private let body = ["Lists are encoded as items between markers.", "Nested lists are allowed in this format.",
                        "Every item keeps its original order."]

    @Test func scannedFixtureHasNoTextLayer() {
        let pdf = FixturePDF.makeScanned(pages: [body.map(FixturePDF.Line.body)])
        #expect(PDFTextExtractor.lines(from: pdf).pagesWithoutText == [1])
    }

    @Test func readsTheTextOfAScannedPage() async throws {
        let pdf = FixturePDF.makeScanned(pages: [body.map(FixturePDF.Line.body)])
        let lines = try await PageOCR.lines(of: try #require(pdf.page(at: 0)), pageNumber: 4)
        let texts = lines.filter { !$0.isBlank }.map { $0.text.trimmingCharacters(in: .whitespaces) }
        #expect(texts == body)
        #expect(lines.allSatisfy { $0.isOCR && $0.page == 4 && !$0.isMonospaced })
    }

    @Test func headingIsLargerAndBodyLinesShareOneSize() async throws {
        let pdf = FixturePDF.makeScanned(pages: [[.heading("Chapter Two Lists")] + body.map(FixturePDF.Line.body)])
        let lines = try await PageOCR.lines(of: try #require(pdf.page(at: 0)), pageNumber: 1).filter { !$0.isBlank }
        let heading = try #require(lines.first { $0.text.contains("Chapter") })
        let bodyLines = lines.filter { !$0.text.contains("Chapter") }
        #expect(bodyLines.count == 3)
        #expect(bodyLines.allSatisfy { $0.fontSize == PageOCR.bodyFontSize })
        #expect(heading.fontSize >= 1.3 * PageOCR.bodyFontSize)
    }

    @Test func blankPageGivesNoText() async throws {
        let pdf = FixturePDF.makeScanned(pages: [[]])
        let lines = try await PageOCR.lines(of: try #require(pdf.page(at: 0)), pageNumber: 1)
        #expect(lines.allSatisfy(\.isBlank))
    }

    @Test func rotatedScanIsReadUpright() async throws {
        let pdf = FixturePDF.makeScanned(pages: [body.map(FixturePDF.Line.body)])
        let page = try #require(pdf.page(at: 0))
        page.rotation = 90
        let texts = try await PageOCR.lines(of: page, pageNumber: 1).filter { !$0.isBlank }.map(\.text)
        #expect(texts.contains { $0.contains("Nested lists are allowed") })
    }

    @Test func fillReplacesOnlyEmptyPagesAndKeepsPageOrder() async throws {
        // Page 1 typed, page 2 scanned, page 3 blank scan.
        let typed = FixturePDF.make(pages: [[.heading("Typed"), .body("This page has a real text layer with enough characters.")]])
        let scanned = FixturePDF.makeScanned(pages: [body.map(FixturePDF.Line.body), []])
        for index in 0..<scanned.pageCount {
            typed.insert(try #require(scanned.page(at: index)), at: typed.pageCount)
        }
        let extracted = PDFTextExtractor.lines(from: typed)
        #expect(extracted.pagesWithoutText == [2, 3])

        let filled = try await PageOCR.fill(extracted, in: typed)
        #expect(filled.pagesWithoutText == [3])
        #expect(filled.lines.map(\.page) == filled.lines.map(\.page).sorted())
        #expect(filled.lines.filter { $0.page == 1 }.allSatisfy { !$0.isOCR })
        #expect(filled.lines.filter { $0.page == 2 }.allSatisfy(\.isOCR))
        #expect(filled.lines.contains { $0.page == 2 && $0.text.contains("Every item keeps its original order") })
    }
}
#endif
```

- [ ] **Step 3: Run — expect compile failure** (`cannot find 'PageOCR' in scope`): `swift test --package-path Packages/OnDeviceRAG --filter PageOCRTests`

- [ ] **Step 4: Shared character count** — in `PDFTextExtractor.swift`, add:

```swift
    /// Non-whitespace characters in the lines (a page with < `minimumCharactersForTextPage` has no usable text).
    static func visibleCharacterCount(_ lines: [TextLine]) -> Int {
        lines.reduce(0) { $0 + $1.text.filter { !$0.isWhitespace }.count }
    }
```

and in `lines(from:)` replace `let characters = pageLines.reduce(0) { $0 + $1.text.filter { !$0.isWhitespace }.count }` with `let characters = visibleCharacterCount(pageLines)`.

- [ ] **Step 5: Create `PageOCR.swift`**

```swift
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
```

- [ ] **Step 6: Run all tests — expect PASS** (Task 1 count + 6 new). OCR tests take ~0.2 s per page; on a machine that has never run Vision document recognition, the first one can take ~45 s.
- [ ] **Step 7: Mutation checks** (one at a time, restore after each):
  1. `headingHeightRatio = 0.5` (every line sized by raw height) → `headingIsLargerAndBodyLinesShareOneSize` fails.
  2. In `fill`, return `extracted.pagesWithoutText` instead of `stillEmpty` → `fillReplacesOnlyEmptyPagesAndKeepsPageOrder` fails (page 2 would still be reported as without text).
  3. In `fill`, iterate over every page (`1...document.pageCount`) instead of `extracted.pagesWithoutText` → the same test fails (page 1's typed lines become OCR lines).
- [ ] **Step 8: Commit (when approved)** — `git commit -m "OnDeviceRAG: read scanned PDF pages with Vision OCR"`

---

### Task 3: OCR in PDF ingest, the engine and `rag-index`

**Files:**
- Modify: `Sources/OnDeviceRAG/Ingest/DocumentIngestor.swift` (`ingestPDF`, `ingest(pdf:…)`)
- Modify: `Sources/OnDeviceRAG/Engine/RAGConfiguration.swift`, `Sources/OnDeviceRAG/Engine/RAGEngine.swift`
- Modify: `Sources/rag-index/main.swift`
- Modify: `Tests/OnDeviceRAGTests/PDFIngestTests.swift`, `Tests/OnDeviceRAGTests/SourceTests.swift` (async calls)
- Test: `Tests/OnDeviceRAGTests/ScannedIngestTests.swift`

**Interfaces:**
- Consumes: `PageOCR.fill(_:in:)` (Task 2), `DocumentInfo.ocrPages` (Task 1).
- Produces: `public static func ingestPDF(at url: URL, title: String? = nil, id: DocumentID = UUID().uuidString, settings: IngestSettings = IngestSettings(), ocr: Bool = true) async throws -> IngestedDocument`; `static func ingest(pdf document: PDFDocument, title: String, id: DocumentID, settings: IngestSettings, ocr: Bool = true) async throws -> IngestedDocument`; `public var RAGConfiguration.ocrScannedPages: Bool` (default `true`).

- [ ] **Step 1: Write the failing tests** — `ScannedIngestTests.swift`:

```swift
#if os(macOS)
import AppKit
import Foundation
import PDFKit
import Testing
@testable import OnDeviceRAG

private let zorpScan: [[FixturePDF.Line]] = [[
    FixturePDF.Line(text: "Encoding Lists", font: NSFont(name: "Helvetica-Bold", size: 20)!),
    .body("Lists are encoded as items between markers."),
    .body("Nested lists are allowed in this format."),
    .body("Every item keeps its original order."),
    FixturePDF.Line(text: "Encoding Numbers", font: NSFont(name: "Helvetica-Bold", size: 20)!),
    .body("Numbers are written with a leading marker."),
    .body("Negative numbers keep their minus sign."),
]]

private func writeTemporary(_ pdf: PDFDocument) throws -> URL {
    let url = FileManager.default.temporaryDirectory.appending(path: "scan-\(UUID().uuidString).pdf")
    try #require(pdf.write(to: url))
    return url
}

struct ScannedIngestTests {
    @Test func scannedPDFBecomesSectionsAndChunks() async throws {
        let document = try await DocumentIngestor.ingest(pdf: FixturePDF.makeScanned(pages: zorpScan), title: "ZORP", id: "z",
                                                         settings: IngestSettings())
        #expect(document.info.ocrPages == [1])
        #expect(document.info.pagesWithoutText == [])
        #expect(document.info.sectionSource == .headings)
        #expect(document.sections.map(\.title) == ["Encoding Lists", "Encoding Numbers"])
        #expect(document.chunks.contains { $0.text.contains("Every item keeps its original order.") })
    }

    @Test func ocrOffLeavesScannedPagesEmpty() async throws {
        let document = try await DocumentIngestor.ingest(pdf: FixturePDF.makeScanned(pages: zorpScan), title: "ZORP", id: "z",
                                                         settings: IngestSettings(), ocr: false)
        #expect(document.info.ocrPages == [])
        #expect(document.info.pagesWithoutText == [1])
        #expect(document.chunks.isEmpty)
    }

    @Test func reingestKeepsOCRTextWithoutRunningOCRAgain() async throws {
        let original = try await DocumentIngestor.ingest(pdf: FixturePDF.makeScanned(pages: zorpScan), title: "ZORP", id: "z",
                                                         settings: IngestSettings())
        // reingest is synchronous and has no PDF: it can only use the stored OCR lines.
        let again = DocumentIngestor.reingest(original.source, title: "ZORP", id: "z", settings: IngestSettings())
        #expect(again.info == original.info)
        #expect(again.chunks == original.chunks)
    }

    @Test func engineFindsTextOnAScannedPage() async throws {
        let url = try writeTemporary(FixturePDF.makeScanned(pages: zorpScan))
        defer { try? FileManager.default.removeItem(at: url) }
        var configuration = RAGConfiguration(embedder: FakeEmbedder())
        configuration.tokenCounter = EstimatingTokenCounter()
        let engine = RAGEngine(configuration: configuration)
        let info = try await engine.addDocument(pdfAt: url, title: "ZORP")
        #expect(info.ocrPages == [1])
        let top = try await engine.search("negative numbers minus sign").first
        #expect(top?.chunk.sectionTitle == "Encoding Numbers")
    }

    @Test func engineRespectsOCRSwitch() async throws {
        let url = try writeTemporary(FixturePDF.makeScanned(pages: zorpScan))
        defer { try? FileManager.default.removeItem(at: url) }
        var configuration = RAGConfiguration(embedder: FakeEmbedder())
        configuration.ocrScannedPages = false
        let info = try await RAGEngine(configuration: configuration).addDocument(pdfAt: url, title: "ZORP")
        #expect(info.ocrPages == [])
        #expect(info.pagesWithoutText == [1])
    }
}
#endif
```

- [ ] **Step 2: Run — expect compile failure** (`extra argument 'ocr' in call`, `no member 'ocrScannedPages'`): `swift test --package-path Packages/OnDeviceRAG --filter ScannedIngestTests`

- [ ] **Step 3: `DocumentIngestor`** — replace `ingestPDF` and `ingest(pdf:…)`:

```swift
    /// Reads a PDF; pages without a text layer (scans) are read with OCR unless `ocr` is false.
    public static func ingestPDF(at url: URL, title: String? = nil, id: DocumentID = UUID().uuidString,
                                 settings: IngestSettings = IngestSettings(), ocr: Bool = true) async throws -> IngestedDocument {
        guard let document = PDFDocument(url: url) else { throw IngestError.unreadablePDF(url) }
        let pdfTitle = document.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String
        let fallbackTitle = pdfTitle.flatMap { $0.isEmpty ? nil : $0 } ?? url.deletingPathExtension().lastPathComponent
        return try await ingest(pdf: document, title: title ?? fallbackTitle, id: id, settings: settings, ocr: ocr)
    }
```

```swift
    static func ingest(pdf document: PDFDocument, title: String, id: DocumentID, settings: IngestSettings,
                       ocr: Bool = true) async throws -> IngestedDocument {
        var extracted = PDFTextExtractor.lines(from: document)
        if ocr, !extracted.pagesWithoutText.isEmpty {
            extracted = try await PageOCR.fill(extracted, in: document)
        }
        return ingest(lines: extracted.lines, outline: PDFTextExtractor.outline(of: document), pageCount: document.pageCount,
                      pagesWithoutText: extracted.pagesWithoutText, title: title, id: id, settings: settings)
    }
```

- [ ] **Step 4: Configuration and engine**
  - `RAGConfiguration.swift`, after `public var ingest = IngestSettings()`:

```swift
    /// Read PDF pages without a text layer (scans) with Vision OCR (~0.2 s per page on a Mac).
    /// Not part of `IngestSettings`: re-chunking a stored document can't re-run OCR.
    public var ocrScannedPages = true
```

  - `RAGEngine.addDocument(pdfAt:title:)` body becomes:

```swift
        try await addDocument(DocumentIngestor.ingestPDF(at: url, title: title, settings: configuration.ingest,
                                                         ocr: configuration.ocrScannedPages))
```

- [ ] **Step 5: Existing callers of the now-async `ingest(pdf:)` / `ingestPDF`**
  - `PDFIngestTests.swift`: change these test signatures and calls —
    - `headingsBecomeSectionsAndCodeStaysOneChunk()` → `headingsBecomeSectionsAndCodeStaysOneChunk() async throws`
    - `outlineWinsOverHeadings()` → `outlineWinsOverHeadings() async throws`
    - `emptyPageIsReportedAsWithoutText()` → `emptyPageIsReportedAsWithoutText() async throws`
    - `mathHeavyPageIsFlagged()` → `mathHeavyPageIsFlagged() async throws`
    - in those four, `let document = DocumentIngestor.ingest(pdf:` → `let document = try await DocumentIngestor.ingest(pdf:`
    - `unreadableFileThrows()` becomes:

```swift
    @Test func unreadableFileThrows() async {
        let url = URL(fileURLWithPath: "/nonexistent/file.pdf")
        await #expect(throws: IngestError.unreadablePDF(url)) { try await DocumentIngestor.ingestPDF(at: url) }
    }
```

  - `SourceTests.swift`: `pdfSourceReingestsIdenticallyWithoutTheFile()` → `pdfSourceReingestsIdenticallyWithoutTheFile() async throws`, and `let original = DocumentIngestor.ingest(pdf:` → `let original = try await DocumentIngestor.ingest(pdf:`.
  - `Sources/rag-index/main.swift`: the `build` loop's call `try await engine.addDocument(DocumentIngestor.ingestPDF(at: url, id: id))` compiles unchanged (the outer `try await` covers it). After the `pagesWithoutText` line, add:

```swift
        if !info.ocrPages.isEmpty { line += ", OCR pages: \(info.ocrPages)" }
```

- [ ] **Step 6: Run all tests — expect PASS** (previous count + 5 new), then `swift build --package-path Packages/OnDeviceRAG` (CLI compiles).
- [ ] **Step 7: Mutation checks** (one at a time, restore after each):
  1. In `ingest(pdf:…)`, ignore `ocr` (`if !extracted.pagesWithoutText.isEmpty`) → `ocrOffLeavesScannedPagesEmpty` and `engineRespectsOCRSwitch` fail.
  2. In `RAGEngine.addDocument(pdfAt:)`, drop `ocr: configuration.ocrScannedPages` → `engineRespectsOCRSwitch` fails.
  3. In `PageOCR.lines`, remove the blank separator line between paragraphs → report whether `scannedPDFBecomesSectionsAndChunks` still passes (it may: the chunker also splits at sentence boundaries). If it passes, keep the separator anyway (it keeps Vision's paragraph boundaries for chunking) and say so in the report.
- [ ] **Step 8: Manual check with the real model on a real scan** (from the repo root), if the researcher has a scanned or photographed PDF (e.g. a worksheet) to try:

```bash
M=~/Developer/coreai-models/exports/bge_small_buckets_float16.aimodel
swift run -c release --package-path Packages/OnDeviceRAG rag-index build --model $M --output /tmp/scan.ragindex <scanned.pdf>
swift run -c release --package-path Packages/OnDeviceRAG rag-index query --model $M --index /tmp/scan.ragindex "<a question about the scan>"
```

  Expected: `build` prints `OCR pages: [...]`; report sections found, per-page time, and any visible OCR errors. If no real scan is available, say so — don't fabricate one.
- [ ] **Step 9: iOS build** — from `Packages/OnDeviceRAG`: `xcodebuild -scheme OnDeviceRAG -destination 'generic/platform=iOS' -derivedDataPath /tmp/odrag-ios build` → `BUILD SUCCEEDED`.
- [ ] **Step 10: Commit (when approved)** — `git commit -m "OnDeviceRAG: OCR scanned pages during PDF ingest"`

---

### Task 4: Documentation and full verification

**Files:**
- Modify: `Packages/OnDeviceRAG/README.md`
- Modify: `Packages/OnDeviceRAG/Docs/design.md`

- [ ] **Step 1: README**
  - "How it works" step 1: after "PDF text is read with its fonts," add "pages without a text layer (scans) are read with Vision OCR,".
  - Limitations: replace "Scanned pages are reported (`DocumentInfo.pagesWithoutText`) but not yet read (OCR is planned)." with: "Scanned pages are read with Vision OCR (`DocumentInfo.ocrPages`; ~0.2 s per page on a Mac; the first OCR ever on a device can take ~45 s while the system prepares its model). OCR can misread lone letters (`l` → `I`) and drops spacing in math (`x = y + 2` → `x=y +2`); tables aren't handled specially. Turn it off with `RAGConfiguration.ocrScannedPages = false`."
- [ ] **Step 2: design.md**
  - Status line: "milestones 1, 2a and 2b implemented and Mac-tested".
  - §5 step 1: replace "— **milestone 2**; milestone 1 lists these pages in `DocumentInfo.pagesWithoutText`." with the implemented behavior: rendered at 2× (≤ 4,000 px), `RecognizeDocumentsRequest` paragraphs → `TextLine`s marked `isOCR` (blank line between paragraphs), sizes snapped (< 1.3 × median line height = body), the recognized title marked bold; OCR lines stored in the source (re-chunking never re-runs OCR); `ocrPages` / `pagesWithoutText` meanings; switch `RAGConfiguration.ocrScannedPages`; a failing page never fails the document.
  - §10: add the OCR limitations (lone letters, math spacing, tables unverified, first-run delay, iPhone speed unmeasured).
  - §11: mark 2b done (date), plan path `Docs/milestone-2b-plan.md`; 2c next.
- [ ] **Step 3: Full verification**
  - `swift test --package-path Packages/OnDeviceRAG` — all pass, model-gated tests included (not skipped).
  - iOS build as in Task 3 Step 9 → `BUILD SUCCEEDED`.
  - From the repo root: `git diff --stat main -- "Prototype On-Device" evaluation` → empty.
  - Files written before 2b still load: `OCRFormatTests` covers the decoding; in addition, if `/tmp/manuals.ragindex` (built in 2a Task 5) still exists, `rag-index query --index /tmp/manuals.ragindex "How do I encode a list in ZORP-7?"` must print the same sections as in 2a (ZORP-7 first, 1,906 tokens) with no rebuild.
- [ ] **Step 4: Commit (when approved)** — `git commit -m "OnDeviceRAG: milestone 2b docs"`

---

## Self-review notes

- **Spec coverage (design §5 step 1, §11 2b):** detect pages with < 50 characters → existing extractor + Task 2 `fill`; render + `RecognizeDocumentsRequest` → Task 2; paragraphs/lists → Task 2 (lists verified in the spike); tables → deviation 1 (limitation); results feed sections/chunks → Task 3; stored for re-chunking → Tasks 1 + 3 (`reingestKeepsOCRTextWithoutRunningOCRAgain`); 2a files still load → Task 1; docs → Task 4.
- **Not covered here:** OCR language selection (Vision's automatic detection is used; the embedder is English-only anyway), OCR of images imported as photos (that's the app's job in 2c — it already has its own Vision OCR for photos), progress reporting for long scans (a 300-page scan ≈ 1 min on a Mac; add a progress callback only if 2c needs it), measuring OCR speed on the iPhone (milestone 4).
- **Type consistency:** `TextLine(_:page:fontSize:isBold:isMonospaced:isOCR:)`, `DocumentInfo.ocrPages`, `PageOCR.lines(of:pageNumber:)`, `PageOCR.fill(_:in:)`, `PageOCR.bodyFontSize`, `PageOCR.headingHeightRatio`, `PDFTextExtractor.visibleCharacterCount(_:)`, `FixturePDF.makeScanned(pages:)`, `ingest(pdf:title:id:settings:ocr:)`, `ingestPDF(at:title:id:settings:ocr:)`, `RAGConfiguration.ocrScannedPages` — used identically in every task.
