import Foundation
import OnDeviceRAG

// Builds and queries OnDeviceRAG indexes on a Mac, with the same code path the apps use.

let usage = """
usage:
  rag-index build --model <bge.aimodel> --output <file.ragindex> <input.pdf|.md|.txt>...
  rag-index query --model <bge.aimodel> --index <file.ragindex> [--budget <tokens>] <question...>
"""

struct Options {
    var model: URL?
    var output: URL?
    var index: URL?
    var budget = 2000
    var positional: [String] = []
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

func parse(_ arguments: ArraySlice<String>) -> Options {
    var options = Options()
    var remaining = arguments
    while let argument = remaining.popFirst() {
        switch argument {
        case "--model", "--output", "--index", "--budget":
            guard let value = remaining.popFirst() else { fail("missing value for \(argument)\n\n\(usage)") }
            switch argument {
            case "--model": options.model = URL(fileURLWithPath: value)
            case "--output": options.output = URL(fileURLWithPath: value)
            case "--index": options.index = URL(fileURLWithPath: value)
            default:
                guard let budget = Int(value), budget > 0 else { fail("--budget needs a positive number") }
                options.budget = budget
            }
        default:
            options.positional.append(argument)
        }
    }
    return options
}

let arguments = CommandLine.arguments.dropFirst()
guard let command = arguments.first else { fail(usage) }
let options = parse(arguments.dropFirst())
guard let modelURL = options.model else { fail("--model is required\n\n\(usage)") }

let clock = ContinuousClock()
let loadStart = clock.now
let engine = RAGEngine(configuration: RAGConfiguration(embedder: try await BGESmallEmbedder(modelURL: modelURL)))
print("model loaded in \(clock.now - loadStart)")

switch command {
case "build":
    guard let output = options.output, !options.positional.isEmpty else { fail(usage) }
    for path in options.positional {
        let url = URL(fileURLWithPath: path)
        let id = url.deletingPathExtension().lastPathComponent
        let start = clock.now
        let info: DocumentInfo
        if url.pathExtension.lowercased() == "pdf" {
            info = try await engine.addDocument(DocumentIngestor.ingestPDF(at: url, id: id))
        } else {
            let text = try String(contentsOf: url, encoding: .utf8)
            info = try await engine.addDocument(DocumentIngestor.ingestText(text, title: id, id: id))
        }
        var line = "\(id): \(info.sectionCount) sections (\(info.sectionSource.rawValue)), \(info.chunkCount) chunks, \(info.pageCount) pages, \(clock.now - start)"
        if !info.pagesWithoutText.isEmpty { line += ", pages without text: \(info.pagesWithoutText)" }
        if !info.mathHeavyPages.isEmpty { line += ", math-heavy pages: \(info.mathHeavyPages)" }
        print(line)
    }
    try await engine.exportIndex(to: output)
    let size = (try? FileManager.default.attributesOfItem(atPath: output.path)[.size] as? Int) ?? 0
    print("wrote \(output.path) (\(size / 1024) KB)")

case "query":
    guard let indexURL = options.index, !options.positional.isEmpty else { fail(usage) }
    try await engine.addPrebuiltIndex(at: indexURL)
    let question = options.positional.joined(separator: " ")
    let start = clock.now
    let context = try await engine.context(for: question, tokenBudget: options.budget)
    print("query took \(clock.now - start); \(context.tokenCount) tokens\(context.tokenCountIsEstimate ? " (estimate)" : "")")
    for section in context.sections {
        print("- \(section.title) (\(section.documentTitle), \(section.pages.label))\(section.isWholeSection ? "" : " [window]")")
    }
    print("\n\(context.text)")

default:
    fail(usage)
}
