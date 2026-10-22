// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "OnDeviceRAG",
    platforms: [.iOS(.v27), .macOS(.v27)],
    products: [
        .library(name: "OnDeviceRAG", targets: ["OnDeviceRAG"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.4"),
    ],
    targets: [
        .target(
            name: "OnDeviceRAG",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")]
        ),
        .testTarget(
            name: "OnDeviceRAGTests",
            dependencies: ["OnDeviceRAG"],
            resources: [.copy("Fixtures")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
