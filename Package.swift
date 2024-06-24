// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Embedding",
    platforms: [
        .macOS(.v13),
        .iOS(.v14),
    ],
    products: [
            .library(name: "Embedding", targets: ["Embedding"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "0.1.9"),
    ],
    targets: [
        .target(name: "Embedding", dependencies: [
            .product(name: "transformers", package: "swift-transformers"),
        ]),
        .testTarget(
            name: "EmbeddingTests",
            dependencies: [
                "Embedding",
                .product(name: "transformers", package: "swift-transformers"),
            ]),
    ]
)
