// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "livetranslate-capture",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "livetranslate-capture",
            path: "Sources"
        )
    ]
)
