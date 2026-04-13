import Foundation
import ScreenCaptureKit
import AVFoundation

struct Config {
    var sampleRate: Int = 48000
    var channels: Int = 1
    var format: String = "f32le"
    var listSources: Bool = false
    var device: String? = nil
}

func parseArgs() -> Config {
    var config = Config()
    var args = Array(CommandLine.arguments.dropFirst())
    var i = 0

    while i < args.count {
        switch args[i] {
        case "--sample-rate":
            i += 1
            if i < args.count, let rate = Int(args[i]) {
                config.sampleRate = rate
            }
        case "--channels":
            i += 1
            if i < args.count, let ch = Int(args[i]) {
                config.channels = ch
            }
        case "--format":
            i += 1
            if i < args.count {
                config.format = args[i]
            }
        case "--device":
            i += 1
            if i < args.count {
                config.device = args[i]
            }
        case "--list-sources":
            config.listSources = true
        case "--help":
            printUsage()
            exit(0)
        default:
            break
        }
        i += 1
    }
    return config
}

func printUsage() {
    fputs("""
    Usage: livetranslate-capture [OPTIONS]

    Options:
      --sample-rate <HZ>   Sample rate (default: 48000)
      --channels <N>       Number of channels (default: 1)
      --format <FMT>       Output format: f32le (default)
      --device <NAME>      Specific audio device
      --list-sources       List available audio sources
      --help               Show this help

    Outputs raw PCM audio to stdout.
    """, stderr)
}

// Entry point: top-level async execution via main.swift
let _entryConfig = parseArgs()
if _entryConfig.listSources {
    await LiveTranslateCapture.listAudioSources()
} else {
    do {
        try await LiveTranslateCapture.startCapture(config: _entryConfig)
    } catch {
        fputs("Error: \(error.localizedDescription)\n", stderr)
        exit(1)
    }
}

struct LiveTranslateCapture {
    static func listAudioSources() async {
        do {
            let content = try await SCShareableContent.current
            print("Available audio sources:")
            for app in content.applications {
                print("  - \(app.applicationName)")
            }
        } catch {
            fputs("Error listing sources: \(error.localizedDescription)\n", stderr)
            exit(2)
        }
    }

    static func startCapture(config: Config) async throws {
        let capture = try await AudioCapture(config: config)

        signal(SIGTERM) { _ in
            Task { await AudioCapture.shared?.stop() }
        }
        signal(SIGINT) { _ in
            Task { await AudioCapture.shared?.stop() }
        }

        try await capture.start()

        await withCheckedContinuation { continuation in
            Task {
                await capture.setOnStop { continuation.resume() }
            }
        }
    }
}
