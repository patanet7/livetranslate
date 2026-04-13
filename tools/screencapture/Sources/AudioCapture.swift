import Foundation
import ScreenCaptureKit
import AVFoundation
import CoreMedia

// Separate NSObject-based delegate so the actor doesn't need to inherit NSObject
private final class StreamDelegate: NSObject, SCStreamDelegate, SCStreamOutput, @unchecked Sendable {
    weak var capture: AudioCapture?

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }

        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }

        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>?

        let status = CMBlockBufferGetDataPointer(
            blockBuffer,
            atOffset: 0,
            lengthAtOffsetOut: nil,
            totalLengthOut: &length,
            dataPointerOut: &dataPointer
        )

        guard status == kCMBlockBufferNoErr, let data = dataPointer else { return }

        // Write raw PCM to stdout
        let bytesWritten = fwrite(data, 1, length, stdout)
        fflush(stdout)

        if bytesWritten != length {
            fputs("Warning: Only wrote \(bytesWritten) of \(length) bytes\n", stderr)
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("Stream stopped with error: \(error.localizedDescription)\n", stderr)
        Task { await self.capture?.stop() }
    }
}

actor AudioCapture {
    static var shared: AudioCapture?

    private let config: Config
    private var stream: SCStream?
    private var delegate: StreamDelegate?
    private var isRunning = false
    var onStop: (() -> Void)?

    init(config: Config) {
        self.config = config
        AudioCapture.shared = self
    }

    func start() async throws {
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(
                false,
                onScreenWindowsOnly: false
            )
        } catch {
            throw CaptureError.permissionDenied
        }

        guard let display = content.displays.first else {
            throw CaptureError.noDisplays
        }

        let streamConfig = SCStreamConfiguration()
        streamConfig.capturesAudio = true
        streamConfig.excludesCurrentProcessAudio = false
        streamConfig.sampleRate = config.sampleRate
        streamConfig.channelCount = config.channels

        // Minimal video config (required but ignored)
        streamConfig.width = 2
        streamConfig.height = 2
        streamConfig.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let filter = SCContentFilter(display: display, excludingWindows: [])

        let del = StreamDelegate()
        del.capture = self
        self.delegate = del

        stream = SCStream(filter: filter, configuration: streamConfig, delegate: del)

        try stream?.addStreamOutput(del, type: .audio, sampleHandlerQueue: .global(qos: .userInteractive))

        try await stream?.startCapture()
        isRunning = true

        fputs("Capturing system audio at \(config.sampleRate)Hz, \(config.channels) channel(s)\n", stderr)
    }

    func stop() async {
        guard isRunning else { return }
        isRunning = false

        try? await stream?.stopCapture()
        stream = nil
        delegate = nil

        fputs("Capture stopped\n", stderr)
        onStop?()
    }

    func setOnStop(_ handler: @escaping () -> Void) {
        onStop = handler
    }
}

enum CaptureError: Error, LocalizedError {
    case permissionDenied
    case noDisplays

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Screen Recording permission denied. Grant access in System Settings > Privacy & Security > Screen Recording."
        case .noDisplays:
            return "No displays available for capture."
        }
    }
}
