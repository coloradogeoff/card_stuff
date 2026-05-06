import Foundation

enum EnvelopePrinterError: Error, LocalizedError {
    case lpFailed(exitCode: Int32, stderr: String, stdout: String)

    var errorDescription: String? {
        switch self {
        case .lpFailed(let exitCode, let stderr, let stdout):
            let detail = stderr.isEmpty ? stdout : stderr
            let trimmed = detail.trimmingCharacters(in: .whitespacesAndNewlines)
            let message = trimmed.isEmpty ? "Unknown error" : trimmed
            return "lp exited with code \(exitCode):\n\(message)"
        }
    }
}

struct EnvelopePrinter {
    let settings: PrintSettings

    var outputURL: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let appDir = dir.appendingPathComponent("NedMail", isDirectory: true)
        try? FileManager.default.createDirectory(at: appDir, withIntermediateDirectories: true)
        return appDir.appendingPathComponent(settings.outputFilename)
    }

    func renderPDF(spec: EnvelopeSpec, returnLines: [String], toLines: [String]) throws -> URL {
        let url = outputURL
        try EnvelopePDF.write(to: url, spec: spec, returnLines: returnLines, toLines: toLines)
        return url
    }

    func printPDF(at url: URL, spec: EnvelopeSpec) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/lp")

        var args = ["-d", settings.printerName, "-o", "media=\(spec.media)"]
        for option in settings.options {
            args.append("-o")
            args.append(option)
        }
        args.append(url.path)
        process.arguments = args

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()
        process.waitUntilExit()

        if process.terminationStatus != 0 {
            let stdout = String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            let stderr = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            throw EnvelopePrinterError.lpFailed(
                exitCode: process.terminationStatus,
                stderr: stderr,
                stdout: stdout
            )
        }
    }

    func removePDF(at url: URL) throws {
        do {
            try FileManager.default.removeItem(at: url)
        } catch CocoaError.fileNoSuchFile {
            // Already gone — fine.
        }
    }
}
