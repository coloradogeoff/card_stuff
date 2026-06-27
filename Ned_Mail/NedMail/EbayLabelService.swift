import CoreGraphics
import Foundation

enum EbayLabelError: Error, LocalizedError {
    case downloadsUnavailable
    case invalidPDF(URL)
    case unexpectedPageSize(CGSize)
    case outputCreationFailed(URL)
    case printFailed(exitCode: Int32, stderr: String, stdout: String)

    var errorDescription: String? {
        switch self {
        case .downloadsUnavailable:
            return "The Downloads folder could not be located."
        case .invalidPDF(let url):
            return "Could not read the eBay label PDF:\n\(url.path)"
        case .unexpectedPageSize(let size):
            return String(
                format: "The selected PDF is %.2f × %.2f inches. Expected an eBay letter-size label PDF.",
                size.width / 72,
                size.height / 72
            )
        case .outputCreationFailed(let url):
            return "Could not create the printable PDF:\n\(url.path)"
        case .printFailed(let exitCode, let stderr, let stdout):
            let detail = stderr.isEmpty ? stdout : stderr
            let message = detail.trimmingCharacters(in: .whitespacesAndNewlines)
            return "lp exited with code \(exitCode):\n\(message.isEmpty ? "Unknown error" : message)"
        }
    }
}

struct EbayLabelService {
    private static let pointsPerInch: CGFloat = 72
    private static let expectedPageSize = CGSize(width: 612, height: 792)
    private static let labelBox = CGRect(x: 90, y: 446.46, width: 432, height: 288)
    private static let envelopeSize = CGSize(width: 432, height: 648)

    let settings = PrintSettings(
        printerName: EnvelopeCatalog.printerName,
        outputFilename: "ebay_label_print.pdf",
        options: EnvelopeCatalog.envelopeSlowOptions
    )

    func findLabels() throws -> [LabelFile] {
        guard let downloads = FileManager.default.urls(
            for: .downloadsDirectory,
            in: .userDomainMask
        ).first else {
            throw EbayLabelError.downloadsUnavailable
        }

        let keys: Set<URLResourceKey> = [.isRegularFileKey, .contentModificationDateKey]
        let urls = try FileManager.default.contentsOfDirectory(
            at: downloads,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]
        )

        return try urls.compactMap { url in
            guard url.pathExtension.lowercased() == "pdf",
                  normalizedName(url.deletingPathExtension().lastPathComponent)
                    .hasPrefix("ebaylabel") else {
                return nil
            }
            let values = try url.resourceValues(forKeys: keys)
            guard values.isRegularFile == true else { return nil }
            return LabelFile(url: url, modifiedAt: values.contentModificationDate ?? .distantPast)
        }
        .sorted { $0.modifiedAt > $1.modifiedAt }
    }

    func renderEnvelopePDF(from sourceURL: URL) throws -> URL {
        guard let document = CGPDFDocument(sourceURL as CFURL),
              document.numberOfPages == 1,
              let sourcePage = document.page(at: 1) else {
            throw EbayLabelError.invalidPDF(sourceURL)
        }

        let sourceBox = sourcePage.getBoxRect(.mediaBox)
        guard abs(sourceBox.width - Self.expectedPageSize.width) < 1,
              abs(sourceBox.height - Self.expectedPageSize.height) < 1 else {
            throw EbayLabelError.unexpectedPageSize(sourceBox.size)
        }

        let outputURL = try outputURL()
        var mediaBox = CGRect(origin: .zero, size: Self.envelopeSize)
        guard let context = CGContext(outputURL as CFURL, mediaBox: &mediaBox, nil) else {
            throw EbayLabelError.outputCreationFailed(outputURL)
        }

        context.beginPDFPage(nil)
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(mediaBox)

        let rotatedSize = CGSize(
            width: Self.labelBox.height,
            height: Self.labelBox.width
        )
        let offsetX = (mediaBox.width - rotatedSize.width) / 2
        let offsetY = (mediaBox.height - rotatedSize.height) / 2
        let targetBox = CGRect(
            x: offsetX,
            y: offsetY,
            width: rotatedSize.width,
            height: rotatedSize.height
        )

        context.saveGState()
        context.clip(to: targetBox)
        context.concatenate(
            CGAffineTransform(
                a: 0,
                b: -1,
                c: 1,
                d: 0,
                tx: offsetX - Self.labelBox.minY,
                ty: Self.labelBox.maxX + offsetY
            )
        )
        context.drawPDFPage(sourcePage)
        context.restoreGState()

        context.endPDFPage()
        context.closePDF()
        return outputURL
    }

    func printPDF(at url: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/lp")

        var arguments = [
            "-d", settings.printerName,
            "-o", "media=\(EnvelopeCatalog.media6x9)",
        ]
        for option in settings.options {
            arguments.append(contentsOf: ["-o", option])
        }
        arguments.append(url.path)
        process.arguments = arguments

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        try process.run()
        process.waitUntilExit()

        if process.terminationStatus != 0 {
            let stdout = String(
                data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8
            ) ?? ""
            let stderr = String(
                data: stderrPipe.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8
            ) ?? ""
            throw EbayLabelError.printFailed(
                exitCode: process.terminationStatus,
                stderr: stderr,
                stdout: stdout
            )
        }
    }

    func archive(_ sourceURL: URL) throws -> URL {
        try EnvelopeCatalog.archive(sourceURL)
    }

    func removeOutput(at url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private func normalizedName(_ name: String) -> String {
        String(name.lowercased().filter { $0.isLetter || $0.isNumber })
    }

    private func outputURL() throws -> URL {
        let applicationSupport = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = applicationSupport.appendingPathComponent(
            "NedMail",
            isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory.appendingPathComponent(settings.outputFilename)
    }
}
