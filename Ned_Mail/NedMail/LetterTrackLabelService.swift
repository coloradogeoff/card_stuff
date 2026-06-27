import CoreGraphics
import Foundation

enum LetterTrackLabelError: Error, LocalizedError {
    case downloadsUnavailable
    case invalidPDF(URL)
    case outputCreationFailed(URL)

    var errorDescription: String? {
        switch self {
        case .downloadsUnavailable:
            return "The Downloads folder could not be located."
        case .invalidPDF(let url):
            return "Could not read PDF: \(url.lastPathComponent)"
        case .outputCreationFailed(let url):
            return "Could not create the printable PDF:\n\(url.path)"
        }
    }
}

struct LetterTrackLabelService {
    // Matches LetterTrackPro export filenames like 1577097_9772_679104269999
    private static let labelPattern = try! NSRegularExpression(pattern: #"^\d+_\d+_\d+$"#)

    private let printer = EnvelopePrinter(settings: PrintSettings(
        printerName: EnvelopeCatalog.printerName,
        outputFilename: "lettertrack_label_print.pdf",
        options: EnvelopeCatalog.envelopeSlowOptions
    ))

    func findLabels() throws -> [LabelFile] {
        guard let downloads = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first else {
            throw LetterTrackLabelError.downloadsUnavailable
        }

        let keys: Set<URLResourceKey> = [.isRegularFileKey, .contentModificationDateKey]
        let urls = try FileManager.default.contentsOfDirectory(
            at: downloads,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]
        )

        return try urls.compactMap { url in
            guard url.pathExtension.lowercased() == "pdf" else { return nil }
            let stem = url.deletingPathExtension().lastPathComponent
            let range = NSRange(stem.startIndex..., in: stem)
            guard Self.labelPattern.firstMatch(in: stem, range: range) != nil else { return nil }
            let values = try url.resourceValues(forKeys: keys)
            guard values.isRegularFile == true else { return nil }
            return LabelFile(url: url, modifiedAt: values.contentModificationDate ?? .distantPast)
        }
        .sorted { $0.modifiedAt > $1.modifiedAt }
    }

    func renderEnvelopePDF(from sourceURL: URL, spec: EnvelopeSpec) throws -> URL {
        guard let document = CGPDFDocument(sourceURL as CFURL),
              document.numberOfPages == 1,
              let sourcePage = document.page(at: 1) else {
            throw LetterTrackLabelError.invalidPDF(sourceURL)
        }

        let sourceBox = sourcePage.getBoxRect(.mediaBox)
        let envW = spec.pageWidthIn  * 72
        let envH = spec.pageHeightIn * 72

        let outputURL = printer.outputURL
        var mediaBox = CGRect(x: 0, y: 0, width: envW, height: envH)
        guard let context = CGContext(outputURL as CFURL, mediaBox: &mediaBox, nil) else {
            throw LetterTrackLabelError.outputCreationFailed(outputURL)
        }

        let scale = min(envW / sourceBox.width, envH / sourceBox.height)
        let tx = (envW - sourceBox.width  * scale) / 2
        let ty = (envH - sourceBox.height * scale) / 2

        context.beginPDFPage(nil)
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(mediaBox)
        context.saveGState()
        context.translateBy(x: tx, y: ty)
        context.scaleBy(x: scale, y: scale)
        context.drawPDFPage(sourcePage)
        context.restoreGState()
        context.endPDFPage()
        context.closePDF()

        return outputURL
    }

    func printPDF(at url: URL, spec: EnvelopeSpec) throws {
        try printer.printPDF(at: url, spec: spec)
    }

    func archive(_ sourceURL: URL) throws -> URL {
        try EnvelopeCatalog.archive(sourceURL)
    }

    func removeOutput(at url: URL) {
        try? FileManager.default.removeItem(at: url)
    }
}
