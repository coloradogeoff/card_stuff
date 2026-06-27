import Foundation

enum ContentRotation: String {
    case none
    case clockwise
    case counterclockwise
}

struct EnvelopeSpec: Identifiable, Hashable {
    var id: String { label }
    let label: String
    let pageWidthIn: Double
    let pageHeightIn: Double
    let contentRotation: ContentRotation
    let media: String
}

struct PrintSettings {
    let printerName: String
    let outputFilename: String
    let options: [String]
}

// Shared label-file model used by both EbayLabelService and LetterTrackLabelService.
struct LabelFile: Identifiable, Hashable {
    let url: URL
    let modifiedAt: Date

    var id: URL { url }
    var name: String { url.lastPathComponent }
}

// Shared alert model used by views that show error alerts.
struct AlertItem: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

enum EnvelopeCatalog {
    static let printerName = "_192_168_86_174"
    static let media5x7    = "Custom.5x7in"
    static let media6x9    = "Custom.6x9in"
    static let mediaA7     = "Custom.5.25x7.25in"

    static let envelopeSlowOptions = [
        "MediaType=MidWeight96110",
        "HPPrintQuality=ProRes1200",
    ]

    // Named specs — use these to avoid stringly-typed lookups.
    static let spec5x7 = EnvelopeSpec(label: "5x7", pageWidthIn: 5.0,  pageHeightIn: 7.0,  contentRotation: .counterclockwise, media: media5x7)
    static let spec6x9 = EnvelopeSpec(label: "6x9", pageWidthIn: 6.0,  pageHeightIn: 9.0,  contentRotation: .counterclockwise, media: media6x9)
    static let specA7  = EnvelopeSpec(label: "A7",  pageWidthIn: 5.25, pageHeightIn: 7.25, contentRotation: .counterclockwise, media: mediaA7)

    // Envelope Print tab: address-printing sizes.
    static let specs: [EnvelopeSpec] = [spec5x7, spec6x9]

    // Letter Track tab: label-printing sizes.
    static let letterTrackSpecs: [EnvelopeSpec] = [specA7, spec6x9]

    static let printSettings = PrintSettings(
        printerName: printerName,
        outputFilename: "envelope_print.pdf",
        options: envelopeSlowOptions
    )

    // Shared archive location used by all label-printing services.
    static let archiveDirectory = URL(
        fileURLWithPath: "/Users/geoff/Sales/shipping/ebay_labels",
        isDirectory: true
    )

    static func archive(_ sourceURL: URL) throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: archiveDirectory, withIntermediateDirectories: true)

        var destination = archiveDirectory.appendingPathComponent(sourceURL.lastPathComponent)
        var counter = 2
        while fm.fileExists(atPath: destination.path) {
            let stem = sourceURL.deletingPathExtension().lastPathComponent
            destination = archiveDirectory.appendingPathComponent(
                "\(stem) (\(counter)).\(sourceURL.pathExtension)"
            )
            counter += 1
        }

        try fm.moveItem(at: sourceURL, to: destination)
        return destination
    }
}

enum DefaultAddresses {
    static let returnLines = [
        "Geoff Dutton",
        "NedDog's Stamps & Cards",
        "2644 Ridge Rd",
        "Nederland, CO 80466",
    ]

    static let toLines = [
        "Andrew Attard",
        "13 Lady Smith Drive",
        "ABN#64652016681 Code:PAID",
        "Edmondson Park NSW 2174",
        "Australia",
    ]
}
