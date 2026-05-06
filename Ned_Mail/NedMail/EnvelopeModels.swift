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

enum EnvelopeCatalog {
    static let printerName = "_192_168_86_174"
    static let media6x9 = "Custom.6x9in"
    static let media5x7 = "Custom.5x7in"

    static let envelopeSlowOptions = [
        "MediaType=MidWeight96110",
        "HPPrintQuality=ProRes1200",
    ]

    static let specs: [EnvelopeSpec] = [
        EnvelopeSpec(
            label: "5x7",
            pageWidthIn: 5.0,
            pageHeightIn: 7.0,
            contentRotation: .counterclockwise,
            media: media5x7
        ),
        EnvelopeSpec(
            label: "6x9",
            pageWidthIn: 6.0,
            pageHeightIn: 9.0,
            contentRotation: .counterclockwise,
            media: media6x9
        ),
    ]

    static let printSettings = PrintSettings(
        printerName: printerName,
        outputFilename: "envelope_print.pdf",
        options: envelopeSlowOptions
    )
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
