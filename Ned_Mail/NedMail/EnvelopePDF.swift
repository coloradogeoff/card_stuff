import CoreGraphics
import CoreText
import Foundation

enum EnvelopePDFError: Error {
    case contextCreationFailed(URL)
    case attributedStringCreationFailed
}

struct EnvelopePDFOptions {
    var insetIn: CGFloat = 0.5
    var paddingIn: CGFloat = 0.15
    var returnFontName: String = "Helvetica"
    var returnFontSize: CGFloat = 11
    var returnLeading: CGFloat = 13
    var toFontName: String = "Helvetica"
    var toFontSize: CGFloat = 16
    var toLeading: CGFloat = 20
    var baseCenterYFrac: CGFloat = 0.43
    var toShiftDownIn: CGFloat = 1.0
    var toShiftRightIn: CGFloat = 0.5
}

enum EnvelopePDF {
    static let pointsPerInch: CGFloat = 72.0

    static func write(
        to url: URL,
        spec: EnvelopeSpec,
        returnLines: [String],
        toLines: [String],
        options: EnvelopePDFOptions = EnvelopePDFOptions()
    ) throws {
        let pageW = CGFloat(spec.pageWidthIn) * pointsPerInch
        let pageH = CGFloat(spec.pageHeightIn) * pointsPerInch
        var mediaBox = CGRect(x: 0, y: 0, width: pageW, height: pageH)

        guard let context = CGContext(url as CFURL, mediaBox: &mediaBox, nil) else {
            throw EnvelopePDFError.contextCreationFailed(url)
        }

        context.beginPDFPage(nil as CFDictionary?)

        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: pageW, height: pageH))

        let W: CGFloat
        let H: CGFloat
        let rotated: Bool

        switch spec.contentRotation {
        case .counterclockwise:
            context.saveGState()
            context.translateBy(x: pageW, y: 0)
            context.rotate(by: .pi / 2)
            W = pageH
            H = pageW
            rotated = true
        case .clockwise:
            context.saveGState()
            context.translateBy(x: 0, y: pageH)
            context.rotate(by: -.pi / 2)
            W = pageH
            H = pageW
            rotated = true
        case .none:
            W = pageW
            H = pageH
            rotated = false
        }

        let inset = options.insetIn * pointsPerInch
        let padding = options.paddingIn * pointsPerInch
        let textColor = CGColor(red: 0, green: 0, blue: 0, alpha: 1)

        let returnFont = CTFontCreateWithName(options.returnFontName as CFString, options.returnFontSize, nil)
        let xRet = inset + padding
        let yRetTop = H - inset - padding
        for (i, line) in returnLines.enumerated() {
            try drawText(
                in: context,
                text: line,
                font: returnFont,
                color: textColor,
                x: xRet,
                y: yRetTop - CGFloat(i) * options.returnLeading
            )
        }

        let toFont = CTFontCreateWithName(options.toFontName as CFString, options.toFontSize, nil)
        let lineWidths = toLines.map { stringWidth($0, font: toFont) }
        let blockWidth = lineWidths.max() ?? 0
        let xLeft = (W - blockWidth) / 2 + options.toShiftRightIn * pointsPerInch

        let count = max(toLines.count, 1)
        let blockHeight = options.toLeading * CGFloat(count - 1) + options.toFontSize
        let centerY = H * options.baseCenterYFrac - options.toShiftDownIn * pointsPerInch
        let yBlockTop = centerY + blockHeight / 2

        for (i, line) in toLines.enumerated() {
            try drawText(
                in: context,
                text: line,
                font: toFont,
                color: textColor,
                x: xLeft,
                y: yBlockTop - CGFloat(i) * options.toLeading
            )
        }

        if rotated {
            context.restoreGState()
        }

        context.endPDFPage()
        context.closePDF()
    }

    private static func makeLine(_ text: String, font: CTFont, color: CGColor) throws -> CTLine {
        let attrs: [CFString: Any] = [
            kCTFontAttributeName: font,
            kCTForegroundColorAttributeName: color,
        ]
        guard let attributed = CFAttributedStringCreate(
            kCFAllocatorDefault,
            text as CFString,
            attrs as CFDictionary
        ) else {
            throw EnvelopePDFError.attributedStringCreationFailed
        }
        return CTLineCreateWithAttributedString(attributed)
    }

    private static func drawText(
        in context: CGContext,
        text: String,
        font: CTFont,
        color: CGColor,
        x: CGFloat,
        y: CGFloat
    ) throws {
        let line = try makeLine(text, font: font, color: color)
        context.textPosition = CGPoint(x: x, y: y)
        CTLineDraw(line, context)
    }

    private static func stringWidth(_ string: String, font: CTFont) -> CGFloat {
        let attrs: [CFString: Any] = [kCTFontAttributeName: font]
        guard let attributed = CFAttributedStringCreate(
            kCFAllocatorDefault,
            string as CFString,
            attrs as CFDictionary
        ) else {
            return 0
        }
        let line = CTLineCreateWithAttributedString(attributed)
        return CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
    }
}
