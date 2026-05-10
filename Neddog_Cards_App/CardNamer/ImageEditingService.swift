import Foundation
import AppKit
import ImageIO
import UniformTypeIdentifiers

enum ImageEditingService {
    static func rotateClockwise(fileURL: URL) throws {
        guard let source = CGImageSourceCreateWithURL(fileURL as CFURL, nil),
              let sourceImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw CocoaError(.fileReadCorruptFile)
        }

        let width = sourceImage.width
        let height = sourceImage.height
        guard let colorSpace = sourceImage.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB) else {
            throw CocoaError(.fileWriteUnknown)
        }

        let alphaInfo = sourceImage.alphaInfo
        let bitmapInfo: UInt32
        switch alphaInfo {
        case .none, .noneSkipFirst, .noneSkipLast:
            bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue
        case .premultipliedFirst, .first:
            bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
        case .premultipliedLast, .last:
            bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        case .alphaOnly:
            bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        @unknown default:
            bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        }

        guard let context = CGContext(
            data: nil,
            width: height,
            height: width,
            bitsPerComponent: sourceImage.bitsPerComponent,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            throw CocoaError(.fileWriteUnknown)
        }

        // Core Graphics uses a bottom-left origin here. For a 90 degree
        // clockwise rotation, shift the drawing origin to the right edge of the
        // destination canvas, then rotate and draw the original image at (0, 0).
        context.translateBy(x: CGFloat(height), y: 0)
        context.rotate(by: .pi / 2)
        context.draw(sourceImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let rotatedImage = context.makeImage() else {
            throw CocoaError(.fileWriteUnknown)
        }

        let destinationType = try destinationTypeIdentifier(for: fileURL)
        let temporaryURL = fileURL
            .deletingLastPathComponent()
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(fileURL.pathExtension)

        let destinationData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            destinationData,
            destinationType as CFString,
            1,
            nil
        ) else {
            throw CocoaError(.fileWriteUnknown)
        }

        let properties = destinationProperties(for: destinationType)
        CGImageDestinationAddImage(destination, rotatedImage, properties)
        guard CGImageDestinationFinalize(destination) else {
            throw CocoaError(.fileWriteUnknown)
        }

        try (destinationData as Data).write(to: temporaryURL, options: .atomic)
        _ = try FileManager.default.replaceItemAt(fileURL, withItemAt: temporaryURL)
    }

    private static func destinationTypeIdentifier(for fileURL: URL) throws -> String {
        let fileExtension = fileURL.pathExtension.lowercased()
        if let type = UTType(filenameExtension: fileExtension) {
            return type.identifier
        }

        switch fileExtension {
        case "jpg", "jpeg":
            return UTType.jpeg.identifier
        case "png":
            return UTType.png.identifier
        case "bmp":
            return UTType.bmp.identifier
        case "tif", "tiff":
            return UTType.tiff.identifier
        case "webp":
            return UTType.webP.identifier
        default:
            throw CocoaError(.fileWriteUnsupportedScheme)
        }
    }

    private static func destinationProperties(for typeIdentifier: String) -> CFDictionary? {
        switch typeIdentifier {
        case UTType.jpeg.identifier, UTType.webP.identifier:
            return [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary
        default:
            return nil
        }
    }
}
