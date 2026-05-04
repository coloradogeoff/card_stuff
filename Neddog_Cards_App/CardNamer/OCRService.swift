import Vision
import AppKit

enum OCRService {

    static func recognize(imageURL: URL, cropBottom: Bool = false) async -> String {
        guard let nsImage = NSImage(contentsOf: imageURL),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return ""
        }

        let targetImage: CGImage
        if cropBottom {
            let h = cgImage.height
            let cropRect = CGRect(x: 0, y: 0, width: cgImage.width, height: h / 4)
            targetImage = cgImage.cropping(to: cropRect) ?? cgImage
        } else {
            targetImage = cgImage
        }

        return await withCheckedContinuation { continuation in
            let request = VNRecognizeTextRequest { req, _ in
                let text = (req.results as? [VNRecognizedTextObservation] ?? [])
                    .compactMap { $0.topCandidates(1).first?.string }
                    .joined(separator: "\n")
                continuation.resume(returning: text)
            }
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true

            let handler = VNImageRequestHandler(cgImage: targetImage, options: [:])
            try? handler.perform([request])
        }
    }
}
