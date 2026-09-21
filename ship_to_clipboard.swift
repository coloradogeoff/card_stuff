import AppKit
import Foundation
import Vision

// MARK: - Screenshot

func captureScreen(to path: String) throws {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
    process.arguments = ["-x", path]
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw NSError(domain: "screencapture", code: Int(process.terminationStatus),
                      userInfo: [NSLocalizedDescriptionKey: "screencapture exited \(process.terminationStatus)"])
    }
}

// MARK: - OCR

struct TextObs {
    let text: String
    let box: CGRect  // Vision normalized: (0,0)=bottom-left, (1,1)=top-right
}

func recognizeText(in imageURL: URL, level: VNRequestTextRecognitionLevel = .accurate) throws -> [TextObs] {
    var result: [TextObs] = []
    var ocrError: Error?

    let request = VNRecognizeTextRequest { req, error in
        if let error { ocrError = error; return }
        guard let obs = req.results as? [VNRecognizedTextObservation] else { return }
        // Sort top-to-bottom; within same row, left-to-right
        let sorted = obs.sorted {
            if abs($0.boundingBox.minY - $1.boundingBox.minY) > 0.005 {
                return $0.boundingBox.minY > $1.boundingBox.minY
            }
            return $0.boundingBox.minX < $1.boundingBox.minX
        }
        result = sorted.compactMap { o in
            o.topCandidates(1).first.map { TextObs(text: $0.string, box: o.boundingBox) }
        }
    }
    request.recognitionLevel = level
    request.usesLanguageCorrection = false

    let handler = VNImageRequestHandler(url: imageURL, options: [:])
    try handler.perform([request])
    if let error = ocrError { throw error }
    return result
}

// MARK: - Address parsing

let cityStateZip = try! NSRegularExpression(
    pattern: #"^(.+?),?\s+([A-Z]{2})\s*(\d{5}(?:-\d{4})?)$"#
)

func isCityStateLine(_ text: String) -> Bool {
    cityStateZip.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
}

func formatLine(_ line: String) -> String {
    let range = NSRange(line.startIndex..., in: line)
    guard let match = cityStateZip.firstMatch(in: line, range: range) else {
        return fixLeadingDigit(line)
    }
    let ns = line as NSString
    return "\(ns.substring(with: match.range(at: 1))), \(ns.substring(with: match.range(at: 2))) \(ns.substring(with: match.range(at: 3)))"
}

// "i Hazel Ave" / "l Hazel Ave" → "1 Hazel Ave"
func fixLeadingDigit(_ line: String) -> String {
    guard let first = line.first, (first == "i" || first == "l"),
          line.count > 1, line[line.index(after: line.startIndex)].isWhitespace else { return line }
    return "1" + line.dropFirst()
}

func isShipToLabel(_ text: String) -> Bool {
    // Vision occasionally surrounds an otherwise correct label with an
    // invisible formatting character. Match its letters rather than requiring
    // an exact whitespace/punctuation representation of "Ship to".
    let letters = text.lowercased().unicodeScalars.filter {
        CharacterSet.letters.contains($0)
    }
    let normalized = String(String.UnicodeScalarView(letters))
    return normalized == "shipto"
}

// MARK: - Pass 1: locate address region in full-screen OCR

struct AddressRegion {
    let boxes: [CGRect]
}

func isShipFromLabel(_ text: String) -> Bool {
    let lower = text.lowercased()
    return lower.hasPrefix("ship from") || lower.hasPrefix("return to")
}

func isReturnAddressLine(_ text: String) -> Bool {
    let lower = text.lowercased()
    return lower.contains("nederland") && lower.contains("80466")
}

let countryNames: Set<String> = Set(Locale.Region.isoRegions.compactMap {
    Locale(identifier: "en_US").localizedString(forRegionCode: $0.identifier)?.lowercased()
})
let postalCode = try! NSRegularExpression(pattern: #"\b\d{5}(?:-\d{4})?\b"#)

func isCountryLine(_ text: String) -> Bool {
    countryNames.contains(text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
}

func containsPostalCode(_ text: String) -> Bool {
    postalCode.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
}

func findAddressRegion(in observations: [TextObs]) -> AddressRegion? {
    let shipToCandidates = observations.indices.filter { isShipToLabel(observations[$0].text) }
    guard !shipToCandidates.isEmpty else {
        return nil
    }
    // A full-screen capture can include a terminal or another app that happens
    // to contain the words “Ship to.” Prefer a label with a nearby, aligned
    // postal-code or country line—the structure of an actual recipient block.
    let shipToIdx = shipToCandidates.min { lhs, rhs in
        func boundaryDistance(for idx: Int) -> CGFloat {
            let label = observations[idx].box
            return observations.compactMap { observation in
                let verticalDistance = label.minY - observation.box.minY
                guard verticalDistance >= 0, verticalDistance <= 0.12,
                      observation.box.minX >= label.minX - 0.02,
                      observation.box.minX <= label.maxX + 0.20,
                      observation.text.count <= 60,
                      isCityStateLine(observation.text) || isCountryLine(observation.text) || containsPostalCode(observation.text) else {
                    return nil
                }
                return verticalDistance
            }.min() ?? .greatestFiniteMagnitude
        }
        return boundaryDistance(for: lhs) < boundaryDistance(for: rhs)
    }!
    let shipToBox = observations[shipToIdx].box
    let shipFromIdx = observations.dropFirst(shipToIdx + 1).firstIndex {
        $0.box.minY < shipToBox.minY && isShipFromLabel($0.text)
    }
    let returnAddressIdx = observations.dropFirst(shipToIdx + 1).firstIndex {
        $0.box.minY < shipToBox.minY && isReturnAddressLine($0.text)
    }
    let recipientEndIdx = shipFromIdx ?? observations.endIndex
    let recipientObservations = observations[(shipToIdx + 1)..<recipientEndIdx]
    let cityObs = recipientObservations.first {
        $0.box.minY < shipToBox.minY && $0.text.count <= 60 && isCityStateLine($0.text)
    }
    let countryObs = recipientObservations.first {
        $0.box.minY < shipToBox.maxY && $0.text.count <= 60 && isCountryLine($0.text)
    }

    // International addresses have no U.S. city/state/ZIP line. Prefer their
    // country line as the lower boundary. Some eBay layouts put Ship from above
    // Ship to, so it cannot be the only fallback boundary.
    guard let bottomBox = cityObs?.box ?? countryObs?.box ?? returnAddressIdx.map({ observations[$0].box }) ?? shipFromIdx.map({ observations[$0].box }) else {
        return nil
    }

    // Include every line in the recipient column when determining the crop
    // bounds. The country line can be much narrower than a street or an
    // address-specific identifier, so using only the label and bottom line can
    // clip valid address text.
    let lowerY = bottomBox.minY
    let recipientBoxes = observations[shipToIdx...].compactMap { observation -> CGRect? in
        guard observation.text.count <= 60,
              observation.box.minX >= shipToBox.minX - 0.02,
              observation.box.minX <= shipToBox.maxX + 0.05,
              observation.box.minY <= shipToBox.maxY + 0.02,
              observation.box.maxY >= lowerY - 0.01 else {
            return nil
        }
        return observation.box
    }
    return AddressRegion(boxes: recipientBoxes + [shipToBox, bottomBox])
}

// MARK: - Crop and scale

func cropAndScale(imageURL: URL, region: AddressRegion, scale: CGFloat = 4) throws -> URL {
    guard let src = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
          let cg  = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        throw NSError(domain: "crop", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not load image"])
    }

    let W = CGFloat(cg.width), H = CGFloat(cg.height)
    let pad: CGFloat = 0.015

    // Vision coords → CG coords (flip Y)
    let vLeft   = max(0, region.boxes.map(\.minX).min()! - pad)
    let vRight  = min(1, region.boxes.map(\.maxX).max()! + pad)
    let vTop    = min(1, region.boxes.map(\.maxY).max()! + pad)   // top on screen = high Vision Y
    let vBottom = max(0, region.boxes.map(\.minY).min()! - pad)

    let cgRect = CGRect(x: vLeft * W, y: (1 - vTop) * H,
                        width: (vRight - vLeft) * W, height: (vTop - vBottom) * H)

    guard let cropped = cg.cropping(to: cgRect) else {
        throw NSError(domain: "crop", code: 2, userInfo: [NSLocalizedDescriptionKey: "Crop failed"])
    }

    let sw = Int(cgRect.width * scale), sh = Int(cgRect.height * scale)
    guard let ctx = CGContext(data: nil, width: sw, height: sh,
                              bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        throw NSError(domain: "crop", code: 3, userInfo: [NSLocalizedDescriptionKey: "CGContext failed"])
    }
    ctx.interpolationQuality = .high
    ctx.draw(cropped, in: CGRect(x: 0, y: 0, width: sw, height: sh))

    guard let scaled = ctx.makeImage() else {
        throw NSError(domain: "crop", code: 4, userInfo: [NSLocalizedDescriptionKey: "makeImage failed"])
    }

    let outURL = URL(fileURLWithPath: "/tmp/ship_to_crop.png")
    guard let dest = CGImageDestinationCreateWithURL(outURL as CFURL, "public.png" as CFString, 1, nil) else {
        throw NSError(domain: "crop", code: 5, userInfo: [NSLocalizedDescriptionKey: "CGImageDestination failed"])
    }
    CGImageDestinationAddImage(dest, scaled, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw NSError(domain: "crop", code: 6, userInfo: [NSLocalizedDescriptionKey: "Finalize failed"])
    }
    return outURL
}

// MARK: - Pass 2: extract address from cropped OCR

func extractAddressFromCrop(_ observations: [TextObs]) -> String? {
    var lines: [String] = []
    for obs in observations {
        guard obs.text.count <= 60, !isShipToLabel(obs.text) else { continue }
        if isShipFromLabel(obs.text) || isReturnAddressLine(obs.text) { break }
        lines.append(obs.text)
        if isCityStateLine(obs.text) { break }
        if lines.count >= 6 { break }
    }
    guard !lines.isEmpty else { return nil }
    return lines.map(formatLine).joined(separator: "\n")
}

// MARK: - Main

let debug = CommandLine.arguments.contains("--debug")
let tempPath = "/tmp/ship_to_screenshot.png"
let suppliedImagePath: String? = {
    guard let imageFlag = CommandLine.arguments.firstIndex(of: "--image"),
          CommandLine.arguments.indices.contains(imageFlag + 1) else {
        return nil
    }
    return CommandLine.arguments[imageFlag + 1]
}()

do {
    let imageURL: URL
    if let suppliedImagePath {
        imageURL = URL(fileURLWithPath: suppliedImagePath)
    } else {
        try captureScreen(to: tempPath)
        imageURL = URL(fileURLWithPath: tempPath)
    }

    // Pass 1: fast OCR on full screen — just to find the address bounding box
    let pass1 = try recognizeText(in: imageURL, level: .fast)

    if debug {
        fputs("=== Pass 1 (full screen, fast) ===\n", stderr)
        for o in pass1 {
            fputs(String(format: "  x=%.2f-%.2f y=%.2f-%.2f  \"%@\"\n",
                         o.box.minX, o.box.maxX, o.box.minY, o.box.maxY, o.text), stderr)
        }
    }

    guard let region = findAddressRegion(in: pass1) else {
        fputs("error: 'Ship to' not found\n", stderr)
        exit(1)
    }

    // Pass 2: crop to address region, scale 4×, accurate OCR
    let cropURL = try cropAndScale(imageURL: imageURL, region: region)
    let pass2   = try recognizeText(in: cropURL, level: .accurate)

    if debug {
        fputs("=== Pass 2 (cropped 4×, accurate) ===\n", stderr)
        for o in pass2 {
            fputs(String(format: "  x=%.2f-%.2f y=%.2f-%.2f  \"%@\"\n",
                         o.box.minX, o.box.maxX, o.box.minY, o.box.maxY, o.text), stderr)
        }
    }

    guard let address = extractAddressFromCrop(pass2) else {
        fputs("error: Could not extract address from cropped region\n", stderr)
        exit(1)
    }

    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(address, forType: .string)
    print(address)

    if debug {
        fputs("Screenshot: \(tempPath)\nCrop: \(cropURL.path)\n", stderr)
    } else {
        if suppliedImagePath == nil {
            try? FileManager.default.removeItem(at: imageURL)
        }
        try? FileManager.default.removeItem(at: cropURL)
    }
} catch {
    fputs("error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
