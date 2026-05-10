import Foundation
import AppKit
import ImageIO

enum PSADownloadError: LocalizedError {
    case missingToken
    case badResponse(Int, String)
    case invalidJSON
    case noImageURLs
    case imageDownloadFailed(String)
    case imageWriteFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingToken:
            "No PSA API token found. Add it in Settings."
        case .badResponse(let status, let detail):
            detail.isEmpty ? "PSA API returned HTTP \(status)." : "PSA API returned HTTP \(status): \(detail)"
        case .invalidJSON:
            "PSA API response was not valid JSON."
        case .noImageURLs:
            "No front/back image URLs found for that PSA cert."
        case .imageDownloadFailed(let detail):
            "Could not download PSA image: \(detail)"
        case .imageWriteFailed(let detail):
            "Could not save PSA image: \(detail)"
        }
    }
}

enum PSADownloadService {
    private static let certAPIURL = "https://api.psacard.com/publicapi/cert/GetByCertNumber/%@"
    private static let imagesAPIURL = "https://api.psacard.com/publicapi/cert/GetImagesByCertNumber/%@"
    private static let curlStatusMarker = "__HTTP_STATUS__"
    private static let userAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    private static let frontTags = ["front", "obverse", "recto"]
    private static let backTags = ["back", "reverse", "verso"]
    private static let knownManufacturerPrefixes = ["Upper Deck", "Panini", "Topps", "Donruss", "Fleer", "Leaf", "Bowman", "Score"]
    private static let knownVarietySuffixes = ["Campus Legends"]

    static func download(certNumber: String, outputDirectory: URL, token: String) async throws -> String {
        let cert = certNumber.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedToken = normalizeToken(token)
        guard !normalizedToken.isEmpty else { throw PSADownloadError.missingToken }

        let certData = try await fetchJSON(
            urlString: String(format: certAPIURL, cert),
            token: normalizedToken
        )
        let imageData = try await fetchJSON(
            urlString: String(format: imagesAPIURL, cert),
            token: normalizedToken
        )

        let urls = extractImageURLs(from: imageData)
        let pickedURLs = try pickFrontBack(from: urls.all, preferredFront: urls.front, preferredBack: urls.back)
        let details = extractCardDetails(from: certData)
        let baseName = buildBaseName(from: details)

        let frontPath = outputDirectory.appendingPathComponent("\(baseName).jpg")
        let backPath = outputDirectory.appendingPathComponent("\(baseName)_b.jpg")

        async let frontBytes = downloadImage(from: pickedURLs.front, token: normalizedToken)
        async let backBytes = downloadImage(from: pickedURLs.back, token: normalizedToken)
        let (frontData, backData) = try await (frontBytes, backBytes)

        try saveProcessedJPEG(frontData, to: frontPath, scale: 0.75, quality: 0.65)
        try saveProcessedJPEG(backData, to: backPath, scale: 0.75, quality: 0.65)

        return "Saved \(frontPath.path)\nSaved \(backPath.path)"
    }

    private static func fetchJSON(urlString: String, token: String) async throws -> Any {
        if let json = try? fetchJSONViaCurl(urlString: urlString, token: token) {
            return json
        }

        guard let url = URL(string: urlString) else { throw PSADownloadError.invalidJSON }
        var request = URLRequest(url: url, timeoutInterval: 30)
        applyDefaultHeaders(to: &request, token: token)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw PSADownloadError.invalidJSON }
        guard (200..<300).contains(http.statusCode) else {
            let detail = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .prefix(240) ?? ""
            throw PSADownloadError.badResponse(http.statusCode, String(detail))
        }

        return try JSONSerialization.jsonObject(with: data)
    }

    private static func fetchJSONViaCurl(urlString: String, token: String) throws -> Any {
        let curlURL = URL(fileURLWithPath: "/usr/bin/curl")
        guard FileManager.default.isExecutableFile(atPath: curlURL.path) else {
            throw PSADownloadError.badResponse(0, "curl is not available.")
        }

        let process = Process()
        process.executableURL = curlURL
        process.arguments = [
            "-sS",
            "-L",
            "-w",
            "\n\(curlStatusMarker)%{http_code}",
            "-H",
            "Authorization: bearer \(token)",
            "-H",
            "Accept: application/json",
            urlString,
        ]

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        try process.run()
        process.waitUntilExit()

        let outputData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errorData = stderr.fileHandleForReading.readDataToEndOfFile()
        guard process.terminationStatus == 0 else {
            let error = String(data: errorData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw PSADownloadError.badResponse(Int(process.terminationStatus), error ?? "curl failed.")
        }

        guard let output = String(data: outputData, encoding: .utf8),
              let markerRange = output.range(of: curlStatusMarker, options: .backwards) else {
            throw PSADownloadError.invalidJSON
        }

        let payload = output[..<markerRange.lowerBound]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let statusText = output[markerRange.upperBound...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let status = Int(statusText) ?? 0

        guard (200..<300).contains(status) else {
            if isSecurityCheckHTML(payload) {
                throw PSADownloadError.badResponse(status, "PSA returned a Cloudflare security check page.")
            }
            throw PSADownloadError.badResponse(status, String(payload.prefix(240)))
        }

        guard !payload.isEmpty, !isSecurityCheckHTML(payload), let data = payload.data(using: .utf8) else {
            throw PSADownloadError.invalidJSON
        }
        return try JSONSerialization.jsonObject(with: data)
    }

    private static func downloadImage(from urlString: String, token: String) async throws -> Data {
        guard let url = URL(string: urlString) else {
            throw PSADownloadError.imageDownloadFailed("Invalid URL.")
        }
        var request = URLRequest(url: url, timeoutInterval: 60)
        applyDefaultHeaders(to: &request, token: token)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw PSADownloadError.imageDownloadFailed("HTTP \(status) from \(urlString).")
        }
        return data
    }

    private static func applyDefaultHeaders(to request: inout URLRequest, token: String) {
        request.setValue("application/json, text/plain, */*", forHTTPHeaderField: "Accept")
        request.setValue("en-US,en;q=0.9", forHTTPHeaderField: "Accept-Language")
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("https://www.psacard.com", forHTTPHeaderField: "Origin")
        request.setValue("https://www.psacard.com/", forHTTPHeaderField: "Referer")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    private static func normalizeToken(_ value: String) -> String {
        var raw = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if raw.hasPrefix("{"),
           let data = raw.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            for key in ["access_token", "token"] {
                if let token = json[key] as? String, !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    raw = token
                    break
                }
            }
        }

        raw = raw.replacingOccurrences(
            of: #"(?i)^authorization:\s*"#,
            with: "",
            options: .regularExpression
        )
        raw = raw.replacingOccurrences(
            of: #"(?i)^bearer\s+"#,
            with: "",
            options: .regularExpression
        )
        return raw.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func extractCardDetails(from json: Any) -> PSACardDetails {
        let allText = collectStrings(from: json).joined(separator: " ")
        let manufacturer = normalizeWhitespace(firstValue(in: json, keys: ["brand", "manufacturer", "brandname"]))
        let series = normalizeWhitespace(firstValue(in: json, keys: ["setname", "series", "cardset", "set"]))
        let player = normalizeWhitespace(firstValue(in: json, keys: ["player", "subject", "playername"]))
        let variety = normalizeWhitespace(firstValue(in: json, keys: ["variety", "parallel"]))
        let cardNumber = normalizeWhitespace(firstValue(in: json, keys: ["cardnumber", "cardno", "cardnum"]))

        let year = firstValue(in: json, keys: ["year", "cardyear", "certyear"])
            .flatMap(extractYear)
            ?? extractYear(allText)

        return PSACardDetails(
            year: year,
            player: player,
            manufacturer: manufacturer,
            series: series,
            variety: variety,
            cardNumber: cardNumber
        )
    }

    private static func buildBaseName(from details: PSACardDetails) -> String {
        let year = extractYear(details.year ?? "") ?? details.year ?? "Unknown"
        let lastName = smartTitle(extractLastName(details.player) ?? "Unknown")
        let split = splitManufacturerSeries(
            manufacturer: details.manufacturer,
            series: details.series
        )
        let manufacturer = smartTitle(split.manufacturer)
        var series = smartTitle(split.series)
        var variety = details.variety.flatMap(smartTitle)

        let seriesVariety = splitSeriesVariety(series: series, variety: variety)
        series = seriesVariety.series
        variety = seriesVariety.variety

        var parts = [slugify(year), slugify(lastName)]
        let manufacturerSlug = slugify(manufacturer)
        let seriesSlug = slugify(series)
        if manufacturerSlug != "Unknown" { parts.append(manufacturerSlug) }
        if seriesSlug != "Unknown" { parts.append(seriesSlug) }
        if let variety {
            let varietySlug = compactSlug(variety)
            if varietySlug != "Unknown" { parts.append(varietySlug) }
        }
        if let cardNumber = details.cardNumber {
            let cardNumberSlug = slugify(cardNumber)
            if cardNumberSlug != "Unknown" { parts.append(cardNumberSlug) }
        }
        return parts.joined(separator: "-")
    }

    private static func extractImageURLs(from json: Any) -> (all: [String], front: String?, back: String?) {
        var urls: [String] = []
        var frontURL: String?
        var backURL: String?

        func add(_ raw: String, keyHint: String?, typeHint: String?) {
            let normalized = normalizeURL(raw)
            urls.append(normalized)
            let hint = [keyHint, typeHint].compactMap { $0 }.joined(separator: " ").lowercased()
            if frontURL == nil, hint.range(of: #"front|obverse|recto"#, options: .regularExpression) != nil {
                frontURL = normalized
            }
            if backURL == nil, hint.range(of: #"back|reverse|verso"#, options: .regularExpression) != nil {
                backURL = normalized
            }
        }

        func walk(_ value: Any) {
            if let dict = value as? [String: Any] {
                var typeHint: String?
                var imageURLValue: String?
                var isFrontValue: Bool?

                for (key, child) in dict {
                    let keyLower = key.lowercased()
                    if let string = child as? String {
                        if ["imageurl", "url", "image", "imageuri"].contains(keyLower) {
                            imageURLValue = string
                        }
                        if ["imagetype", "view", "side", "position", "phototype"].contains(keyLower) {
                            typeHint = string
                        }
                    } else if ["isfrontimage", "isfront"].contains(keyLower), let bool = child as? Bool {
                        isFrontValue = bool
                    }
                }

                if let imageURLValue {
                    if isFrontValue == true {
                        add(imageURLValue, keyHint: "front", typeHint: typeHint)
                    } else if isFrontValue == false {
                        add(imageURLValue, keyHint: "back", typeHint: typeHint)
                    } else {
                        add(imageURLValue, keyHint: nil, typeHint: typeHint)
                    }
                }

                for (key, child) in dict {
                    if let string = child as? String, looksLikeImageURL(string, keyHint: key) {
                        add(string, keyHint: key, typeHint: typeHint)
                    }
                }
                dict.values.forEach(walk)
            } else if let array = value as? [Any] {
                array.forEach(walk)
            } else if let string = value as? String, looksLikeImageURL(string, keyHint: nil) {
                add(string, keyHint: nil, typeHint: nil)
            }
        }

        walk(json)
        var seen = Set<String>()
        let unique = urls.filter { seen.insert($0).inserted }
        return (unique, frontURL, backURL)
    }

    private static func pickFrontBack(from imageURLs: [String], preferredFront: String?, preferredBack: String?) throws -> (front: String, back: String) {
        if let preferredFront, let preferredBack {
            return (preferredFront, preferredBack)
        }

        var front = preferredFront
        var back = preferredBack

        if front == nil {
            front = imageURLs.first { hasFaceTag($0, tags: frontTags) }
        }
        if back == nil {
            back = imageURLs.first { hasFaceTag($0, tags: backTags) }
        }
        if let frontValue = front, back == nil {
            let guess = frontValue.replacingOccurrences(of: #"(?i)\bfront\b"#, with: "back", options: .regularExpression)
            if guess != frontValue { back = guess }
        }
        if let backValue = back, front == nil {
            let guess = backValue.replacingOccurrences(of: #"(?i)\bback\b"#, with: "front", options: .regularExpression)
            if guess != backValue { front = guess }
        }
        if let front, let back {
            return (front, back)
        }
        if imageURLs.count >= 2 {
            return (imageURLs[0], imageURLs[1])
        }
        throw PSADownloadError.noImageURLs
    }

    private static func saveProcessedJPEG(_ data: Data, to url: URL, scale: CGFloat, quality: CGFloat) throws {
        guard let image = NSImage(data: data),
              let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw PSADownloadError.imageWriteFailed("Downloaded data was not an image.")
        }

        let width = max(1, Int(CGFloat(cgImage.width) * scale))
        let height = max(1, Int(CGFloat(cgImage.height) * scale))

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw PSADownloadError.imageWriteFailed("Could not create image context.")
        }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let resized = context.makeImage() else {
            throw PSADownloadError.imageWriteFailed("Could not resize image.")
        }

        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, "public.jpeg" as CFString, 1, nil) else {
            throw PSADownloadError.imageWriteFailed("Could not create JPEG destination.")
        }
        CGImageDestinationAddImage(destination, resized, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw PSADownloadError.imageWriteFailed("Could not encode JPEG.")
        }
        try (output as Data).write(to: url, options: .atomic)
    }

    private static func firstValue(in json: Any, keys: Set<String>) -> String? {
        if let dict = json as? [String: Any] {
            for (key, value) in dict {
                if keys.contains(key.lowercased()), let string = scalarString(value) {
                    return string
                }
                if let nested = firstValue(in: value, keys: keys) {
                    return nested
                }
            }
        } else if let array = json as? [Any] {
            for item in array {
                if let nested = firstValue(in: item, keys: keys) {
                    return nested
                }
            }
        }
        return nil
    }

    private static func collectStrings(from json: Any) -> [String] {
        if let dict = json as? [String: Any] {
            return dict.values.flatMap(collectStrings)
        }
        if let array = json as? [Any] {
            return array.flatMap(collectStrings)
        }
        if let string = json as? String {
            return [string]
        }
        return []
    }

    private static func scalarString(_ value: Any) -> String? {
        if let string = value as? String { return string }
        if let number = value as? NSNumber { return number.stringValue }
        return nil
    }

    private static func normalizeWhitespace(_ value: String?) -> String? {
        guard let value, !value.isEmpty else { return nil }
        let normalized = value.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return normalized.isEmpty ? nil : normalized
    }

    private static func smartTitle(_ value: String) -> String {
        value.split(separator: " ", omittingEmptySubsequences: false)
            .map { word in
                let value = String(word)
                if value.uppercased() == value, value.count > 3 {
                    return value.capitalized
                }
                return value
            }
            .joined(separator: " ")
    }

    private static func slugify(_ value: String) -> String {
        let ascii = value.data(using: .ascii, allowLossyConversion: true)
            .flatMap { String(data: $0, encoding: .ascii) } ?? ""
        let cleaned = ascii.replacingOccurrences(of: #"[^\w\s-]"#, with: "", options: .regularExpression)
        let slug = cleaned.replacingOccurrences(of: #"[\s_-]+"#, with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return slug.isEmpty ? "Unknown" : slug
    }

    private static func compactSlug(_ value: String) -> String {
        let compact = slugify(value).replacingOccurrences(of: "-", with: "")
        return compact.isEmpty ? "Unknown" : compact
    }

    private static func extractYear(_ value: String) -> String? {
        value.firstMatch(pattern: #"(19|20)\d{2}"#)
    }

    private static func extractLastName(_ player: String?) -> String? {
        guard let player else { return nil }
        var tokens = player.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard !tokens.isEmpty else { return nil }
        let suffixes = ["jr", "sr", "ii", "iii", "iv", "v"]
        if let last = tokens.last?.trimmingCharacters(in: CharacterSet(charactersIn: ".")).lowercased(),
           suffixes.contains(last),
           tokens.count > 1 {
            tokens.removeLast()
        }
        return tokens.last
    }

    private static func splitManufacturerSeries(manufacturer: String?, series: String?) -> (manufacturer: String, series: String) {
        var manufacturer = manufacturer ?? ""
        var series = series ?? ""

        series = series.replacingOccurrences(of: #"^(?:19|20)\d{2}\s+"#, with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if !manufacturer.isEmpty, series.isEmpty, manufacturer.contains(" ") {
            for prefix in knownManufacturerPrefixes where manufacturer.lowercased().hasPrefix(prefix.lowercased() + " ") {
                let remainder = String(manufacturer.dropFirst(prefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
                if !remainder.isEmpty {
                    manufacturer = prefix
                    series = remainder
                    break
                }
            }
        }

        if manufacturer.isEmpty, !series.isEmpty {
            let parts = series.split(separator: " ").map(String.init)
            if let first = parts.first {
                manufacturer = first
                series = parts.dropFirst().joined(separator: " ")
                if series.isEmpty { series = first }
            }
        }

        if !manufacturer.isEmpty, !series.isEmpty, series.lowercased().hasPrefix(manufacturer.lowercased()) {
            series = String(series.dropFirst(manufacturer.count))
                .trimmingCharacters(in: CharacterSet(charactersIn: " -"))
        }

        return (manufacturer.isEmpty ? "Unknown" : manufacturer, series.isEmpty ? "Unknown" : series)
    }

    private static func splitSeriesVariety(series: String, variety: String?) -> (series: String, variety: String?) {
        guard variety == nil else { return (series, variety) }
        for suffix in knownVarietySuffixes where series.lowercased().hasSuffix(" " + suffix.lowercased()) {
            let trimmed = String(series.dropLast(suffix.count)).trimmingCharacters(in: CharacterSet(charactersIn: " -"))
            if !trimmed.isEmpty { return (trimmed, suffix) }
        }
        return (series, variety)
    }

    private static func normalizeURL(_ value: String) -> String {
        if value.hasPrefix("//") { return "https:\(value)" }
        if let url = URL(string: value), url.scheme != nil { return value }
        return URL(string: value, relativeTo: URL(string: "https://www.psacard.com/"))?.absoluteString ?? value
    }

    private static func looksLikeImageURL(_ value: String, keyHint: String?) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.range(of: #"\.(jpg|jpeg|png|webp|gif|tif|tiff)(\?|#|$)"#, options: [.regularExpression, .caseInsensitive]) != nil {
            return true
        }
        guard trimmed.hasPrefix("http://") || trimmed.hasPrefix("https://") || trimmed.hasPrefix("//") || trimmed.hasPrefix("/") else {
            return false
        }
        let lower = trimmed.lowercased()
        if lower.contains("/image") || lower.contains("/img") { return true }
        return keyHint?.lowercased().contains("image") == true
    }

    private static func isSecurityCheckHTML(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("<title>security check</title>")
            || lower.contains("<title>just a moment...</title>")
            || lower.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("<!doctype html")
    }

    private static func hasFaceTag(_ url: String, tags: [String]) -> Bool {
        tags.contains { tag in
            url.range(
                of: #"(?<![a-z0-9])\#(tag)(?![a-z0-9])"#,
                options: [.regularExpression, .caseInsensitive]
            ) != nil
        }
    }
}

private struct PSACardDetails {
    var year: String?
    var player: String?
    var manufacturer: String?
    var series: String?
    var variety: String?
    var cardNumber: String?
}

private extension String {
    func firstMatch(pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(startIndex..., in: self)
        guard let match = regex.firstMatch(in: self, range: range),
              let resultRange = Range(match.range, in: self) else {
            return nil
        }
        return String(self[resultRange])
    }
}
