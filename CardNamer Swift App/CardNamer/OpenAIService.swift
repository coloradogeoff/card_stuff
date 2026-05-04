import Foundation
import AppKit

enum OpenAIError: Error, LocalizedError {
    case noAPIKey
    case badResponse(Int)
    case noContent
    case jsonParseFailed(String)

    var errorDescription: String? {
        switch self {
        case .noAPIKey: return "No OpenAI API key found. Add it in Settings."
        case .badResponse(let code): return "OpenAI returned HTTP \(code)."
        case .noContent: return "OpenAI returned an empty response."
        case .jsonParseFailed(let raw): return "Could not parse response: \(raw.prefix(200))"
        }
    }
}

enum OpenAIService {

    static var model: String { SettingsStore.shared.selectedModel }

    // MARK: - Model discovery

    static func fetchChatModels() async -> [String] {
        guard let key = loadAPIKey(), !key.isEmpty else { return [] }
        var req = URLRequest(url: URL(string: "https://api.openai.com/v1/models")!)
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        guard let (data, _) = try? await URLSession.shared.data(for: req),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let list = json["data"] as? [[String: Any]] else { return [] }
        let datePattern = try? NSRegularExpression(pattern: #"-\d{4}-\d{2}-\d{2}"#)
        return list
            .compactMap { $0["id"] as? String }
            .filter { id in
                let l = id.lowercased()
                guard l.hasPrefix("gpt-") else { return false }
                guard !l.contains("instruct") && !l.contains("realtime") && !l.contains("audio") && !l.contains("search") else { return false }
                guard datePattern?.firstMatch(in: id, range: NSRange(id.startIndex..., in: id)) == nil else { return false }
                return true
            }
            .sorted(by: { $0 > $1 })
    }

    // MARK: - Key loading

    static func loadAPIKey() -> String? {
        if let key = UserDefaults.standard.string(forKey: "openai_api_key"), !key.isEmpty { return key }
        let candidates: [URL] = [
            URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("code/card_stuff/.openai-api-key.txt"),
            URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".openai-api-key.txt"),
        ]
        for url in candidates {
            if let text = try? String(contentsOf: url, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
                return text
            }
        }
        return ProcessInfo.processInfo.environment["OPENAI_API_KEY"]
    }

    // MARK: - eBay title generation

    static func generateTitle(
        frontURL: URL,
        backURL: URL,
        category: EbayCategory,
        setOverride: String?,
        varietyOverride: String?
    ) async throws -> String {
        guard let key = loadAPIKey() else { throw OpenAIError.noAPIKey }

        let ocrFront = await OCRService.recognize(imageURL: frontURL)
        let frontB64 = try imageToBase64(url: frontURL)
        let backB64  = try imageToBase64(url: backURL)

        var system = category.systemPrompt
        if !ocrFront.isEmpty { system += "\nText found on the front of the item: \(ocrFront)\n" }
        if let s = setOverride, !s.isEmpty {
            system += "Card set override — use exactly: \(s)\n"
        }
        if let v = varietyOverride, !v.isEmpty {
            system += "Variety override — use exactly: \(v)\n"
        }

        let body: [String: Any] = [
            "model": model,
            "max_completion_tokens": 500,
            "messages": [
                ["role": "system", "content": system],
                ["role": "user", "content": [
                    ["type": "text", "text": "This is the front of the item."],
                    ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64,\(frontB64)"]],
                ]],
                ["role": "user", "content": [
                    ["type": "text", "text": "This is the back of the item."],
                    ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64,\(backB64)"]],
                ]],
            ],
        ]

        var req = URLRequest(url: URL(string: "https://api.openai.com/v1/chat/completions")!)
        req.httpMethod = "POST"
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: req)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw OpenAIError.badResponse(code) }

        guard let outer = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = outer["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw OpenAIError.noContent
        }

        // Parse "Title: ..." from response
        let titlePattern = try? NSRegularExpression(pattern: #"^\s*(?:\d+\.\s*)?\**\s*title\s*:\s*(.+)$"#, options: [.caseInsensitive])
        for line in content.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let re = titlePattern,
               let m = re.firstMatch(in: trimmed, range: NSRange(trimmed.startIndex..., in: trimmed)),
               let r = Range(m.range(at: 1), in: trimmed) {
                return String(trimmed[r]).replacingOccurrences(of: "*", with: "").trimmingCharacters(in: .whitespaces)
            }
        }
        // Fallback: first non-empty line
        return content.components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "*", with: "") }
            .first(where: { !$0.isEmpty }) ?? content
    }

    // MARK: - Main call

    static func identifyCard(
        frontURL: URL,
        backURL: URL,
        ocrFront: String,
        ocrBack: String,
        ocrBackBottom: String
    ) async throws -> CardDetails {
        guard let key = loadAPIKey() else { throw OpenAIError.noAPIKey }

        let frontB64 = try imageToBase64(url: frontURL)
        let backB64 = try imageToBase64(url: backURL)

        var system = """
        You are a sports card identification assistant. Using the images and OCR text, \
        extract: year (4-digit start year of the season), last_name (player last name only), \
        manufacturer (e.g., Topps, Panini), series (e.g., Chrome, Select, Mosaic), and number \
        (card number only, no #). If the card is an insert, add the insert name before \
        the card number. Return ONLY a JSON object with keys: \
        year, last_name, manufacturer, series, number. If unknown, use 'Unknown'.
        """
        if !ocrFront.isEmpty { system += "\nOCR front text:\n\(ocrFront)\n" }
        if !ocrBack.isEmpty  { system += "\nOCR back text:\n\(ocrBack)\n" }
        if !ocrBackBottom.isEmpty {
            system += "\nOCR back bottom text (often includes the card year):\n\(ocrBackBottom)\n"
        }

        let body: [String: Any] = [
            "model": model,
            "max_completion_tokens": 400,
            "messages": [
                ["role": "system", "content": system],
                ["role": "user", "content": [
                    ["type": "text", "text": "Front of card."],
                    ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64,\(frontB64)"]],
                ]],
                ["role": "user", "content": [
                    ["type": "text", "text": "Back of card."],
                    ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64,\(backB64)"]],
                ]],
            ],
        ]

        var req = URLRequest(url: URL(string: "https://api.openai.com/v1/chat/completions")!)
        req.httpMethod = "POST"
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: req)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw OpenAIError.badResponse(code) }

        return try parseResponse(data: data)
    }

    // MARK: - Parsing

    private static func parseResponse(data: Data) throws -> CardDetails {
        guard let outer = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = outer["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw OpenAIError.noContent
        }

        let cleaned = stripCodeFences(content)
        guard let jsonData = cleaned.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw OpenAIError.jsonParseFailed(cleaned)
        }

        var d = CardDetails()
        d.year         = (dict["year"]         as? String) ?? "Unknown"
        d.lastName     = (dict["last_name"]    as? String) ?? "Unknown"
        d.manufacturer = (dict["manufacturer"] as? String) ?? "Unknown"
        d.series       = (dict["series"]       as? String) ?? "Unknown"
        d.number       = (dict["number"]       as? String) ?? "Unknown"
        return d
    }

    private static func stripCodeFences(_ value: String) -> String {
        var s = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```") {
            s = s.replacingOccurrences(of: #"^```[a-zA-Z]*\n"#, with: "", options: [.regularExpression])
            s = s.trimmingCharacters(in: .init(charactersIn: "`")).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return s
    }

    // MARK: - Image encoding

    private static func imageToBase64(url: URL, maxSize: Int = 1024, quality: CGFloat = 0.85) throws -> String {
        guard let nsImage = NSImage(contentsOf: url),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw CocoaError(.fileReadUnknown)
        }

        var w = cgImage.width, h = cgImage.height
        if w > maxSize {
            let scale = Double(maxSize) / Double(w)
            w = maxSize
            h = Int(Double(h) * scale)
        }

        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw CocoaError(.fileWriteUnknown)
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let resized = ctx.makeImage() else { throw CocoaError(.fileWriteUnknown) }

        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, "public.jpeg" as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(dest, resized, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }

        return (data as Data).base64EncodedString()
    }
}
