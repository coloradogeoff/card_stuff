import Foundation
import AppKit

enum OpenAIError: Error, LocalizedError {
    case noAPIKey
    case badResponse(Int, String)
    case noContent(String)
    case jsonParseFailed(String)

    var errorDescription: String? {
        switch self {
        case .noAPIKey: return "No OpenAI API key found. Add it in Settings."
        case .badResponse(let code, let detail):
            return detail.isEmpty ? "OpenAI returned HTTP \(code)." : "OpenAI returned HTTP \(code): \(detail)"
        case .noContent(let detail):
            return detail.isEmpty ? "OpenAI returned an empty response." : "OpenAI returned an empty response (\(detail))."
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
        varietyOverride: String?,
        supplementalRules: String?
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
        if let rules = supplementalRules, !rules.isEmpty {
            system += """

            Supplemental rules supplied by the user:
            \(rules)
            """
        }

        func buildRequest(budget: Int) throws -> URLRequest {
            var body: [String: Any] = [
                "model": model,
                // A title is ~30 tokens, but on GPT-5 models the reasoning tokens
                // come out of this same budget, so leave headroom.
                "max_completion_tokens": budget,
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
            // Writing one title needs no deliberation. Without this a reasoning
            // model can spend the entire budget thinking and return empty content
            // with finish_reason "length".
            if model.lowercased().hasPrefix("gpt-5") {
                body["reasoning_effort"] = "none"
            }

            var req = URLRequest(url: URL(string: "https://api.openai.com/v1/chat/completions")!)
            req.httpMethod = "POST"
            req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
            return req
        }

        // A 200 with empty content happens occasionally (content filter, truncation,
        // or a plain flake). Retry once rather than surfacing a blank title.
        var budget = 2_000
        var lastError: Error = OpenAIError.noContent("")
        for attempt in 0..<2 {
            if attempt > 0 { try? await Task.sleep(nanoseconds: 1_000_000_000) }
            do {
                let title = try await requestTitle(buildRequest(budget: budget))
                return await enforcingLengthLimit(title, key: key)
            } catch {
                lastError = error
                guard case OpenAIError.noContent(let detail) = error else { throw error }
                // Truncation, not a flake — retrying the same budget fails the same way.
                if detail.contains("length") { budget *= 4 }
            }
        }
        throw lastError
    }

    // MARK: - Title length

    /// Language models are unreliable at counting characters, so the eBay limit
    /// is enforced here instead of being left to the prompt. A rewrite is asked
    /// for first — the model judges what detail to sacrifice far better than a
    /// blind truncation — with a deterministic trim as the guarantee.
    private static func enforcingLengthLimit(_ title: String, key: String) async -> String {
        let limit = EbayTitleLimit.maxCharacters
        guard title.count > limit else { return title }

        var best = title
        if let shortened = try? await requestShortenedTitle(title, key: key),
           !shortened.isEmpty,
           shortened.count < best.count {
            best = shortened
        }
        return best.count <= limit ? best : trimmedToLimit(best, limit: limit)
    }

    /// Text-only follow-up: shortening needs the title, not the card images.
    private static func requestShortenedTitle(_ title: String, key: String) async throws -> String {
        let limit = EbayTitleLimit.maxCharacters
        let instruction = """
        The eBay title below is \(title.count) characters. The limit is \(limit).
        Rewrite it to \(limit) characters or fewer, counting spaces.

        Keep, in order of importance: the player or subject name, the season years,
        the manufacturer and set, and the card number.
        Drop, in this order, only as much as needed to fit: the team nickname at the
        end, then the word "Insert", then remaining variety/parallel wording.
        Never abbreviate or initialize the player's name.

        Reply with the title only — no label, quotes, or explanation.

        Title: \(title)
        """

        var body: [String: Any] = [
            "model": model,
            "max_completion_tokens": 2_000,
            "messages": [["role": "user", "content": instruction]],
        ]
        if model.lowercased().hasPrefix("gpt-5") {
            body["reasoning_effort"] = "none"
        }

        var req = URLRequest(url: URL(string: "https://api.openai.com/v1/chat/completions")!)
        req.httpMethod = "POST"
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        return try await requestTitle(req)
    }

    /// Last-resort guarantee that a title fits. Drops whole " - " sections from
    /// the end first (team, then variety), since the format puts the least
    /// important detail last, and only cuts words if that is still not enough.
    static func trimmedToLimit(_ title: String, limit: Int) -> String {
        guard title.count > limit else { return title }

        var sections = title.components(separatedBy: " - ")
        // Keep at least the name and the set/number that identify the card.
        while sections.count > 2, sections.joined(separator: " - ").count > limit {
            sections.removeLast()
        }

        var result = sections.joined(separator: " - ")
        while result.count > limit, let lastSpace = result.lastIndex(of: " ") {
            result = String(result[..<lastSpace])
        }

        result = String(result.prefix(limit))
        while let last = result.last, last == " " || last == "-" {
            result.removeLast()
        }
        return result.trimmingCharacters(in: .whitespaces)
    }

    private static func requestTitle(_ req: URLRequest) async throws -> String {
        let (data, response) = try await URLSession.shared.data(for: req)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw OpenAIError.badResponse(code, apiErrorMessage(from: data)) }

        guard let outer = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = outer["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String,
              !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OpenAIError.noContent(responseDetail(from: data))
        }

        // Parse "Title: ..." from response
        let titlePattern = try? NSRegularExpression(pattern: #"^\s*(?:\d+\.\s*)?\**\s*title\s*:\s*(.+)$"#, options: [.caseInsensitive])
        for line in content.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let re = titlePattern,
               let m = re.firstMatch(in: trimmed, range: NSRange(trimmed.startIndex..., in: trimmed)),
               let r = Range(m.range(at: 1), in: trimmed) {
                let title = String(trimmed[r]).replacingOccurrences(of: "*", with: "").trimmingCharacters(in: .whitespaces)
                if !title.isEmpty { return title }
            }
        }
        // Fallback: first non-empty line
        guard let fallback = content.components(separatedBy: "\n")
            .map({ $0.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "*", with: "") })
            .first(where: { !$0.isEmpty }) else {
            throw OpenAIError.noContent(responseDetail(from: data))
        }
        return fallback
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
        the card number. For basketball, the year is ALWAYS the first year of the card's \
        season—not the copyright, licensing, or printed production year. For example, a \
        2025-26 card is year "2025" for both Panini and Topps; a Topps ©2026 line means \
        year "2025", while a Panini 2024-25 card is year "2024" even if its copyright says \
        ©2025. Return ONLY a JSON object with keys: \
        year, last_name, manufacturer, series, number. If unknown, use 'Unknown'.
        """
        if !ocrFront.isEmpty { system += "\nOCR front text:\n\(ocrFront)\n" }
        if !ocrBack.isEmpty  { system += "\nOCR back text:\n\(ocrBack)\n" }
        if !ocrBackBottom.isEmpty {
            system += "\nOCR back bottom text (often includes the card year):\n\(ocrBackBottom)\n"
        }

        var body: [String: Any] = [
            "model": model,
            // Reasoning tokens count against this limit on GPT-5 models. A limit
            // of 400 can be exhausted before any visible JSON is emitted.
            "max_completion_tokens": 1_200,
            "response_format": [
                "type": "json_schema",
                "json_schema": [
                    "name": "card_details",
                    "strict": true,
                    "schema": [
                        "type": "object",
                        "properties": [
                            "year": ["type": "string"],
                            "last_name": ["type": "string"],
                            "manufacturer": ["type": "string"],
                            "series": ["type": "string"],
                            "number": ["type": "string"],
                        ],
                        "required": ["year", "last_name", "manufacturer", "series", "number"],
                        "additionalProperties": false,
                    ],
                ],
            ],
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
        if model.lowercased().hasPrefix("gpt-5") {
            body["reasoning_effort"] = "low"
        }

        var req = URLRequest(url: URL(string: "https://api.openai.com/v1/chat/completions")!)
        req.httpMethod = "POST"
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: req)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw OpenAIError.badResponse(code, apiErrorMessage(from: data)) }

        return try parseResponse(data: data)
    }

    // MARK: - Parsing

    private static func parseResponse(data: Data) throws -> CardDetails {
        guard let outer = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = outer["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw OpenAIError.noContent(responseDetail(from: data))
        }

        let cleaned = stripCodeFences(content)
        guard !cleaned.isEmpty else {
            throw OpenAIError.noContent(responseDetail(from: data))
        }
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

    private static func apiErrorMessage(from data: Data) -> String {
        if let outer = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let error = outer["error"] as? [String: Any],
           let message = error["message"] as? String {
            return message
        }
        return String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .prefix(500)
            .description ?? ""
    }

    private static func responseDetail(from data: Data) -> String {
        guard let outer = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = outer["choices"] as? [[String: Any]],
              let choice = choices.first else {
            return ""
        }
        var parts: [String] = []
        if let finishReason = choice["finish_reason"] as? String {
            parts.append("finish reason: \(finishReason)")
        }
        if let message = choice["message"] as? [String: Any],
           let refusal = message["refusal"] as? String, !refusal.isEmpty {
            parts.append("refusal: \(refusal.prefix(200))")
        }
        return parts.joined(separator: ", ")
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
