import Foundation

enum CardNameBuilder {

    // MARK: - Public API

    static func buildBaseName(from details: CardDetails) -> String {
        let year = normalizeYear(details.year)
        let last = extractLastName(details.lastName)
        let mfg = slugify(details.manufacturer.isEmpty ? "Unknown" : details.manufacturer)
        let series = slugify(details.series.isEmpty ? "Unknown" : details.series)
        let number = slugify(details.number.isEmpty ? "Unknown" : details.number)
        return [slugify(year), slugify(last), mfg, series, number]
            .filter { !$0.isEmpty }
            .joined(separator: "-")
    }

    static func sanitize(_ value: String) -> String {
        var s = value.trimmingCharacters(in: .init(charactersIn: " ."))
        s = s.replacingOccurrences(of: #"[\\/:\*\?"<>\|]+"#, with: "-", options: .regularExpression)
        s = s.replacingOccurrences(of: #"\s+"#, with: "-", options: .regularExpression)
        s = s.replacingOccurrences(of: #"-{2,}"#, with: "-", options: .regularExpression)
        return s
    }

    static func tcdbURL(fromBaseName name: String) -> URL? {
        let parts = name.components(separatedBy: "-").filter { !$0.isEmpty }
        guard parts.count >= 5 else { return nil }
        let year = parts[0]
        let last = parts[1]
        let mfg = parts[2]
        let series = parts[3]
        let number = parts.last ?? "Unknown"
        let variety = parts.count > 5 ? parts[4..<(parts.count - 1)].joined(separator: "-") : "Base"

        let season = seasonString(fromYear: year, manufacturer: mfg)
        var tokens = [season, mfg, series]
        if !variety.isEmpty && variety.lowercased() != "base" && variety.lowercased() != "unknown" {
            tokens.append(variety)
        }
        tokens.append(number)
        tokens.append(last)

        let q = tokens.joined(separator: " ")
        var comps = URLComponents(string: "https://www.tcdb.com/Search.cfm")
        comps?.queryItems = [
            .init(name: "SearchCategory", value: "Basketball"),
            .init(name: "q", value: q),
        ]
        return comps?.url
    }

    static func ebayURL(fromBaseName name: String) -> URL? {
        let terms = name
            .replacingOccurrences(of: #"(?<!\d)-|-(?!\d)"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
        var comps = URLComponents(string: "https://www.ebay.com/sch/i.html")
        comps?.queryItems = [.init(name: "_nkw", value: terms)]
        return comps?.url
    }

    // MARK: - Year extraction (mirrors Python logic)

    static func refineYear(details: inout CardDetails, ocrFront: String, ocrBack: String, ocrBackBottom: String) {
        let combined = [ocrFront, ocrBack, ocrBackBottom, details.year].joined(separator: "\n")

        if let y = extractSetSeasonYear(from: ocrBackBottom) ?? extractSetSeasonYear(from: ocrBack) ?? extractSetSeasonYear(from: ocrFront) {
            details.year = y
            return
        }
        if let y = extractCopyrightYear(from: combined) {
            details.year = y
            return
        }
        let isTopps = (details.manufacturer.lowercased() == "topps") || combined.lowercased().contains("topps")
        if isTopps, let y = extractShortSeasonEndYear(from: combined) {
            details.year = y
            return
        }
        if let y = extractSeasonYear(from: combined) {
            details.year = y
            return
        }
        if details.year == "Unknown" || details.year.isEmpty {
            let fallback = [ocrBackBottom, ocrBack, ocrFront].joined(separator: "\n")
            if let y = firstYear(in: fallback, preferLast: true) {
                details.year = y
            }
        }
    }

    // MARK: - Private helpers

    private static let brands = #"(?:panini|topps|upper\s*deck|donruss|fleer|bowman|leaf|score)"#

    private static func extractSetSeasonYear(from text: String) -> String? {
        guard !text.isEmpty else { return nil }
        let patterns = [
            #"((?:19|20)\d{2})\s*[-/]\s*(\d{2}|(?:19|20)\d{2})[^\n]{0,60}"# + #"\b"# + brands + #"\b"#,
            #"\b"# + brands + #"\b[^\n]{0,60}((?:19|20)\d{2})\s*[-/]\s*(\d{2}|(?:19|20)\d{2})"#,
        ]
        var found: [String] = []
        for pattern in patterns {
            guard let re = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { continue }
            let range = NSRange(text.startIndex..., in: text)
            for match in re.matches(in: text, range: range) {
                if let r = Range(match.range(at: 1), in: text) {
                    found.append(String(text[r]))
                }
            }
        }
        return found.last
    }

    private static func extractCopyrightYear(from text: String) -> String? {
        guard let re = try? NSRegularExpression(pattern: #"(?:copyright|\(c\)|©|\(©\))"#, options: [.caseInsensitive]) else { return nil }
        for line in text.components(separatedBy: "\n") {
            let range = NSRange(line.startIndex..., in: line)
            if re.firstMatch(in: line, range: range) != nil {
                if let y = allYears(in: line).last { return y }
            }
        }
        return nil
    }

    private static func extractSeasonYear(from text: String) -> String? {
        var candidates: [Int] = []
        // Full: 2024-25 or 2024/2025
        if let re = try? NSRegularExpression(pattern: #"((?:19|20)\d{2})\s*[-/]\s*(\d{2}|(?:19|20)\d{2})"#) {
            let range = NSRange(text.startIndex..., in: text)
            for m in re.matches(in: text, range: range) {
                if let r = Range(m.range(at: 1), in: text), let y = Int(text[r]) { candidates.append(y) }
            }
        }
        // Short: 24-25
        if let re = try? NSRegularExpression(pattern: #"\b(\d{2})\s*[-/]\s*(\d{2})\b"#) {
            let range = NSRange(text.startIndex..., in: text)
            for m in re.matches(in: text, range: range) {
                if let r = Range(m.range(at: 1), in: text), let yy = Int(text[r]) {
                    candidates.append(expandShortYear(yy))
                }
            }
        }
        return candidates.max().map(String.init)
    }

    private static func extractShortSeasonEndYear(from text: String) -> String? {
        var candidates: [Int] = []
        if let re = try? NSRegularExpression(pattern: #"\b(\d{2})\s*[-/]\s*(\d{2})\b"#) {
            let range = NSRange(text.startIndex..., in: text)
            for m in re.matches(in: text, range: range) {
                if let r = Range(m.range(at: 2), in: text), let yy = Int(text[r]) {
                    candidates.append(expandShortYear(yy))
                }
            }
        }
        return candidates.max().map(String.init)
    }

    private static func expandShortYear(_ yy: Int) -> Int {
        yy <= 79 ? 2000 + yy : 1900 + yy
    }

    private static func allYears(in text: String) -> [String] {
        guard let re = try? NSRegularExpression(pattern: #"(?:19|20)\d{2}"#) else { return [] }
        let range = NSRange(text.startIndex..., in: text)
        return re.matches(in: text, range: range).compactMap { Range($0.range, in: text).map { String(text[$0]) } }
    }

    private static func firstYear(in text: String, preferLast: Bool) -> String? {
        let years = allYears(in: text)
        return preferLast ? years.last : years.first
    }

    private static func normalizeYear(_ value: String) -> String {
        guard let re = try? NSRegularExpression(pattern: #"(?:19|20)\d{2}"#) else { return "Unknown" }
        let range = NSRange(value.startIndex..., in: value)
        if let m = re.firstMatch(in: value, range: range), let r = Range(m.range, in: value) {
            return String(value[r])
        }
        return "Unknown"
    }

    private static func extractLastName(_ value: String) -> String {
        let tokens = value.split(whereSeparator: \.isWhitespace).map(String.init).filter { !$0.isEmpty }
        guard !tokens.isEmpty else { return "Unknown" }
        let suffixes: Set<String> = ["jr", "sr", "ii", "iii", "iv", "v"]
        var last = tokens.last!.trimmingCharacters(in: .init(charactersIn: "."))
        if suffixes.contains(last.lowercased()) && tokens.count > 1 {
            last = tokens[tokens.count - 2].trimmingCharacters(in: .init(charactersIn: "."))
        }
        return last.isEmpty ? "Unknown" : last
    }

    private static func slugify(_ value: String) -> String {
        let ascii = value.unicodeScalars.filter { $0.isASCII }.map { Character($0) }
        let s = String(ascii)
        var result = s.replacingOccurrences(of: #"[^\w\s-]"#, with: "", options: .regularExpression)
        result = result.replacingOccurrences(of: #"[\s_-]+"#, with: "-", options: .regularExpression)
        result = result.trimmingCharacters(in: .init(charactersIn: "-"))
        return result.isEmpty ? "Unknown" : result
    }

    private static func seasonString(fromYear year: String, manufacturer: String) -> String {
        guard let re = try? NSRegularExpression(pattern: #"(19|20)\d{2}"#),
              let m = re.firstMatch(in: year, range: NSRange(year.startIndex..., in: year)),
              let r = Range(m.range, in: year),
              let y = Int(year[r]) else { return year }
        let isTopps = manufacturer.lowercased() == "topps"
        let start = isTopps ? y - 1 : y
        let end = (start + 1) % 100
        return String(format: "%d-%02d", start, end)
    }
}
