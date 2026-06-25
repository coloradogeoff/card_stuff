import Foundation

struct BasketballRookieLookup: Sendable {
    private let namesBySeason: [String: Set<String>]

    static let bundled: BasketballRookieLookup = {
        guard let url = Bundle.main.url(forResource: "basketball-rookies", withExtension: "csv"),
              let csv = try? String(contentsOf: url, encoding: .utf8) else {
            return BasketballRookieLookup(namesBySeason: [:])
        }
        return BasketballRookieLookup(csv: csv)
    }()

    init(csv: String) {
        var names: [String: Set<String>] = [:]

        for line in csv.split(whereSeparator: \.isNewline).dropFirst() {
            let columns = line.split(separator: ",", maxSplits: 2, omittingEmptySubsequences: false)
            guard columns.count == 3 else { continue }

            let season = String(columns[0]).trimmingCharacters(in: .whitespacesAndNewlines)
            let player = String(columns[2]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !season.isEmpty, !player.isEmpty else { continue }

            for key in Self.normalizedVariants(player) {
                names[season, default: []].insert(key)
            }
        }

        namesBySeason = names
    }

    private init(namesBySeason: [String: Set<String>]) {
        self.namesBySeason = namesBySeason
    }

    func isRookie(player: String, season: String) -> Bool {
        guard let seasonNames = namesBySeason[season], !seasonNames.isEmpty else { return false }
        let variants = Self.normalizedVariants(player)

        if variants.contains(where: seasonNames.contains) {
            return true
        }

        // Tolerate one OCR/AI character error only when exactly one season
        // entry is that close. Ambiguous names are intentionally rejected.
        for variant in variants where variant.count >= 8 {
            let closeMatches = seasonNames.filter {
                abs($0.count - variant.count) <= 1 && Self.editDistanceAtMostOne(variant, $0)
            }
            if closeMatches.count == 1 {
                return true
            }
        }

        return false
    }

    func correctingRookieMarker(in title: String) -> String {
        guard let parsed = Self.playerAndSeason(from: title) else { return title }
        let shouldBeRookie = isRookie(player: parsed.player, season: parsed.season)
        let titleWithoutMarker = title.replacingOccurrences(
            of: #"\s*\(RC\)"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )

        guard shouldBeRookie else { return titleWithoutMarker }
        guard let separatorRange = titleWithoutMarker.range(of: " - ") else {
            return titleWithoutMarker
        }

        var corrected = titleWithoutMarker
        corrected.insert(contentsOf: " (RC)", at: separatorRange.lowerBound)
        return corrected
    }

    private static func playerAndSeason(from title: String) -> (player: String, season: String)? {
        guard let separatorRange = title.range(of: " - ") else { return nil }
        let player = String(title[..<separatorRange.lowerBound])
            .replacingOccurrences(
                of: #"\s*\(RC\)"#,
                with: "",
                options: [.regularExpression, .caseInsensitive]
            )
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard let seasonRange = title.range(
            of: #"\b20\d{2}(?:-\d{2})?\b"#,
            options: .regularExpression
        ) else {
            return nil
        }

        return (player, String(title[seasonRange]))
    }

    private static func normalizedVariants(_ name: String) -> Set<String> {
        let latin = name.applyingTransform(.toLatin, reverse: false) ?? name
        let folded = latin.folding(
            options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive],
            locale: Locale(identifier: "en_US_POSIX")
        )
        let words = folded
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }

        guard !words.isEmpty else { return [] }

        var variants = [words.joined()]
        if let suffix = words.last,
           ["jr", "sr", "ii", "iii", "iv"].contains(suffix),
           words.count > 1 {
            variants.append(words.dropLast().joined())
        }
        return Set(variants)
    }

    private static func editDistanceAtMostOne(_ lhs: String, _ rhs: String) -> Bool {
        if lhs == rhs { return true }
        let left = Array(lhs)
        let right = Array(rhs)
        if abs(left.count - right.count) > 1 { return false }

        if left.count == right.count {
            var differences = 0
            for index in left.indices where left[index] != right[index] {
                differences += 1
                if differences > 1 { return false }
            }
            return true
        }

        let shorter = left.count < right.count ? left : right
        let longer = left.count < right.count ? right : left
        var shortIndex = 0
        var longIndex = 0
        var skipped = false

        while shortIndex < shorter.count, longIndex < longer.count {
            if shorter[shortIndex] == longer[longIndex] {
                shortIndex += 1
                longIndex += 1
            } else {
                if skipped { return false }
                skipped = true
                longIndex += 1
            }
        }

        return true
    }
}
