import Combine
import Foundation

enum PostageRateID: String, CaseIterable, Codable, Identifiable {
    case firstClassLetter1Ounce
    case firstClassLetter2Ounces
    case firstClassLetter3Ounces
    case domesticFlat2Ounces
    case domesticFlat3Ounces
    case internationalLetter1Ounce
    case canadaLetter2Ounces
    case mexicoLetter2Ounces
    case otherCountriesLetter2Ounces

    var id: String { rawValue }

    var label: String {
        switch self {
        case .firstClassLetter1Ounce:
            return "First-Class Letter, 1 oz"
        case .firstClassLetter2Ounces:
            return "First-Class Letter, 2 oz"
        case .firstClassLetter3Ounces:
            return "First-Class Letter, 3 oz"
        case .domesticFlat2Ounces:
            return "Domestic Flat, 2 oz"
        case .domesticFlat3Ounces:
            return "Domestic Flat, 3 oz"
        case .internationalLetter1Ounce:
            return "International Letter, 1 oz"
        case .canadaLetter2Ounces:
            return "Canada Letter, 2 oz"
        case .mexicoLetter2Ounces:
            return "Mexico Letter, 2 oz"
        case .otherCountriesLetter2Ounces:
            return "Other Countries Letter, 2 oz"
        }
    }

    var defaultCents: Int {
        switch self {
        case .firstClassLetter1Ounce: return 78
        case .firstClassLetter2Ounces: return 107
        case .firstClassLetter3Ounces: return 136
        case .domesticFlat2Ounces: return 190
        case .domesticFlat3Ounces: return 217
        case .internationalLetter1Ounce: return 170
        case .canadaLetter2Ounces: return 200
        case .mexicoLetter2Ounces: return 255
        case .otherCountriesLetter2Ounces: return 340
        }
    }
}

struct PostageRate: Codable, Identifiable, Equatable {
    let id: PostageRateID
    var cents: Int
}

struct PostageSettings: Codable, Equatable {
    var rates: [PostageRate]
    var stampDenominations: [Int]

    static let defaults = PostageSettings(
        rates: PostageRateID.allCases.map {
            PostageRate(id: $0, cents: $0.defaultCents)
        },
        stampDenominations: [1, 2, 3, 4, 5, 10, 20, 25, 29, 40, 50, 78, 170]
    )

    func normalized() -> PostageSettings {
        let savedRates = rates.reduce(into: [PostageRateID: Int]()) { result, rate in
            result[rate.id] = rate.cents
        }
        let normalizedRates = PostageRateID.allCases.map { id in
            PostageRate(id: id, cents: max(1, savedRates[id] ?? id.defaultCents))
        }
        let normalizedDenominations = Array(Set(stampDenominations.filter { $0 > 0 })).sorted()

        return PostageSettings(
            rates: normalizedRates,
            stampDenominations: normalizedDenominations.isEmpty
                ? PostageSettings.defaults.stampDenominations
                : normalizedDenominations
        )
    }
}

final class PostageStore: ObservableObject {
    @Published private(set) var settings: PostageSettings

    init() {
        settings = Self.load()
    }

    func save(_ newSettings: PostageSettings) throws {
        let normalized = newSettings.normalized()
        let url = try Self.settingsURL()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(normalized)
        try data.write(to: url, options: .atomic)
        settings = normalized
    }

    private static func load() -> PostageSettings {
        do {
            let url = try settingsURL()
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(PostageSettings.self, from: data).normalized()
        } catch {
            return .defaults
        }
    }

    private static func settingsURL() throws -> URL {
        let applicationSupport = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = applicationSupport.appendingPathComponent("NedMail", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory.appendingPathComponent("postage-settings.json")
    }
}

struct StampSelection: Equatable {
    let counts: [Int: Int]
    let totalCents: Int
    let targetCents: Int
    let exceedsTolerance: Bool

    var stampCount: Int {
        counts.values.reduce(0, +)
    }

    var overpaymentCents: Int {
        totalCents - targetCents
    }
}

enum StampCombinationSolver {
    static func solve(
        targetCents: Int,
        denominations: [Int],
        toleranceCents: Int,
        requiredDenominations: Set<Int> = []
    ) -> StampSelection? {
        let coins = Array(Set(denominations.filter { $0 > 0 })).sorted(by: >)
        guard targetCents > 0, let largestCoin = coins.first else { return nil }

        let requiredCoins = requiredDenominations.intersection(coins)
        let requiredTotal = requiredCoins.reduce(0, +)
        let upperBound = max(
            targetCents + max(toleranceCents, largestCoin),
            requiredTotal
        )
        var combinations = Array<[Int]?>(repeating: nil, count: upperBound + 1)
        var requiredCombination = Array(repeating: 0, count: coins.count)
        for (index, coin) in coins.enumerated() where requiredCoins.contains(coin) {
            requiredCombination[index] = 1
        }
        combinations[requiredTotal] = requiredCombination

        guard requiredTotal < upperBound else {
            return makeSelection(
                coins: coins,
                combination: requiredCombination,
                total: requiredTotal,
                targetCents: targetCents,
                toleranceCents: toleranceCents
            )
        }
        for total in (requiredTotal + 1)...upperBound {
            for (index, coin) in coins.enumerated() where coin <= total {
                guard var candidate = combinations[total - coin] else { continue }
                candidate[index] += 1
                if isBetter(candidate, than: combinations[total]) {
                    combinations[total] = candidate
                }
            }
        }

        let toleranceUpperBound = min(upperBound, targetCents + max(0, toleranceCents))
        let withinTolerance = (targetCents...toleranceUpperBound).compactMap { total -> (Int, [Int])? in
            guard let combination = combinations[total] else { return nil }
            return (total, combination)
        }

        let selected: (Int, [Int])?
        let exceedsTolerance: Bool
        if !withinTolerance.isEmpty {
            selected = withinTolerance.min { lhs, rhs in
                let lhsCount = lhs.1.reduce(0, +)
                let rhsCount = rhs.1.reduce(0, +)
                if lhsCount != rhsCount { return lhsCount < rhsCount }
                if lhs.0 != rhs.0 { return lhs.0 < rhs.0 }
                return prefersLargerDenominations(lhs.1, over: rhs.1)
            }
            exceedsTolerance = false
        } else {
            if toleranceUpperBound < upperBound {
                selected = ((toleranceUpperBound + 1)...upperBound).compactMap { total -> (Int, [Int])? in
                    guard let combination = combinations[total] else { return nil }
                    return (total, combination)
                }.first
            } else {
                selected = nil
            }
            exceedsTolerance = true
        }

        guard let (total, combination) = selected else { return nil }
        return makeSelection(
            coins: coins,
            combination: combination,
            total: total,
            targetCents: targetCents,
            exceedsTolerance: exceedsTolerance
        )
    }

    private static func makeSelection(
        coins: [Int],
        combination: [Int],
        total: Int,
        targetCents: Int,
        toleranceCents: Int
    ) -> StampSelection {
        makeSelection(
            coins: coins,
            combination: combination,
            total: total,
            targetCents: targetCents,
            exceedsTolerance: total > targetCents + max(0, toleranceCents)
        )
    }

    private static func makeSelection(
        coins: [Int],
        combination: [Int],
        total: Int,
        targetCents: Int,
        exceedsTolerance: Bool
    ) -> StampSelection {
        let counts = Dictionary(uniqueKeysWithValues: zip(coins, combination).compactMap {
            $0.1 > 0 ? ($0.0, $0.1) : nil
        })
        return StampSelection(
            counts: counts,
            totalCents: total,
            targetCents: targetCents,
            exceedsTolerance: exceedsTolerance
        )
    }

    private static func isBetter(_ candidate: [Int], than existing: [Int]?) -> Bool {
        guard let existing else { return true }
        let candidateCount = candidate.reduce(0, +)
        let existingCount = existing.reduce(0, +)
        if candidateCount != existingCount {
            return candidateCount < existingCount
        }
        return prefersLargerDenominations(candidate, over: existing)
    }

    private static func prefersLargerDenominations(_ lhs: [Int], over rhs: [Int]) -> Bool {
        for (leftCount, rightCount) in zip(lhs, rhs) where leftCount != rightCount {
            return leftCount > rightCount
        }
        return false
    }
}
