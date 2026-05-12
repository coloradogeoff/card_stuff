import Foundation

struct CardPair: Identifiable, Equatable {
    let front: URL
    let back: URL

    var id: String {
        "\(front.standardized.path)|\(back.standardized.path)"
    }

    var displayName: String {
        front.lastPathComponent
    }

    var baseName: String {
        let stem = front.deletingPathExtension().lastPathComponent
        if stem.lowercased().hasSuffix("_b") {
            return String(stem.dropLast(2))
        }
        return stem
    }

    var modificationDate: Date {
        let frontDate = (try? front.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        let backDate = (try? back.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        return max(frontDate, backDate)
    }
}

enum CardPairSortField: String, CaseIterable, Identifiable {
    case name = "Name"
    case modificationDate = "Mod Date"

    var id: Self { self }
}

enum CardPairSortOrder: String, CaseIterable, Identifiable {
    case ascending = "Ascending"
    case descending = "Descending"

    var id: Self { self }
}

enum CardTrait: String, CaseIterable, Codable, Identifiable {
    case autograph
    case graded
    case rookie
    case numbered
    case patchJersey
    case insertCaseHit

    var id: Self { self }

    var label: String {
        switch self {
        case .autograph: "Auto"
        case .graded: "Graded"
        case .rookie: "Rookie"
        case .numbered: "Numbered"
        case .patchJersey: "Patch/Jersey"
        case .insertCaseHit: "Insert/Case Hit"
        }
    }

    var systemImage: String {
        switch self {
        case .autograph: "signature"
        case .graded: "seal"
        case .rookie: "star"
        case .numbered: "number"
        case .patchJersey: "tshirt"
        case .insertCaseHit: "sparkles"
        }
    }
}

struct CardTraits: Codable, Equatable {
    var autograph: Bool = false
    var graded: Bool = false
    var rookie: Bool = false
    var numbered: Bool = false
    var patchJersey: Bool = false
    var insertCaseHit: Bool = false
    var updatedAt: Date?

    var hasAnyTrait: Bool {
        autograph || graded || rookie || numbered || patchJersey || insertCaseHit
    }

    func contains(_ trait: CardTrait) -> Bool {
        switch trait {
        case .autograph: autograph
        case .graded: graded
        case .rookie: rookie
        case .numbered: numbered
        case .patchJersey: patchJersey
        case .insertCaseHit: insertCaseHit
        }
    }

    mutating func set(_ trait: CardTrait, to value: Bool) {
        switch trait {
        case .autograph: autograph = value
        case .graded: graded = value
        case .rookie: rookie = value
        case .numbered: numbered = value
        case .patchJersey: patchJersey = value
        case .insertCaseHit: insertCaseHit = value
        }
        updatedAt = Date()
    }
}

struct CardDetails {
    var year: String = "Unknown"
    var lastName: String = "Unknown"
    var manufacturer: String = "Unknown"
    var series: String = "Unknown"
    var number: String = "Unknown"
}

struct Settings: Codable {
    var existingCardsDirectory: String?
}
