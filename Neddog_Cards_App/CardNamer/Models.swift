import Foundation

struct CardPair: Identifiable, Equatable {
    let front: URL
    let back: URL
    let id: String
    let modificationDate: Date

    init(front: URL, back: URL, modificationDate: Date? = nil) {
        self.front = front
        self.back = back
        self.id = "\(front.standardized.path)|\(back.standardized.path)"
        if let modificationDate {
            self.modificationDate = modificationDate
        } else {
            let fd = (try? front.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let bd = (try? back.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            self.modificationDate = max(fd, bd)
        }
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

    // Filename format: YYYY-Player-Manufacturer-Series-...-Number
    var parsedYear: String {
        let first = baseName.split(separator: "-", maxSplits: 1).first.map(String.init) ?? ""
        return first.count == 4 && first.allSatisfy(\.isNumber) ? first : ""
    }

    var parsedPlayer: String {
        let parts = baseName.split(separator: "-")
        guard parts.count >= 2 else { return baseName.lowercased() }
        return String(parts[1]).lowercased()
    }

    var parsedSetText: String {
        let parts = baseName.split(separator: "-")
        guard parts.count >= 3 else { return "" }
        return parts[2...].joined(separator: " ").lowercased()
    }

    /// Manufacturer, series, number, and variation, parsed in a single pass.
    /// Filename format: YYYY-Player-Manufacturer-Series-[Variation-]Number.
    /// Callers that need several of these fields (e.g. sorting) should call
    /// this once and read fields off the result, rather than calling the
    /// individual `parsed*` properties repeatedly — each of those re-splits
    /// the filename from scratch, which is fine for a single lookup but adds
    /// up fast if done on every pairwise comparison in a sort.
    var nameComponents: CardNameComponents {
        let parts = baseName.split(separator: "-")

        let manufacturer = parts.count > 2 ? String(parts[2]).lowercased() : ""
        let series = parts.count > 3 ? String(parts[3]).lowercased() : ""

        let number: Int?
        if let last = parts.last {
            let digits = last.prefix(while: \.isNumber)
            number = digits.isEmpty ? nil : Int(digits)
        } else {
            number = nil
        }

        let variation = parts.count > 4 ? parts[4..<(parts.count - 1)].joined(separator: " ").lowercased() : ""

        return CardNameComponents(manufacturer: manufacturer, series: series, number: number, variation: variation)
    }

}

struct CardNameComponents {
    let manufacturer: String
    let series: String
    let number: Int?
    let variation: String
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

/// An eBay listing record for one card: the title we generated (or that was
/// hand-edited), whether it has actually been listed, and when.
///
/// Persisted by `CardListingStore` in `ebay_metadata.json`, keyed by
/// `CardPair.baseName` — deliberately a separate file from the traits in
/// `card_metadata.json` so listing state can never endanger trait data.
struct CardListing: Codable, Equatable {
    var title: String?
    var listed: Bool = false
    var listedAt: Date?
    var updatedAt: Date?
    /// Raw value of the `EbayCategory` whose rules produced `title`. Stored as a
    /// plain string, not the enum, so a category that is later renamed or removed
    /// degrades to "unknown" instead of failing the whole record's decode.
    var category: String?

    /// An empty record is never written to disk, matching `CardTraits`.
    var isEmpty: Bool { (title?.isEmpty ?? true) && !listed }

    /// The category whose rules generated this title, when it is still a
    /// category the app offers.
    var ebayCategory: EbayCategory? {
        category.flatMap(EbayCategory.init(rawValue:))
    }

    init(
        title: String? = nil,
        listed: Bool = false,
        listedAt: Date? = nil,
        updatedAt: Date? = nil,
        category: String? = nil
    ) {
        self.title = title
        self.listed = listed
        self.listedAt = listedAt
        self.updatedAt = updatedAt
        self.category = category
    }

    // Swift's synthesized `Decodable` ignores default property values and calls
    // `decode(_:forKey:)` for every non-optional property, so adding a field
    // later would throw `keyNotFound` on every record already on disk. Decoding
    // each key with `decodeIfPresent` keeps older files readable.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        title = try container.decodeIfPresent(String.self, forKey: .title)
        listed = try container.decodeIfPresent(Bool.self, forKey: .listed) ?? false
        listedAt = try container.decodeIfPresent(Date.self, forKey: .listedAt)
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt)
        category = try container.decodeIfPresent(String.self, forKey: .category)
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
