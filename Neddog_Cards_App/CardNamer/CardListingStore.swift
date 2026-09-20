import Foundation

struct CardListingFile: Codable {
    var version: Int = 1
    var listings: [String: CardListing] = [:]

    init(version: Int = 1, listings: [String: CardListing] = [:]) {
        self.version = version
        self.listings = listings
    }

    // See the note on `CardListing.init(from:)`: default property values are not
    // applied by the synthesized decoder, so every key is decoded optionally to
    // keep files written by older builds readable.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decodeIfPresent(Int.self, forKey: .version) ?? 1
        listings = try container.decodeIfPresent([String: CardListing].self, forKey: .listings) ?? [:]
    }
}

/// Per-directory store for eBay listing records, kept deliberately separate from
/// `CardMetadataStore`'s `card_metadata.json` so that listing state and trait
/// state can never corrupt one another. Mirrors that class's shape.
final class CardListingStore {
    static let fileName = "ebay_metadata.json"

    private(set) var directoryURL: URL
    private var file = CardListingFile()

    init(directoryURL: URL) {
        self.directoryURL = directoryURL.standardized
        load(directoryURL: directoryURL)
    }

    func load(directoryURL: URL) {
        self.directoryURL = directoryURL.standardized
        let url = listingURL(for: self.directoryURL)
        guard FileManager.default.fileExists(atPath: url.path) else {
            file = CardListingFile()
            return
        }

        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            file = try decoder.decode(CardListingFile.self, from: data)
        } catch {
            file = CardListingFile()
        }
    }

    func listing(for pair: CardPair) -> CardListing {
        file.listings[pair.baseName] ?? CardListing()
    }

    func listing(forBaseName baseName: String) -> CardListing {
        file.listings[baseName] ?? CardListing()
    }

    var listedBaseNames: Set<String> {
        Set(file.listings.compactMap { $0.value.listed ? $0.key : nil })
    }

    /// Records a title and, when known, the category whose rules produced it.
    /// A nil `category` means "leave the existing one alone": hand edits and CSV
    /// backfills change the text without changing which rules generated it.
    func setTitle(_ title: String, category: EbayCategory? = nil, for pair: CardPair) {
        var listing = self.listing(for: pair)
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let newTitle = trimmed.isEmpty ? nil : trimmed
        let newCategory = category?.rawValue ?? listing.category

        guard listing.title != newTitle || listing.category != newCategory else { return }
        listing.title = newTitle
        listing.category = newCategory
        listing.updatedAt = Date()
        setListing(listing, forBaseName: pair.baseName)
    }

    /// Marking keeps whatever title is already stored, so unlisting a card never
    /// throws away the title it was listed under.
    func setListed(_ value: Bool, for pair: CardPair) {
        var listing = self.listing(for: pair)
        listing.listed = value
        listing.listedAt = value ? Date() : nil
        listing.updatedAt = Date()
        setListing(listing, forBaseName: pair.baseName)
    }

    func removeListing(for pair: CardPair) {
        guard file.listings.removeValue(forKey: pair.baseName) != nil else { return }
        save()
    }

    func pruneListings(keepingBaseNames validBaseNames: Set<String>) {
        let originalCount = file.listings.count
        file.listings = file.listings.filter { validBaseNames.contains($0.key) }
        if file.listings.count != originalCount {
            save()
        }
    }

    /// Follows a card through a rename within the same directory.
    func moveListing(from oldBaseName: String, to newBaseName: String) {
        guard oldBaseName != newBaseName, var listing = file.listings.removeValue(forKey: oldBaseName) else { return }
        listing.updatedAt = Date()
        setListing(listing, forBaseName: newBaseName)
    }

    /// Follows a card into another directory's store.
    func moveListing(for pair: CardPair, to destinationDirectory: URL) {
        let listing = self.listing(for: pair)
        guard !listing.isEmpty else { return }

        let destinationStore = CardListingStore(directoryURL: destinationDirectory)
        destinationStore.setListing(listing, forBaseName: pair.baseName)
        file.listings.removeValue(forKey: pair.baseName)
        save()
    }

    private func setListing(_ listing: CardListing, forBaseName baseName: String) {
        if listing.isEmpty {
            file.listings.removeValue(forKey: baseName)
        } else {
            file.listings[baseName] = listing
        }
        save()
    }

    private func save() {
        let url = listingURL(for: directoryURL)
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(file)
            try data.write(to: url, options: .atomic)
        } catch {
            // Listing records are non-critical; view models continue to operate if a save fails.
        }
    }

    private func listingURL(for directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(Self.fileName)
    }
}
