import Foundation

struct CardMetadataFile: Codable {
    var version: Int = 1
    var cards: [String: CardTraits] = [:]
}

final class CardMetadataStore {
    static let fileName = "card_metadata.json"

    private(set) var directoryURL: URL
    private var metadata = CardMetadataFile()

    init(directoryURL: URL) {
        self.directoryURL = directoryURL.standardized
        load(directoryURL: directoryURL)
    }

    func load(directoryURL: URL) {
        self.directoryURL = directoryURL.standardized
        let url = metadataURL(for: self.directoryURL)
        guard FileManager.default.fileExists(atPath: url.path) else {
            metadata = CardMetadataFile()
            return
        }

        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            metadata = try decoder.decode(CardMetadataFile.self, from: data)
        } catch {
            metadata = CardMetadataFile()
        }
    }

    func traits(for pair: CardPair) -> CardTraits {
        metadata.cards[pair.baseName] ?? CardTraits()
    }

    func traits(forBaseName baseName: String) -> CardTraits {
        metadata.cards[baseName] ?? CardTraits()
    }

    func set(_ trait: CardTrait, to value: Bool, for pair: CardPair) {
        var traits = metadata.cards[pair.baseName] ?? CardTraits()
        traits.set(trait, to: value)
        setTraits(traits, forBaseName: pair.baseName)
    }

    func removeMetadata(for pair: CardPair) {
        metadata.cards.removeValue(forKey: pair.baseName)
        save()
    }

    func pruneMetadata(keepingBaseNames validBaseNames: Set<String>) {
        let originalCount = metadata.cards.count
        metadata.cards = metadata.cards.filter { validBaseNames.contains($0.key) }
        if metadata.cards.count != originalCount {
            save()
        }
    }

    func moveMetadata(from oldBaseName: String, to newBaseName: String) {
        guard oldBaseName != newBaseName, var traits = metadata.cards.removeValue(forKey: oldBaseName) else { return }
        traits.updatedAt = Date()
        setTraits(traits, forBaseName: newBaseName)
    }

    func moveMetadata(for pair: CardPair, to destinationDirectory: URL) {
        let traits = self.traits(for: pair)
        guard traits.hasAnyTrait else { return }

        let destinationStore = CardMetadataStore(directoryURL: destinationDirectory)
        destinationStore.setTraits(traits, forBaseName: pair.baseName)
        metadata.cards.removeValue(forKey: pair.baseName)
        save()
    }

    func matches(_ pair: CardPair, selectedTraits: Set<CardTrait>) -> Bool {
        guard !selectedTraits.isEmpty else { return true }
        let traits = traits(for: pair)
        return selectedTraits.allSatisfy { traits.contains($0) }
    }

    private func setTraits(_ traits: CardTraits, forBaseName baseName: String) {
        if traits.hasAnyTrait {
            metadata.cards[baseName] = traits
        } else {
            metadata.cards.removeValue(forKey: baseName)
        }
        save()
    }

    private func save() {
        let url = metadataURL(for: directoryURL)
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(metadata)
            try data.write(to: url, options: .atomic)
        } catch {
            // Metadata is non-critical; view models continue to operate if a save fails.
        }
    }

    private func metadataURL(for directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(Self.fileName)
    }
}
