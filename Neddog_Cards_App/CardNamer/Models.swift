import Foundation

struct CardPair: Identifiable, Equatable {
    let id = UUID()
    let front: URL
    let back: URL

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
