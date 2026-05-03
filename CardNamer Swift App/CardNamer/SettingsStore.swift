import Foundation

struct QuickDirectory: Codable, Identifiable {
    var id: String { path }
    var name: String
    var path: String

    var url: URL { URL(fileURLWithPath: path) }
    var isAvailable: Bool { FileManager.default.fileExists(atPath: path) }
}

final class SettingsStore {

    static let shared = SettingsStore()
    private let defaults = UserDefaults.standard

    var openAIKey: String {
        get { defaults.string(forKey: "openai_api_key") ?? "" }
        set { defaults.set(newValue, forKey: "openai_api_key") }
    }

    var selectedModel: String {
        get { defaults.string(forKey: "selectedModel") ?? "gpt-4o" }
        set { defaults.set(newValue, forKey: "selectedModel") }
    }

    var quickDirectories: [QuickDirectory] {
        get {
            guard let data = defaults.data(forKey: "quickDirectories"),
                  let dirs = try? JSONDecoder().decode([QuickDirectory].self, from: data) else {
                return defaultQuickDirectories
            }
            return dirs
        }
        set {
            if let data = try? JSONEncoder().encode(newValue) {
                defaults.set(data, forKey: "quickDirectories")
            }
        }
    }

    var existingCardsDirectory: URL {
        let fallback = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Card Namer/Cards")
        try? FileManager.default.createDirectory(at: fallback, withIntermediateDirectories: true)
        return fallback
    }

    private var defaultQuickDirectories: [QuickDirectory] {
        [
            QuickDirectory(name: "Incoming Cards", path: "/Users/geoff/incoming cards"),
            QuickDirectory(name: "Desktop/Cards", path: "/Volumes/Dutton 2TB/Cards/Mix"),
        ]
    }
}
