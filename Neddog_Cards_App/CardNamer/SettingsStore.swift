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
    private enum DirectoryNames {
        static let incoming = "Incoming (⌘1)"
        static let desktopCards = "Desktop/Cards (⌘2)"
        static let currentSales = "Current Sales (⌘3)"
    }

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
        [
            QuickDirectory(name: DirectoryNames.incoming, path: incomingDirectory.path),
            QuickDirectory(name: DirectoryNames.desktopCards, path: desktopCardsDirectory.path),
            QuickDirectory(name: DirectoryNames.currentSales, path: currentSalesDirectory.path),
        ]
    }

    var incomingDirectory: URL {
        URL(fileURLWithPath: "/Users/geoff/incoming cards")
    }

    var desktopCardsDirectory: URL {
        URL(fileURLWithPath: "/Volumes/Dutton 2TB/Cards/Mix")
    }

    var currentSalesDirectory: URL {
        let fileManager = FileManager.default
        let envRoot = ProcessInfo.processInfo.environment["SALES_ROOT"]
        let root = URL(fileURLWithPath: envRoot ?? NSString(string: "~/Sales").expandingTildeInPath)
            .standardizedFileURL

        // Match sale.py: refuse to create the month directory if the sales root
        // is a broken symlink, and surface the unresolved target path instead.
        if let values = try? root.resourceValues(forKeys: [.isSymbolicLinkKey]),
           values.isSymbolicLink == true,
           !fileManager.fileExists(atPath: root.path) {
            return root
        }

        let components = Calendar.current.dateComponents([.year, .month], from: Date())
        let year = components.year ?? 1970
        let month = components.month ?? 1

        let target = root
            .appendingPathComponent(String(format: "%04d", year))
            .appendingPathComponent(String(format: "%02d", month))
        try? fileManager.createDirectory(at: target, withIntermediateDirectories: true)
        return target
    }

    var existingCardsDirectory: URL {
        let target = URL(fileURLWithPath: "/Volumes/Dutton 2TB/Cards/Mix")
        try? FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        return target
    }
}
