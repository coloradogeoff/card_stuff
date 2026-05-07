import AppKit
import SwiftUI

extension Notification.Name {
    static let goToDirectory = Notification.Name("goToDirectory")
    static let showEbayResultsWindow = Notification.Name("showEbayResultsWindow")
    static let refreshImages = Notification.Name("refreshImages")
}

private enum DirectoryShortcut: String, CaseIterable {
    case incoming = "1"
    case desktopCards = "2"
    case currentSales = "3"
}

private enum AppWindowMetrics {
    static let minWidth: CGFloat = 1200
    static let minHeight: CGFloat = 800
    static let defaultWidth: CGFloat = 1400
    static let defaultHeight: CGFloat = 900
}

enum SceneID {
    static let ebayResults = "ebayResults"
}

@main
struct CardNamerApp: App {
    @NSApplicationDelegateAdaptor(CardNamerAppDelegate.self) private var appDelegate
    @State private var cardVM = CardNamerViewModel()
    @State private var ebayVM = EbayTitlesViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(cardVM: cardVM, ebayVM: ebayVM)
                .frame(
                    minWidth: AppWindowMetrics.minWidth,
                    minHeight: AppWindowMetrics.minHeight
                )
        }
        .defaultSize(
            width: AppWindowMetrics.defaultWidth,
            height: AppWindowMetrics.defaultHeight
        )
        .windowStyle(.titleBar)
        WindowGroup("eBay Title Results", id: SceneID.ebayResults) {
            EbayTitlesResultsWindow(vm: ebayVM)
        }
        .defaultSize(width: 840, height: 560)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Go") {
                directoryCommand(
                    title: "Incoming (⌘1)",
                    directory: QuickDirectory(
                        name: "Incoming (⌘1)",
                        path: SettingsStore.shared.incomingDirectory.path
                    ),
                    shortcut: .incoming
                )
                directoryCommand(
                    title: "Desktop/Cards (⌘2)",
                    directory: QuickDirectory(
                        name: "Desktop/Cards (⌘2)",
                        path: SettingsStore.shared.desktopCardsDirectory.path
                    ),
                    shortcut: .desktopCards
                )
                directoryCommand(
                    title: "Current Sales (⌘3)",
                    directory: QuickDirectory(
                        name: "Current Sales (⌘3)",
                        path: SettingsStore.shared.currentSalesDirectory.path
                    ),
                    shortcut: .currentSales
                )
                Divider()
                Button("Refresh") {
                    NotificationCenter.default.post(name: .refreshImages, object: nil)
                }
                .keyboardShortcut("r", modifiers: .command)
            }
        }
    }

    private func directoryCommand(
        title: String,
        directory: QuickDirectory,
        shortcut: DirectoryShortcut
    ) -> some View {
        Button(title) {
            NotificationCenter.default.post(name: .goToDirectory, object: directory)
        }
        .keyboardShortcut(KeyEquivalent(Character(shortcut.rawValue)), modifiers: .command)
        .disabled(!directory.isAvailable)
    }
}

final class CardNamerAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
