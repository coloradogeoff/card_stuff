import SwiftUI

extension Notification.Name {
    static let goToDirectory = Notification.Name("goToDirectory")
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

@main
struct CardNamerApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
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
