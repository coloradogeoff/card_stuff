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
    static let defaultWidth: CGFloat = 1700
    static let defaultHeight: CGFloat = 1250
    static let startupMargin: CGFloat = 40
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
    private var didPlaceMainWindow = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        DispatchQueue.main.async { [weak self] in
            self?.configureStartupWindowIfNeeded()
        }
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        hideWindowTitles()
        configureStartupWindowIfNeeded()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    private func hideWindowTitles() {
        for window in NSApplication.shared.windows {
            window.titleVisibility = .hidden
        }
    }

    private func configureStartupWindowIfNeeded() {
        guard !didPlaceMainWindow else { return }
        guard let window = NSApplication.shared.windows.first(where: { $0.isVisible && !$0.isMiniaturized }) else {
            DispatchQueue.main.async { [weak self] in
                self?.configureStartupWindowIfNeeded()
            }
            return
        }

        let visibleFrame = (window.screen ?? NSScreen.main)?.visibleFrame ?? window.frame
        let margin = AppWindowMetrics.startupMargin
        let width = min(AppWindowMetrics.defaultWidth, max(AppWindowMetrics.minWidth, visibleFrame.width - margin * 2))
        let height = min(AppWindowMetrics.defaultHeight, max(AppWindowMetrics.minHeight, visibleFrame.height - margin * 2))
        let x = visibleFrame.maxX - width - margin
        let y = visibleFrame.maxY - height - margin
        let frame = NSRect(
            x: max(visibleFrame.minX + margin, x),
            y: max(visibleFrame.minY + margin, y),
            width: width,
            height: height
        )

        window.setFrame(frame, display: true)
        didPlaceMainWindow = true
    }
}
