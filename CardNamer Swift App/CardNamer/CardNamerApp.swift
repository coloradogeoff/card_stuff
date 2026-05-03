import SwiftUI

extension Notification.Name {
    static let goToDirectory = Notification.Name("goToDirectory")
}

@main
struct CardNamerApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 1200, minHeight: 800)
        }
        .windowStyle(.titleBar)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Go") {
                Button("Incoming Cards") {
                    if let dir = SettingsStore.shared.quickDirectories.first {
                        NotificationCenter.default.post(name: .goToDirectory, object: dir)
                    }
                }
                .keyboardShortcut("i", modifiers: .command)
            }
        }
    }
}
