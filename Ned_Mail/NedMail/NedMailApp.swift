import AppKit
import SwiftUI

@main
struct NedMailApp: App {
    @NSApplicationDelegateAdaptor(NedMailAppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            RootView()
                .frame(minWidth: 320, minHeight: 600)
        }
        .defaultSize(width: 320, height: 640)
        .windowStyle(.titleBar)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

struct RootView: View {
    var body: some View {
        TabView {
            LetterTrackView()
                .tabItem {
                    Label("Letter Track", systemImage: "shippingbox")
                }

            EnvelopePrintView()
                .tabItem {
                    Label("Envelope Print", systemImage: "envelope")
                }
        }
    }
}

final class NedMailAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

#Preview {
    RootView()
}
