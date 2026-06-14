import AppKit
import SwiftUI

struct LetterTrackView: View {
    @State private var trackingNumber: String = ""
    @State private var trackingURL: String = ""
    @State private var message: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            quickLinks
            trackingInput
            trackingLinkSection
            messageSection
        }
        .padding(16)
    }

    private var quickLinks: some View {
        GroupBox("Quick Links") {
            HStack(spacing: 8) {
                Button("Awaiting Shipping") {
                    NSWorkspace.shared.open(LetterTrackLinks.awaitingShipping)
                }
                Button("LetterTrackPro") {
                    NSWorkspace.shared.open(LetterTrackLinks.lettertrackpro)
                }
                Spacer()
            }
            .padding(.vertical, 4)
        }
    }

    private var trackingInput: some View {
        GroupBox("Tracking Number") {
            HStack(spacing: 8) {
                TextField("Enter USPS tracking number…", text: $trackingNumber)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { generateMessage() }
                Button("Clear", action: clearAll)
            }
            .padding(.vertical, 4)
        }
    }

    private var trackingLinkSection: some View {
        GroupBox("Tracking Link") {
            HStack(spacing: 8) {
                Button("Generate Tracking URL") {
                    generateTrackingURL()
                }
                TextField("Tracking URL will appear here", text: $trackingURL)
                    .textFieldStyle(.roundedBorder)
                    .disabled(true)
            }
            .padding(.vertical, 4)
        }
    }

    private var messageSection: some View {
        GroupBox("Message") {
            VStack(alignment: .leading, spacing: 8) {
                TextEditor(text: $message)
                    .font(.body)
                    .frame(minHeight: 140)

                Button(action: generateMessage) {
                    Label("Generate Message", systemImage: "wand.and.stars")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: [.command])

                Button("Copy to Clipboard", action: copyMessage)
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Actions

    private func generateTrackingURL() {
        let trimmed = trackingNumber.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let url = LetterTrackLinks.trackingURL(for: trimmed) else {
            trackingURL = ""
            return
        }
        trackingURL = url.absoluteString
    }

    private func generateMessage() {
        let trimmed = trackingNumber.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let url = LetterTrackLinks.trackingURL(for: trimmed) else {
            message = "Please enter a tracking number."
            return
        }
        trackingURL = url.absoluteString
        message = MessageBuilder.make(tinyURL: trackingURL)
        copyToPasteboard(message)
    }

    private func clearAll() {
        trackingNumber = ""
        trackingURL = ""
        message = ""
    }

    private func copyMessage() {
        copyToPasteboard(message)
    }

    private func copyToPasteboard(_ string: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(string, forType: .string)
    }
}

#Preview {
    LetterTrackView()
}
