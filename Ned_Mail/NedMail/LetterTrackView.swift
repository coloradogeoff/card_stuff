import AppKit
import SwiftUI

struct LetterTrackView: View {
    @State private var trackingNumber: String = ""
    @State private var tinyURL: String = ""
    @State private var message: String = ""
    @State private var isWorking: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            quickLinks
            trackingInput
            tinyLinkSection
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
                    .onSubmit {
                        Task { await generateMessage() }
                    }
                Button("Clear", action: clearAll)
            }
            .padding(.vertical, 4)
        }
    }

    private var tinyLinkSection: some View {
        GroupBox("Short Link") {
            HStack(spacing: 8) {
                Button("Generate TinyURL") {
                    Task { await generateTinyURL() }
                }
                .disabled(isWorking)

                TextField("Shortened URL will appear here", text: $tinyURL)
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

                Button(action: { Task { await generateMessage() } }) {
                    Label(
                        isWorking ? "Working…" : "Generate Message",
                        systemImage: "wand.and.stars"
                    )
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(isWorking)

                Button("Copy to Clipboard", action: copyMessage)
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Actions

    private func generateTinyURL() async {
        let trimmed = trackingNumber.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            tinyURL = ""
            return
        }
        guard let url = LetterTrackLinks.trackingURL(for: trimmed) else { return }

        isWorking = true
        defer { isWorking = false }

        do {
            tinyURL = try await TinyURLService.shorten(url)
        } catch {
            tinyURL = "Error: \(error.localizedDescription)"
        }
    }

    private func generateMessage() async {
        let trimmed = trackingNumber.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            message = "Please enter a tracking number."
            return
        }
        guard let url = LetterTrackLinks.trackingURL(for: trimmed) else { return }

        isWorking = true
        defer { isWorking = false }

        do {
            let short = try await TinyURLService.shorten(url)
            tinyURL = short
            let text = MessageBuilder.make(tinyURL: short)
            message = text
            copyToPasteboard(text)
        } catch {
            message = "Error generating TinyURL: \(error.localizedDescription)"
        }
    }

    private func clearAll() {
        trackingNumber = ""
        tinyURL = ""
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
