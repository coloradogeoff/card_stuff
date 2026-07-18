import AppKit
import SwiftUI

private struct LetterTrackPrintJob {
    let sourceURL: URL
    let filename: String
    let spec: EnvelopeSpec
    let pageCount: Int
}

private struct EnvelopeInsertPrompt: Identifiable {
    let pageNumber: Int
    let pageCount: Int

    var id: Int { pageNumber }

    var title: String {
        switch pageNumber {
        case 2: return "Second PDF. Insert envelope"
        case 3: return "Third PDF. Insert envelope"
        case 4: return "Fourth PDF. Insert envelope"
        default: return "Page \(pageNumber) PDF. Insert envelope"
        }
    }

    var message: String {
        "Load an envelope, then choose Print Page \(pageNumber) of \(pageCount)."
    }
}

private enum LetterTrackAlert: Identifiable {
    case error(AlertItem)
    case insertEnvelope(EnvelopeInsertPrompt)

    var id: String {
        switch self {
        case .error(let item): return "error-\(item.id.uuidString)"
        case .insertEnvelope(let prompt): return "envelope-\(prompt.pageNumber)"
        }
    }
}

struct LetterTrackView: View {
    @State private var trackingNumber: String = ""
    @State private var trackingURL: String = ""
    @State private var message: String = ""

    // Label printing
    @State private var labels: [LabelFile] = []
    @State private var selectedURL: URL?
    @State private var selectedSpec: EnvelopeSpec = EnvelopeCatalog.spec6x9
    @State private var isPrinting = false
    @State private var statusText = "Looking for LetterTrack labels in Downloads…"
    @State private var activeAlert: LetterTrackAlert?
    @State private var activePrintJob: LetterTrackPrintJob?

    private let service = LetterTrackLabelService()

    private var selectedLabel: LabelFile? {
        labels.first { $0.url == selectedURL }
    }

    var body: some View {
        VStack(spacing: 0) {
            labelSection
                .padding(16)

            VStack(alignment: .leading, spacing: 12) {
                trackingInput
                trackingLinkSection
                messageSection
            }
            .padding(16)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .background(Color.green.opacity(0.08))
        }
        .task { refreshLabels() }
        .alert(item: $activeAlert) { alert in
            switch alert {
            case .error(let item):
                Alert(title: Text(item.title), message: Text(item.message), dismissButton: .default(Text("OK")))
            case .insertEnvelope(let prompt):
                Alert(
                    title: Text(prompt.title),
                    message: Text(prompt.message),
                    dismissButton: .default(Text("Print Page \(prompt.pageNumber)")) {
                        printPage(prompt.pageNumber)
                    }
                )
            }
        }
    }

    // MARK: - Label printing

    private var labelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            GroupBox("LetterTrack Labels in Downloads") {
                VStack(alignment: .leading, spacing: 8) {
                    if labels.isEmpty {
                        ContentUnavailableView(
                            "No LetterTrack Labels",
                            systemImage: "doc.text.magnifyingglass",
                            description: Text("Download a label from LetterTrackPro, then refresh.")
                        )
                        .frame(maxWidth: .infinity, minHeight: 100)
                    } else {
                        List(labels, selection: $selectedURL) { label in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(label.name)
                                    .lineLimit(1)
                                Text(label.modifiedAt, style: .relative)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .tag(label.url)
                        }
                        .frame(minHeight: 100)
                    }

                    HStack {
                        Button("Refresh", systemImage: "arrow.clockwise") {
                            refreshLabels()
                        }
                        Spacer()
                        Text("\(labels.count) PDF\(labels.count == 1 ? "" : "s") found")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.vertical, 4)
            }

            GroupBox("Print Label") {
                VStack(alignment: .leading, spacing: 8) {
                    Text(selectedLabel?.name ?? "No label selected")
                        .font(.headline)
                        .lineLimit(2)

                    HStack(spacing: 12) {
                        Picker("Envelope", selection: $selectedSpec) {
                            ForEach(EnvelopeCatalog.letterTrackSpecs) { spec in
                                Text(spec.label).tag(spec)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 260)

                        Button(action: printSelected) {
                            Label(isPrinting ? "Printing…" : "Print Label", systemImage: "printer.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(selectedLabel == nil || isPrinting)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 4)
            }

            Text(statusText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
        }
    }

    private func refreshLabels() {
        Task {
            let result = await Task.detached { Result { try service.findLabels() } }.value
            switch result {
            case .success(let found):
                labels = found
                if !found.contains(where: { $0.url == selectedURL }) {
                    selectedURL = found.first?.url
                }
                statusText = found.isEmpty
                    ? "No matching PDFs found in ~/Downloads."
                    : "Newest label selected."
            case .failure(let error):
                labels = []
                selectedURL = nil
                statusText = "Could not read Downloads."
                showAlert(title: "Refresh error", error: error)
            }
        }
    }

    private func printSelected() {
        guard let label = selectedLabel else { return }
        isPrinting = true
        statusText = "Preparing \(label.name)…"
        let spec = selectedSpec

        Task {
            let result = await Task.detached {
                Result { try service.pageCount(in: label.url) }
            }.value

            switch result {
            case .success(let pageCount):
                activePrintJob = LetterTrackPrintJob(
                    sourceURL: label.url,
                    filename: label.name,
                    spec: spec,
                    pageCount: pageCount
                )
                printPage(1)
            case .failure(let error):
                finishPrintWithError(error)
            }
        }
    }

    private func printPage(_ pageNumber: Int) {
        guard let job = activePrintJob else { return }
        statusText = "Printing page \(pageNumber) of \(job.pageCount) from \(job.filename)…"

        Task {
            let result = await Task.detached { () -> Result<Void, Error> in
                var outputURL: URL?
                do {
                    let rendered = try service.renderEnvelopePDF(
                        from: job.sourceURL,
                        pageNumber: pageNumber,
                        spec: job.spec
                    )
                    outputURL = rendered
                    try service.printPDF(at: rendered, spec: job.spec)
                    service.removeOutput(at: rendered)
                    return .success(())
                } catch {
                    if let outputURL { service.removeOutput(at: outputURL) }
                    return .failure(error)
                }
            }.value

            switch result {
            case .success:
                if pageNumber == job.pageCount {
                    archive(job)
                } else {
                    statusText = "Page \(pageNumber) printed. Insert an envelope for page \(pageNumber + 1)."
                    activeAlert = .insertEnvelope(EnvelopeInsertPrompt(
                        pageNumber: pageNumber + 1,
                        pageCount: job.pageCount
                    ))
                }
            case .failure(let error):
                finishPrintWithError(error)
            }
        }
    }

    private func archive(_ job: LetterTrackPrintJob) {
        statusText = "Archiving \(job.filename)…"

        Task {
            let result = await Task.detached {
                Result { try service.archive(job.sourceURL) }
            }.value

            switch result {
            case .success(let archived):
                isPrinting = false
                activePrintJob = nil
                statusText = "Printed and archived \(archived.lastPathComponent)."
                refreshLabels()
            case .failure(let error):
                finishPrintWithError(error)
            }
        }
    }

    private func finishPrintWithError(_ error: Error) {
        isPrinting = false
        activePrintJob = nil
        activeAlert = nil
        statusText = "Print failed."
        showAlert(title: "Print error", error: error)
    }

    private func showAlert(title: String, error: Error) {
        activeAlert = .error(AlertItem(title: title, message: error.localizedDescription))
    }

    // MARK: - Tracking number section (unchanged)

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
                Button(action: { copyToPasteboard(trackingURL) }) {
                    Image(systemName: "doc.on.clipboard")
                }
                .disabled(trackingURL.isEmpty)
                .help("Copy URL to clipboard")
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
