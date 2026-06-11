import AppKit
import SwiftUI

struct EbayLabelView: View {
    @State private var labels: [EbayLabelFile] = []
    @State private var selectedURL: URL?
    @State private var isPrinting = false
    @State private var statusText = "Looking for eBay labels in Downloads…"
    @State private var alertItem: EbayLabelAlertItem?

    private let service = EbayLabelService()

    private var selectedLabel: EbayLabelFile? {
        labels.first { $0.url == selectedURL }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            GroupBox("eBay Labels in Downloads") {
                VStack(alignment: .leading, spacing: 8) {
                    if labels.isEmpty {
                        ContentUnavailableView(
                            "No eBay Labels",
                            systemImage: "doc.text.magnifyingglass",
                            description: Text("Download an eBay label PDF, then refresh.")
                        )
                        .frame(maxWidth: .infinity, minHeight: 150)
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
                        .frame(minHeight: 170)
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

            GroupBox("Selected Label") {
                VStack(alignment: .leading, spacing: 8) {
                    Text(selectedLabel?.name ?? "No label selected")
                        .font(.headline)
                        .lineLimit(2)

                    HStack {
                        Button("Preview", systemImage: "eye") {
                            previewSelected()
                        }
                        .disabled(selectedLabel == nil || isPrinting)

                        Button(action: printSelected) {
                            Label(
                                isPrinting ? "Printing…" : "Print on 6x9 Envelope",
                                systemImage: "printer.fill"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(selectedLabel == nil || isPrinting)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 4)
            }

            Divider()

            Text(statusText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
        }
        .padding(16)
        .task {
            refreshLabels()
        }
        .alert(item: $alertItem) { item in
            Alert(
                title: Text(item.title),
                message: Text(item.message),
                dismissButton: .default(Text("OK"))
            )
        }
    }

    private func refreshLabels() {
        do {
            let found = try service.findLabels()
            labels = found

            if let selectedURL, found.contains(where: { $0.url == selectedURL }) {
                self.selectedURL = selectedURL
            } else {
                selectedURL = found.first?.url
            }

            statusText = found.isEmpty
                ? "No matching PDFs found in ~/Downloads."
                : "Newest label selected. Choose another PDF above if needed."
        } catch {
            labels = []
            selectedURL = nil
            statusText = "Could not read Downloads."
            showError(title: "Refresh error", error: error)
        }
    }

    private func previewSelected() {
        guard let label = selectedLabel else { return }
        do {
            let outputURL = try service.renderEnvelopePDF(from: label.url)
            NSWorkspace.shared.open(outputURL)
            statusText = "Opened 6x9 preview for \(label.name)."
        } catch {
            showError(title: "Preview error", error: error)
        }
    }

    private func printSelected() {
        guard let label = selectedLabel else { return }
        isPrinting = true
        statusText = "Preparing \(label.name)…"

        Task {
            let result = await Task.detached {
                () -> Result<URL, Error> in
                var outputURL: URL?
                do {
                    let rendered = try service.renderEnvelopePDF(from: label.url)
                    outputURL = rendered
                    try service.printPDF(at: rendered)
                    let archived = try service.archive(label.url)
                    service.removeOutput(at: rendered)
                    return .success(archived)
                } catch {
                    if let outputURL {
                        service.removeOutput(at: outputURL)
                    }
                    return .failure(error)
                }
            }.value

            await MainActor.run {
                isPrinting = false
                switch result {
                case .success(let archived):
                    statusText = "Printed and archived \(archived.lastPathComponent)."
                    refreshLabels()
                case .failure(let error):
                    statusText = "The label was not archived."
                    showError(title: "Print error", error: error)
                }
            }
        }
    }

    private func showError(title: String, error: Error) {
        alertItem = EbayLabelAlertItem(
            title: title,
            message: error.localizedDescription
        )
    }
}

private struct EbayLabelAlertItem: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

#Preview {
    EbayLabelView()
}
