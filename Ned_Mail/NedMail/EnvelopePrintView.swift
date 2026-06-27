import SwiftUI

struct EnvelopePrintView: View {
    @State private var returnText: String = DefaultAddresses.returnLines.joined(separator: "\n")
    @State private var toText: String = DefaultAddresses.toLines.joined(separator: "\n")
    @State private var selectedLabel: String = "5x7"
    @State private var removePDFAfterPrint: Bool = true
    @State private var alertItem: AlertItem?
    @State private var isPrinting: Bool = false

    private let printer = EnvelopePrinter(settings: EnvelopeCatalog.printSettings)

    private var selectedSpec: EnvelopeSpec {
        EnvelopeCatalog.specs.first(where: { $0.label == selectedLabel })
            ?? EnvelopeCatalog.specs[0]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            sizeSection
            addressBox(title: "Return Address", text: $returnText, height: 100)
            addressBox(title: "Send To", text: $toText, height: 130)

            Button(action: handlePrint) {
                Label(isPrinting ? "Printing…" : "Print Envelope", systemImage: "printer.fill")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
            }
            .controlSize(.large)
            .buttonStyle(.borderedProminent)
            .keyboardShortcut(.return, modifiers: [.command])
            .disabled(isPrinting)

            Divider()

            Text(infoText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
        }
        .padding(16)
        .alert(item: $alertItem) { item in
            Alert(
                title: Text(item.title),
                message: Text(item.message),
                dismissButton: .default(Text("OK"))
            )
        }
    }

    private var sizeSection: some View {
        GroupBox("Envelope Size") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("", selection: $selectedLabel) {
                    ForEach(EnvelopeCatalog.specs) { spec in
                        Text(spec.label).tag(spec.label)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(maxWidth: 220)

                HStack {
                    Toggle("Remove PDF after print", isOn: $removePDFAfterPrint)
                        .toggleStyle(.checkbox)

                    Spacer()

                    Button("Reset Addresses", systemImage: "arrow.counterclockwise") {
                        resetAddresses()
                    }
                    .disabled(isPrinting)
                }
            }
            .padding(.vertical, 4)
        }
    }

    private func addressBox(title: String, text: Binding<String>, height: CGFloat) -> some View {
        GroupBox(title) {
            TextEditor(text: text)
                .font(.system(.body, design: .monospaced))
                .frame(height: height)
                .padding(.vertical, 2)
        }
    }

    private var infoText: String {
        let spec = selectedSpec
        let settings = EnvelopeCatalog.printSettings
        return "Size: \(spec.label) | Printer: \(settings.printerName) | Media: \(spec.media) | Output: \(settings.outputFilename)"
    }

    private func resetAddresses() {
        returnText = DefaultAddresses.returnLines.joined(separator: "\n")
        toText = DefaultAddresses.toLines.joined(separator: "\n")
    }

    private func handlePrint() {
        let returnLines = cleanedLines(returnText)
        let toLines = cleanedLines(toText)

        if returnLines.isEmpty {
            alertItem = AlertItem(title: "Missing return address", message: "Enter a return address.")
            return
        }
        if toLines.isEmpty {
            alertItem = AlertItem(title: "Missing destination address", message: "Enter a destination address.")
            return
        }

        let spec = selectedSpec
        let shouldRemove = removePDFAfterPrint
        isPrinting = true

        Task {
            let result = await Task.detached { () -> Result<Void, Error> in
                do {
                    let url = try printer.renderPDF(spec: spec, returnLines: returnLines, toLines: toLines)
                    try printer.printPDF(at: url, spec: spec)
                    if shouldRemove {
                        try printer.removePDF(at: url)
                    }
                    return .success(())
                } catch {
                    return .failure(error)
                }
            }.value

            await MainActor.run {
                isPrinting = false
                if case .failure(let error) = result {
                    alertItem = AlertItem(
                        title: "Print error",
                        message: error.localizedDescription
                    )
                }
            }
        }
    }

    private func cleanedLines(_ text: String) -> [String] {
        text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }
}

#Preview {
    EnvelopePrintView()
}
