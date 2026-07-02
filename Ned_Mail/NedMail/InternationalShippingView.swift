import AppKit
import SwiftUI

private let messageSingle = "Hello, and thank you for the business. Your card is on the way. I hope it reaches you safely and fast. Best Geoff"
private let messagePlural = "Hello, and thank you for the business. Your cards are on the way. I hope they reach you safely and fast. Best Geoff"
private let intlOutputDir = URL(fileURLWithPath: "/Volumes/Dutton 2TB/Sales/shipping", isDirectory: true)
private let intlProcessedDir = URL(fileURLWithPath: "/Volumes/Dutton 2TB/Sales/shipping/processed", isDirectory: true)
private let intlImageExtensions: Set<String> = ["heic", "jpg", "jpeg", "png", "webp", "tif", "tiff"]
private let intlOpenAIModel = "gpt-5"

struct InternationalShippingView: View {
    @State private var imageFiles: [URL] = []
    @State private var selectedURL: URL?
    @State private var thumbnailImage: NSImage?
    @State private var isPlural = false
    @State private var messageText: String = messageSingle
    @State private var detectedCountry: String?      // editable after identification
    @State private var detectedConfidence: Double?
    @State private var isIdentifying = false
    @State private var isBusy = false
    @State private var statusText = "Select an envelope photo, then Identify Country or Archive."
    @State private var alertItem: AlertItem?

    private var isWorking: Bool { isIdentifying || isBusy }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            imageSection
            messageSection

            Button(action: archiveImage) {
                Label(isBusy ? "Archiving…" : "Archive Image", systemImage: "archivebox.fill")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
            }
            .controlSize(.large)
            .buttonStyle(.borderedProminent)
            .tint(.teal)
            .keyboardShortcut(.return, modifiers: [.command])
            .disabled(selectedURL == nil || isWorking)

            Divider()

            Text(statusText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
        }
        .padding(16)
        .task { refreshImages() }
        .task(id: selectedURL) { await loadThumbnail() }
        .onChange(of: selectedURL) {
            detectedCountry = nil
            detectedConfidence = nil
        }
        .alert(item: $alertItem) { item in
            Alert(
                title: Text(item.title),
                message: Text(item.message),
                dismissButton: .default(Text("OK"))
            )
        }
    }

    // MARK: - Image section

    private var imageSection: some View {
        GroupBox("Envelope Photo") {
            VStack(alignment: .leading, spacing: 8) {
                if imageFiles.isEmpty {
                    ContentUnavailableView(
                        "No Images",
                        systemImage: "photo.badge.magnifyingglass",
                        description: Text("Add an envelope photo to Downloads, then refresh.")
                    )
                    .frame(maxWidth: .infinity, minHeight: 90, maxHeight: 90)
                } else {
                    List(imageFiles, id: \.self, selection: $selectedURL) { url in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(url.lastPathComponent)
                                .lineLimit(1)
                            if let mod = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate {
                                Text(mod, style: .relative)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .tag(url)
                    }
                    .frame(height: 90)
                }

                if let img = thumbnailImage {
                    Image(nsImage: img)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: 120)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }

                // Identify button
                Button(action: identifyCountry) {
                    Label(isIdentifying ? "Identifying…" : "Identify Country", systemImage: "globe.americas")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.indigo)
                .disabled(selectedURL == nil || isWorking)

                // Editable country result
                if detectedCountry != nil {
                    HStack(spacing: 8) {
                        Image(systemName: "mappin.circle.fill")
                            .foregroundStyle(.secondary)
                        TextField("Country", text: Binding(
                            get: { detectedCountry ?? "" },
                            set: { detectedCountry = $0.isEmpty ? nil : $0 }
                        ))
                        .textFieldStyle(.roundedBorder)
                        if let conf = detectedConfidence {
                            Text(String(format: "%.0f%%", conf * 100))
                                .font(.caption)
                                .foregroundStyle(conf > 0.75 ? .green : .orange)
                                .monospacedDigit()
                        }
                    }
                }

                HStack {
                    Button("Refresh", systemImage: "arrow.clockwise") { refreshImages() }
                    Spacer()
                    Text("\(imageFiles.count) image\(imageFiles.count == 1 ? "" : "s") found")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Message section

    private var messageSection: some View {
        GroupBox("Message") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("", selection: $isPlural) {
                    Text("Single card").tag(false)
                    Text("Multiple cards").tag(true)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .onChange(of: isPlural) { _, plural in
                    messageText = plural ? messagePlural : messageSingle
                }

                TextEditor(text: $messageText)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .frame(maxWidth: .infinity, minHeight: 140)
                    .padding(6)
                    .background(Color(nsColor: .textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.primary.opacity(0.12)))

                HStack(spacing: 8) {
                    Button("Copy", systemImage: "doc.on.doc") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(messageText, forType: .string)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    Button("Reset", systemImage: "arrow.counterclockwise") {
                        messageText = isPlural ? messagePlural : messageSingle
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Actions

    private func loadThumbnail() async {
        guard let url = selectedURL else {
            thumbnailImage = nil
            return
        }
        let img = await Task.detached(priority: .userInitiated) {
            NSImage(contentsOf: url)
        }.value
        guard !Task.isCancelled else { return }
        thumbnailImage = img
    }

    private func refreshImages() {
        let downloadsURL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Downloads")
        guard let items = try? FileManager.default.contentsOfDirectory(
            at: downloadsURL,
            includingPropertiesForKeys: [.contentModificationDateKey, .creationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            imageFiles = []
            statusText = "Could not read Downloads folder."
            return
        }

        let found = items
            .filter { intlImageExtensions.contains($0.pathExtension.lowercased()) }
            .sorted {
                let a = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let b = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return a > b
            }

        imageFiles = found

        if let current = selectedURL, found.contains(current) {
            // keep selection
        } else {
            selectedURL = found.first
        }

        statusText = found.isEmpty
            ? "No image files found in Downloads."
            : "Newest image selected. Choose another above if needed."
    }

    private func identifyCountry() {
        guard let imageURL = selectedURL else { return }
        isIdentifying = true
        detectedCountry = nil
        detectedConfidence = nil
        statusText = "Converting image…"

        Task {
            do {
                let tmpJPEG = try await Task.detached(priority: .userInitiated) {
                    let tmp = FileManager.default.temporaryDirectory
                        .appendingPathComponent(UUID().uuidString + "_intl.jpg")
                    try intlConvertToJPEG(from: imageURL, to: tmp)
                    return tmp
                }.value

                statusText = "Detecting destination country…"
                let detection = try await intlDetectCountry(jpegURL: tmpJPEG)
                try? FileManager.default.removeItem(at: tmpJPEG)

                detectedCountry = detection.country
                detectedConfidence = detection.confidence
                let pct = String(format: "%.0f%%", detection.confidence * 100)
                statusText = "Detected: \(detection.country) (\(pct)). Edit if needed, then Archive."
            } catch {
                statusText = "Error: \(error.localizedDescription)"
                alertItem = AlertItem(title: "Detection failed", message: error.localizedDescription)
            }
            isIdentifying = false
        }
    }

    private func archiveImage() {
        guard let imageURL = selectedURL else { return }
        let msg = messageText
        let priorCountry = detectedCountry      // use cached result if available
        let priorConfidence = detectedConfidence
        isBusy = true
        statusText = "Converting image…"

        Task {
            do {
                let tmpJPEG = try await Task.detached(priority: .userInitiated) {
                    let tmp = FileManager.default.temporaryDirectory
                        .appendingPathComponent(UUID().uuidString + "_intl.jpg")
                    try intlConvertToJPEG(from: imageURL, to: tmp)
                    return tmp
                }.value

                // Skip AI call if country was already identified
                let detection: IntlDetection
                if let country = priorCountry {
                    detection = IntlDetection(country: country, confidence: priorConfidence ?? 1.0)
                } else {
                    statusText = "Detecting destination country…"
                    let result = try await intlDetectCountry(jpegURL: tmpJPEG)
                    detectedCountry = result.country
                    detectedConfidence = result.confidence
                    detection = result
                }

                let country = intlSafeCountry(detection.country)
                let dateStr = intlFileDateString(for: imageURL)
                let finalName = "\(dateStr)_\(country).jpg"

                try await Task.detached {
                    let fm = FileManager.default
                    try fm.createDirectory(at: intlOutputDir, withIntermediateDirectories: true)
                    let dest = intlUniquePath(in: intlOutputDir, filename: finalName)
                    try fm.moveItem(at: tmpJPEG, to: dest)
                }.value

                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(msg, forType: .string)

                var summary = "Saved \(finalName). Message copied to clipboard."

                if detection.confidence > 0.75 {
                    if imageURL.pathExtension.lowercased() == "heic" {
                        do {
                            try await Task.detached {
                                let fm = FileManager.default
                                try fm.createDirectory(at: intlProcessedDir, withIntermediateDirectories: true)
                                let dest = intlUniquePath(in: intlProcessedDir, filename: imageURL.lastPathComponent)
                                try fm.moveItem(at: imageURL, to: dest)
                            }.value
                            summary += " Original moved to processed/."
                        } catch {
                            summary += " (Could not move original: \(error.localizedDescription))"
                        }
                    } else {
                        summary += " Non-HEIC original left in Downloads."
                    }
                } else {
                    let pct = String(format: "%.0f%%", detection.confidence * 100)
                    summary += " Low confidence (\(pct)) — original left in Downloads."
                }

                statusText = summary
                refreshImages()
            } catch {
                statusText = "Error: \(error.localizedDescription)"
                alertItem = AlertItem(title: "Archive failed", message: error.localizedDescription)
            }
            isBusy = false
        }
    }
}

// MARK: - File-private helpers

private struct IntlDetection {
    let country: String
    let confidence: Double
}

private func intlConvertToJPEG(from source: URL, to destination: URL) throws {
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: "/usr/bin/sips")
    proc.arguments = ["-s", "format", "jpeg", source.path, "--out", destination.path]
    let errPipe = Pipe()
    proc.standardError = errPipe
    proc.standardOutput = Pipe()
    try proc.run()
    proc.waitUntilExit()
    guard proc.terminationStatus == 0 else {
        let msg = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "unknown sips error"
        throw NSError(domain: "InternationalShipping", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Image conversion failed: \(msg)"])
    }
}

private func intlDetectCountry(jpegURL: URL) async throws -> IntlDetection {
    guard let key = intlLoadAPIKey() else {
        throw NSError(domain: "InternationalShipping", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "No OpenAI API key found. Add .openai-api-key.txt to your home folder."])
    }

    let b64 = try Data(contentsOf: jpegURL).base64EncodedString()

    var body: [String: Any] = [
        "model": intlOpenAIModel,
        "max_completion_tokens": 1000,
        "messages": [
            [
                "role": "system",
                "content": "You are reading a photo of a mailed envelope. Identify the DESTINATION COUNTRY from the destination address. If multiple addresses are present, choose the recipient/destination address. Return ONLY strict JSON: {\"country\":\"...\",\"confidence\":0.0,\"notes\":\"short\"}. Use country=\"UNKNOWN\" if not determinable.",
            ],
            [
                "role": "user",
                "content": [
                    ["type": "text", "text": "Extract the destination country from this envelope photo."],
                    ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64,\(b64)", "detail": "high"]],
                ],
            ],
        ],
    ]
    if intlOpenAIModel.lowercased().hasPrefix("gpt-5") {
        body["reasoning_effort"] = "low"
    }

    var req = URLRequest(url: URL(string: "https://api.openai.com/v1/chat/completions")!)
    req.httpMethod = "POST"
    req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    req.httpBody = try JSONSerialization.data(withJSONObject: body)

    let (data, response) = try await URLSession.shared.data(for: req)
    let code = (response as? HTTPURLResponse)?.statusCode ?? 0
    guard code == 200 else {
        let errMsg: String
        if let outer = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let err = outer["error"] as? [String: Any],
           let msg = err["message"] as? String {
            errMsg = msg
        } else {
            errMsg = String(data: data, encoding: .utf8).flatMap { String($0.prefix(200)) } ?? "unknown"
        }
        throw NSError(domain: "InternationalShipping", code: code,
                      userInfo: [NSLocalizedDescriptionKey: "OpenAI error (\(code)): \(errMsg)"])
    }

    guard let outer = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let choices = outer["choices"] as? [[String: Any]],
          let msg = choices.first?["message"] as? [String: Any],
          let content = msg["content"] as? String else {
        throw NSError(domain: "InternationalShipping", code: 3,
                      userInfo: [NSLocalizedDescriptionKey: "Unexpected response format from OpenAI."])
    }

    func parseJSON(_ s: String) -> IntlDetection? {
        guard let d = s.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return nil }
        return IntlDetection(
            country: dict["country"] as? String ?? "UNKNOWN",
            confidence: dict["confidence"] as? Double ?? 0.0
        )
    }

    let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
    if let result = parseJSON(trimmed) { return result }
    if let range = trimmed.range(of: #"\{[^{}]*\}"#, options: .regularExpression),
       let result = parseJSON(String(trimmed[range])) { return result }

    return IntlDetection(country: "UNKNOWN", confidence: 0.0)
}

private func intlLoadAPIKey() -> String? {
    if let key = ProcessInfo.processInfo.environment["OPENAI_API_KEY"], !key.isEmpty { return key }
    for path in ["code/card_stuff/.openai-api-key.txt", ".openai-api-key.txt"] {
        let url = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(path)
        if let text = try? String(contentsOf: url, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
            return text
        }
    }
    return nil
}

private func intlSafeCountry(_ country: String) -> String {
    var s = country.trimmingCharacters(in: .whitespaces).uppercased()
        .replacingOccurrences(of: " ", with: "_")
    s = s.filter { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" }
    return s.isEmpty ? "UNKNOWN_COUNTRY" : s
}

private func intlFileDateString(for url: URL) -> String {
    let vals = try? url.resourceValues(forKeys: [.creationDateKey, .contentModificationDateKey])
    let date = vals?.creationDate ?? vals?.contentModificationDate ?? Date()
    let fmt = DateFormatter()
    fmt.dateFormat = "yyyy-MM-dd"
    return fmt.string(from: date)
}

private func intlUniquePath(in dir: URL, filename: String) -> URL {
    let candidate = dir.appendingPathComponent(filename)
    guard FileManager.default.fileExists(atPath: candidate.path) else { return candidate }
    let stem = URL(fileURLWithPath: filename).deletingPathExtension().lastPathComponent
    let ext = URL(fileURLWithPath: filename).pathExtension
    var i = 2
    while true {
        let alt = dir.appendingPathComponent(ext.isEmpty ? "\(stem)_\(i)" : "\(stem)_\(i).\(ext)")
        if !FileManager.default.fileExists(atPath: alt.path) { return alt }
        i += 1
    }
}

#Preview {
    InternationalShippingView()
}
