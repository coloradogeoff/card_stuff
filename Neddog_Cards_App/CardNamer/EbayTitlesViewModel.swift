import Foundation
import AppKit
import Observation

struct EbayTitleResult: Identifiable {
    let id = UUID()
    let frontName: String
    let title: String
}

@Observable
final class EbayTitlesViewModel {

    var directoryPath: String = SettingsStore.shared.incomingDirectory.path
    var pairs: [CardPair] = []
    var checkedIDs: Set<CardPair.ID> = []

    var category: EbayCategory = .sportsCards
    var setOverride: String = ""
    var varietyOverride: String = ""

    var results: [EbayTitleResult] = []
    var progress: Double = 0
    var isBusy: Bool = false
    var logText: String = ""

    var showingBack: Bool = false
    var selectedPairID: CardPair.ID? {
        didSet { showingBack = false }
    }

    private let watcher = DirectoryWatcher()
    private var debounceTask: Task<Void, Never>?

    init() {
        startWatching()
        refreshImages()
        NotificationCenter.default.addObserver(forName: .goToDirectory, object: nil, queue: .main) { [self] note in
            if let dir = note.object as? QuickDirectory {
                switchTo(dir)
            }
        }
        NotificationCenter.default.addObserver(forName: .refreshImages, object: nil, queue: .main) { [self] _ in
            refreshImages()
        }
    }

    // MARK: - Computed

    var currentDirectory: URL { URL(fileURLWithPath: directoryPath).standardized }
    var selectedPair: CardPair? { pairs.first { $0.id == selectedPairID } }
    var previewURL: URL? {
        guard let pair = selectedPair else { return nil }
        return showingBack ? pair.back : pair.front
    }
    var checkedPairs: [CardPair] { pairs.filter { checkedIDs.contains($0.id) } }
    var titlesCSVURL: URL { currentDirectory.appendingPathComponent("description.csv") }
    var hasSavedTitles: Bool { FileManager.default.fileExists(atPath: titlesCSVURL.path) }

    // MARK: - Directory

    func switchTo(_ dir: QuickDirectory) {
        directoryPath = dir.path
        refreshImages()
    }

    func chooseDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url {
            directoryPath = url.path
            refreshImages()
        }
    }

    func refreshImages(silent: Bool = false) {
        let dir = currentDirectory
        guard FileManager.default.fileExists(atPath: dir.path) else {
            if !silent { log("Directory does not exist: \(dir.path)") }
            pairs = []; return
        }

        let imageExts: Set<String> = ["jpg","jpeg","png","bmp","tif","tiff","webp"]
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey]))?
            .filter { imageExts.contains($0.pathExtension.lowercased()) }
            .sorted { naturalSortLess($0.lastPathComponent, $1.lastPathComponent) } ?? []

        let newPairs = buildPairs(from: files)
        let prevChecked = checkedIDs
        pairs = newPairs

        if prevChecked.isEmpty {
            // First load: check pairs modified within the last hour
            checkedIDs = Set(pairs.filter { isRecentPair($0) }.map(\.id))
        } else {
            // Re-check pairs that were checked before (by front filename)
            let prevFronts = prevChecked.compactMap { id in pairs.first { $0.id == id }?.front.lastPathComponent }
            checkedIDs = Set(pairs.filter { prevFronts.contains($0.front.lastPathComponent) }.map(\.id))
        }

        if selectedPairID == nil || !pairs.contains(where: { $0.id == selectedPairID }) {
            selectedPairID = pairs.first?.id
        }

        startWatching()
        if !silent { log("Loaded \(newPairs.count) pair(s) from \(dir.lastPathComponent).") }
    }

    func togglePreviewSide() {
        guard selectedPair != nil else { return }
        showingBack.toggle()
    }

    // MARK: - Selection

    func selectAll() { checkedIDs = Set(pairs.map(\.id)) }
    func selectNone() { checkedIDs = [] }
    func toggleCheck(_ pair: CardPair) {
        if checkedIDs.contains(pair.id) { checkedIDs.remove(pair.id) }
        else { checkedIDs.insert(pair.id) }
    }

    // MARK: - Generate

    func generateTitles() {
        guard !isBusy, !checkedPairs.isEmpty else { return }
        isBusy = true
        progress = 0
        results = []
        log("Starting \(category.rawValue) — \(checkedPairs.count) pair(s)…")

        let targets = checkedPairs
        let cat = category
        let setOvr = setOverride.trimmingCharacters(in: .whitespaces)
        let varOvr = varietyOverride.trimmingCharacters(in: .whitespaces)
        let total = Double(targets.count)

        // Cap parallel OpenAI calls. 13+ in flight at once tended to stall on
        // rate limits, URLSession connection limits, and Vision OCR contention.
        let maxConcurrent = 4

        let runOne: @Sendable (Int, CardPair) async -> (Int, EbayTitleResult) = { index, pair in
            let frontName = pair.front.lastPathComponent
            do {
                let title = try await OpenAIService.generateTitle(
                    frontURL: pair.front,
                    backURL: pair.back,
                    category: cat,
                    setOverride: setOvr.isEmpty ? nil : setOvr,
                    varietyOverride: varOvr.isEmpty ? nil : varOvr
                )
                return (index, EbayTitleResult(frontName: frontName, title: title))
            } catch {
                return (index, EbayTitleResult(frontName: frontName, title: "ERROR: \(error.localizedDescription)"))
            }
        }

        Task {
            var completed = 0
            var indexed: [(Int, EbayTitleResult)] = []

            await withTaskGroup(of: (Int, EbayTitleResult).self) { group in
                var iterator = Array(targets.enumerated()).makeIterator()

                // Prime up to maxConcurrent tasks
                for _ in 0..<min(maxConcurrent, targets.count) {
                    guard let (index, pair) = iterator.next() else { break }
                    let frontName = pair.front.lastPathComponent
                    await MainActor.run { log("→ \(frontName)") }
                    group.addTask { await runOne(index, pair) }
                }

                // Drain: as each completes, log it and launch the next one
                while let (idx, r) = await group.next() {
                    indexed.append((idx, r))
                    completed += 1
                    let p = Double(completed) / total
                    await MainActor.run {
                        progress = p
                        log("[\(completed)/\(Int(total))] \(r.frontName): \(r.title)")
                    }
                    if let (index, pair) = iterator.next() {
                        let frontName = pair.front.lastPathComponent
                        await MainActor.run { log("→ \(frontName)") }
                        group.addTask { await runOne(index, pair) }
                    }
                }
            }

            let sorted = indexed.sorted { $0.0 > $1.0 }.map(\.1)
            await MainActor.run {
                results = sorted
                isBusy = false
                progress = 1.0
                saveCSV(sorted)
                NotificationCenter.default.post(name: .showEbayResultsWindow, object: nil)
            }
        }
    }

    func displaySavedTitles() {
        guard hasSavedTitles else { return }
        do {
            let loaded = try loadCSVResults()
            results = loaded
            log("Loaded \(loaded.count) title(s) from description.csv")
            NotificationCenter.default.post(name: .showEbayResultsWindow, object: nil)
        } catch {
            log("Could not load description.csv: \(error.localizedDescription)")
        }
    }

    private func saveCSV(_ rows: [EbayTitleResult]) {
        var csv = "\"front\",\"title\"\n"
        for row in rows {
            let escapedFront = row.frontName.replacingOccurrences(of: "\"", with: "\"\"")
            let escapedTitle = row.title.replacingOccurrences(of: "\"", with: "\"\"")
            csv += "\"\(escapedFront)\",\"\(escapedTitle)\"\n"
        }
        do {
            try csv.write(to: titlesCSVURL, atomically: true, encoding: .utf8)
            log("Saved \(rows.count) titles to description.csv")
        } catch {
            log("Could not save CSV: \(error.localizedDescription)")
        }
    }

    private func loadCSVResults() throws -> [EbayTitleResult] {
        let csv = try String(contentsOf: titlesCSVURL, encoding: .utf8)
        let rows = parseCSVRows(csv)
        guard !rows.isEmpty else { return [] }

        let dataRows = rows.first == ["front", "title"] ? Array(rows.dropFirst()) : rows
        return dataRows.compactMap { columns in
            guard columns.count >= 2 else { return nil }
            return EbayTitleResult(frontName: columns[0], title: columns[1])
        }
    }

    private func parseCSVRows(_ csv: String) -> [[String]] {
        var rows: [[String]] = []
        var currentRow: [String] = []
        var currentField = ""
        var isQuoted = false

        let characters = Array(csv)
        var index = 0

        while index < characters.count {
            let char = characters[index]

            if isQuoted {
                if char == "\"" {
                    if index + 1 < characters.count, characters[index + 1] == "\"" {
                        currentField.append("\"")
                        index += 1
                    } else {
                        isQuoted = false
                    }
                } else {
                    currentField.append(char)
                }
            } else {
                switch char {
                case "\"":
                    isQuoted = true
                case ",":
                    currentRow.append(currentField)
                    currentField = ""
                case "\n":
                    currentRow.append(currentField)
                    rows.append(currentRow)
                    currentRow = []
                    currentField = ""
                case "\r":
                    break
                default:
                    currentField.append(char)
                }
            }

            index += 1
        }

        if !currentField.isEmpty || !currentRow.isEmpty {
            currentRow.append(currentField)
            rows.append(currentRow)
        }

        return rows
    }

    // MARK: - Log

    func log(_ message: String) {
        let f = DateFormatter(); f.dateFormat = "HH:mm:ss"
        let line = "[\(f.string(from: Date()))] \(message)"
        logText = logText.isEmpty ? line : logText + "\n" + line
    }

    // MARK: - Watcher / pair building

    private func startWatching() {
        watcher.watch(url: currentDirectory)
        watcher.onChange = { [weak self] in
            self?.debounceTask?.cancel()
            self?.debounceTask = Task { @MainActor in
                try? await Task.sleep(nanoseconds: 400_000_000)
                guard !Task.isCancelled else { return }
                self?.refreshImages(silent: true)
            }
        }
    }

    private func isRecentPair(_ pair: CardPair, maxAgeSeconds: TimeInterval = 3600) -> Bool {
        let mtime = { (url: URL) -> Date in
            (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        }
        let latest = max(mtime(pair.front), mtime(pair.back))
        return latest.timeIntervalSinceNow >= -maxAgeSeconds
    }

    private func buildPairs(from files: [URL]) -> [CardPair] {
        let isBack = { (u: URL) in u.deletingPathExtension().lastPathComponent.lowercased().hasSuffix("_b") }
        var fronts: [String: [URL]] = [:]
        for f in files where !isBack(f) {
            fronts[f.deletingPathExtension().lastPathComponent.lowercased(), default: []].append(f)
        }
        var used = Set<URL>(); var pairs: [CardPair] = []
        for back in files.filter(isBack) {
            let stem = String(back.deletingPathExtension().lastPathComponent.lowercased().dropLast(2))
            if let front = fronts[stem]?.first(where: { !used.contains($0) }) {
                pairs.append(CardPair(front: front, back: back))
                used.insert(front); used.insert(back)
            }
        }
        let remaining = files.filter { !used.contains($0) }
        for i in stride(from: 0, to: remaining.count - 1, by: 2) {
            var a = remaining[i], b = remaining[i+1]
            if isBack(a) && !isBack(b) { swap(&a, &b) }
            pairs.append(CardPair(front: a, back: b))
        }
        return pairs.sorted {
            let m0 = (try? $0.front.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let m1 = (try? $1.front.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return m0 > m1
        }
    }

    private func naturalSortLess(_ a: String, _ b: String) -> Bool {
        let re = try? NSRegularExpression(pattern: #"(\d+)|(\D+)"#)
        func tokens(_ s: String) -> [(Int?, String)] {
            (re?.matches(in: s, range: NSRange(s.startIndex..., in: s)) ?? []).compactMap { m in
                guard let r = Range(m.range, in: s) else { return nil }
                let t = String(s[r]); return (Int(t), t)
            }
        }
        let ta = tokens(a), tb = tokens(b)
        for i in 0..<min(ta.count, tb.count) {
            let (ai, as_) = ta[i]; let (bi, bs) = tb[i]
            if let ai, let bi { if ai != bi { return ai < bi } }
            else if as_ != bs { return as_ < bs }
        }
        return ta.count < tb.count
    }
}
