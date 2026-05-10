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
    var childDirectories: [URL] = []
    var pairs: [CardPair] = []
    var visiblePairs: [CardPair] = []
    var filterText: String = "" {
        didSet {
            updateVisiblePairs()
            syncSelectionWithVisiblePairs()
        }
    }
    var sortOrder: CardPairSortOrder = .descending {
        didSet {
            updateVisiblePairs()
            syncSelectionWithVisiblePairs()
        }
    }
    var selectedTraitFilters: Set<CardTrait> = [] {
        didSet {
            updateVisiblePairs()
            syncSelectionWithVisiblePairs()
        }
    }
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
        didSet {
            showingBack = false
            updateSelectedTraits()
        }
    }
    var selectedTraits: CardTraits = CardTraits()
    var previewRevision: Int = 0

    private let watcher = DirectoryWatcher()
    private let metadataStore = CardMetadataStore(directoryURL: SettingsStore.shared.incomingDirectory)
    private var debounceTask: Task<Void, Never>?
    private var refreshTask: Task<Void, Never>?
    private var refreshGeneration = 0

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
    var parentDirectory: URL? {
        let parent = currentDirectory.deletingLastPathComponent().standardized
        return parent == currentDirectory ? nil : parent
    }
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

    func navigateToParentDirectory() {
        guard let parentDirectory else { return }
        navigateToDirectory(parentDirectory)
    }

    func navigateToDirectory(_ directory: URL) {
        directoryPath = directory.standardized.path
        checkedIDs = []
        selectedPairID = nil
        showingBack = false
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
        refreshTask?.cancel()
        refreshGeneration += 1
        let generation = refreshGeneration
        let dir = currentDirectory
        guard FileManager.default.fileExists(atPath: dir.path) else {
            if !silent { log("Directory does not exist: \(dir.path)") }
            childDirectories = []
            visiblePairs = []
            pairs = []; return
        }
        metadataStore.load(directoryURL: dir)
        startWatching()

        if let cachedIndex = CardDirectoryIndexStore.cachedIndex(for: dir) {
            applyDirectoryIndex(cachedIndex, pruneMetadata: false, updateRecentChecks: false)
            if !silent {
                log("Loaded \(cachedIndex.pairs.count) cached pair(s); verifying folder...")
            }
        } else if !silent {
            log("Scanning \(dir.lastPathComponent)...")
        }

        refreshTask = Task {
            let index = await Task.detached(priority: .userInitiated) {
                let scannedIndex = CardDirectoryIndexStore.scanDirectory(dir)
                CardDirectoryIndexStore.saveCacheIfNeeded(for: scannedIndex)
                return scannedIndex
            }.value

            await MainActor.run {
                guard !Task.isCancelled, self.refreshGeneration == generation, self.currentDirectory == dir else { return }
                let previousPairIDs = self.pairs.map(\.id)
                self.applyDirectoryIndex(index, pruneMetadata: true, updateRecentChecks: true)
                if !silent {
                    self.log("Loaded \(index.pairs.count) pair(s) from \(dir.lastPathComponent).")
                } else if previousPairIDs != index.pairs.map(\.id) {
                    self.log("Updated \(index.pairs.count) pair(s) from \(dir.lastPathComponent).")
                }
            }
        }
    }

    func togglePreviewSide() {
        guard selectedPair != nil else { return }
        showingBack.toggle()
    }

    func rotatePreviewImageClockwise() {
        guard let previewURL, !isBusy else { return }
        isBusy = true
        log("Rotating \(previewURL.lastPathComponent) 90 degrees clockwise...")

        Task {
            do {
                try await Task.detached(priority: .userInitiated) {
                    try ImageEditingService.rotateClockwise(fileURL: previewURL)
                }.value

                await MainActor.run {
                    CardPreviewView.invalidateCache(for: previewURL)
                    self.previewRevision += 1
                    log("Rotated \(previewURL.lastPathComponent)")
                    isBusy = false
                    refreshImages(silent: true)
                }
            } catch {
                await MainActor.run {
                    log("Rotate failed: \(error.localizedDescription)")
                    isBusy = false
                }
            }
        }
    }

    func downloadPSACard(certNumber: String) {
        let trimmedCert = certNumber.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedCert.isEmpty, !isBusy else { return }

        let token = SettingsStore.shared.psaToken
        guard !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            log("PSA download failed: No PSA API token found. Add it in Settings.")
            return
        }

        isBusy = true
        log("Downloading PSA cert \(trimmedCert)...")
        let destination = currentDirectory

        Task {
            do {
                let output = try await PSADownloadService.download(
                    certNumber: trimmedCert,
                    outputDirectory: destination,
                    token: token
                )
                await MainActor.run {
                    if !output.isEmpty { log(output) }
                    log("Downloaded PSA cert \(trimmedCert) to \(destination.lastPathComponent)")
                    isBusy = false
                    refreshImages()
                }
            } catch {
                await MainActor.run {
                    log("PSA download failed: \(error.localizedDescription)")
                    isBusy = false
                }
            }
        }
    }

    // MARK: - Selection

    func selectAll() { checkedIDs = Set(visiblePairs.map(\.id)) }
    func selectNone() { checkedIDs = [] }

    func deleteSelectedCard() {
        guard let selectedPair else { return }
        deletePairs([selectedPair])
        selectedPairID = nil
    }

    func deleteSelectedPairs() {
        deletePairs(checkedPairs)
        checkedIDs = []
        selectedPairID = nil
        refreshImages()
    }

    private func deletePairs(_ toDelete: [CardPair]) {
        guard !toDelete.isEmpty else { return }
        var moved = 0
        var failed: [(String, String)] = []
        for pair in toDelete {
            for url in [pair.front, pair.back] {
                do {
                    try FileManager.default.trashItem(at: url, resultingItemURL: nil)
                    moved += 1
                } catch {
                    failed.append((url.lastPathComponent, error.localizedDescription))
                }
            }
            metadataStore.removeMetadata(for: pair)
        }
        let pairWord = toDelete.count == 1 ? "pair" : "pairs"
        log("Moved \(moved) file(s) to Trash from \(toDelete.count) \(pairWord)" + (failed.isEmpty ? "" : "; \(failed.count) failed"))
        for (name, err) in failed { log("  ✗ \(name): \(err)") }
    }
    func toggleCheck(_ pair: CardPair) {
        if checkedIDs.contains(pair.id) { checkedIDs.remove(pair.id) }
        else { checkedIDs.insert(pair.id) }
    }

    func toggleTrait(_ trait: CardTrait) {
        guard let selectedPair else { return }
        let newValue = !metadataStore.traits(for: selectedPair).contains(trait)
        metadataStore.set(trait, to: newValue, for: selectedPair)
        updateSelectedTraits()
        updateVisiblePairs()
        syncSelectionWithVisiblePairs()
    }

    func toggleTraitFilter(_ trait: CardTrait) {
        if selectedTraitFilters.contains(trait) {
            selectedTraitFilters.remove(trait)
        } else {
            selectedTraitFilters.insert(trait)
        }
    }

    func clearFilters() {
        filterText = ""
        selectedTraitFilters = []
    }

    private func syncSelectionWithVisiblePairs() {
        if let selectedPairID, visiblePairs.contains(where: { $0.id == selectedPairID }) {
            return
        }

        selectedPairID = visiblePairs.first?.id
        showingBack = false
    }

    private func updateVisiblePairs() {
        visiblePairs = filteredAndSortedPairs()
    }

    private func updateSelectedTraits() {
        guard let selectedPair else {
            selectedTraits = CardTraits()
            return
        }
        selectedTraits = metadataStore.traits(for: selectedPair)
    }

    private func filteredAndSortedPairs() -> [CardPair] {
        let matchingPairs: [CardPair]
        if filterText.isEmpty {
            matchingPairs = pairs
        } else {
            matchingPairs = pairs.filter { $0.displayName.contains(filterText) }
        }

        let metadataMatchingPairs = matchingPairs.filter {
            metadataStore.matches($0, selectedTraits: selectedTraitFilters)
        }

        return metadataMatchingPairs.sorted { lhs, rhs in
            switch sortOrder {
            case .ascending:
                return CardDirectoryIndexStore.naturalSortLess(lhs.displayName, rhs.displayName)
            case .descending:
                return CardDirectoryIndexStore.naturalSortLess(rhs.displayName, lhs.displayName)
            }
        }
    }

    private func applyDirectoryIndex(_ index: CardDirectoryIndex, pruneMetadata: Bool, updateRecentChecks: Bool) {
        let prevChecked = checkedIDs
        let prevCheckedFronts = pairs
            .filter { prevChecked.contains($0.id) }
            .map { $0.front.lastPathComponent }

        childDirectories = index.childDirectories
        pairs = index.pairs
        if pruneMetadata {
            metadataStore.pruneMetadata(keepingBaseNames: Set(index.pairs.map(\.baseName)))
        }
        updateVisiblePairs()

        if updateRecentChecks {
            if prevChecked.isEmpty {
                checkedIDs = Set(pairs.filter { isRecentPair($0) }.map(\.id))
            } else {
                checkedIDs = Set(pairs.filter { prevCheckedFronts.contains($0.front.lastPathComponent) }.map(\.id))
            }
        } else {
            checkedIDs = Set(pairs.filter { prevChecked.contains($0.id) }.map(\.id))
        }

        syncSelectionWithVisiblePairs()
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
                    let done = completed
                    let p = Double(done) / total
                    await MainActor.run {
                        progress = p
                        log("[\(done)/\(Int(total))] \(r.frontName): \(r.title)")
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

    // MARK: - Watcher

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

}
