import Foundation
import AppKit
import Observation

struct EbayTitleResult: Identifiable {
    let id = UUID()
    let frontName: String
    var title: String
}

@Observable
@MainActor
final class EbayTitlesViewModel {

    var directoryPath: String = SettingsStore.shared.incomingDirectory.path
    var childDirectories: [URL] = []
    var pairs: [CardPair] = []
    var visiblePairs: [CardPair] = []
    private var pairsByID: [CardPair.ID: CardPair] = [:]
    /// Generated results are keyed by front filename; listing records are keyed
    /// by base name, so this bridges the two.
    private var pairsByFrontName: [String: CardPair] = [:]
    var filterText: String = "" { didSet { scheduleFilterUpdate() } }
    private var filterDebounceTask: Task<Void, Never>?
    var sortField: CardPairSortField = .name {
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
    /// eBay Titles works on raw scans, so filename-derived filters don't apply
    /// here; hiding what's already been listed is the filter that matters.
    var hideListed: Bool = SettingsStore.shared.hideListedEbayTitles {
        didSet {
            SettingsStore.shared.hideListedEbayTitles = hideListed
            updateVisiblePairs()
            syncSelectionWithVisiblePairs()
        }
    }
    var listedPairs: Set<String> = []
    var selectedIDs: Set<CardPair.ID> = [] {
        didSet {
            showingBack = false
            updateSelectionMetadata()
        }
    }

    var category: EbayCategory = .sportsCards
    var setOverride: String = ""
    var varietyOverride: String = ""
    var supplementalRules: String = SettingsStore.shared.ebaySupplementalRules {
        didSet {
            SettingsStore.shared.ebaySupplementalRules = supplementalRules
        }
    }

    var results: [EbayTitleResult] = []
    var progress: Double = 0
    var isBusy: Bool = false
    var logText: String = ""

    var showingBack: Bool = false
    var selectedPairID: CardPair.ID? {
        get { selectedIDs.count == 1 ? selectedIDs.first : nil }
        set {
            if let newValue { selectedIDs = [newValue] }
            else { selectedIDs = [] }
        }
    }
    var selectedTraits: CardTraits = CardTraits()
    var selectedListing: CardListing = CardListing()
    var previewRevision: Int = 0

    private let watcher = DirectoryWatcher()
    private let metadataStore = CardMetadataStore(directoryURL: SettingsStore.shared.incomingDirectory)
    private let listingStore = CardListingStore(directoryURL: SettingsStore.shared.incomingDirectory)
    private var debounceTask: Task<Void, Never>?
    private var refreshTask: Task<Void, Never>?
    private var refreshGeneration = 0

    init() {
        startWatching()
        refreshImages()
        // Both observers are delivered on `.main`, so the blocks are already on the
        // main actor by the time they run.
        NotificationCenter.default.addObserver(forName: .goToDirectory, object: nil, queue: .main) { [weak self] note in
            guard let dir = note.object as? QuickDirectory else { return }
            MainActor.assumeIsolated {
                self?.switchTo(dir)
            }
        }
        NotificationCenter.default.addObserver(forName: .refreshImages, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.refreshImages()
            }
        }
    }

    // MARK: - Computed

    var currentDirectory: URL { URL(fileURLWithPath: directoryPath).standardized }
    var parentDirectory: URL? {
        let parent = currentDirectory.deletingLastPathComponent().standardized
        return parent == currentDirectory ? nil : parent
    }
    var selectedPair: CardPair? {
        guard let selectedPairID else { return nil }
        return pairsByID[selectedPairID]
    }
    var previewURL: URL? {
        guard let pair = selectedPair else { return nil }
        return showingBack ? pair.back : pair.front
    }
    var selectedPairs: [CardPair] { visiblePairs.filter { selectedIDs.contains($0.id) } }
    var titlesCSVURL: URL { currentDirectory.appendingPathComponent("description.csv") }
    var hasSavedTitles: Bool { FileManager.default.fileExists(atPath: titlesCSVURL.path) }
    var hasActiveFilter: Bool {
        !filterText.isEmpty || hideListed
    }
    var hiddenListedCount: Int {
        pairs.reduce(into: 0) { count, pair in
            if listedPairs.contains(pair.baseName) { count += 1 }
        }
    }

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
        selectedIDs = []
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
        listingStore.load(directoryURL: dir)
        startWatching()

        if let cachedIndex = CardDirectoryIndexStore.cachedIndex(for: dir) {
            applyDirectoryIndex(cachedIndex, pruneMetadata: false)
            if !silent {
                log("Loaded \(cachedIndex.pairs.count) cached pair(s); verifying folder...")
            }
        } else if !silent {
            log("Scanning \(dir.lastPathComponent)...")
        }

        refreshTask = Task {
            let index = await Task.detached(priority: .userInitiated) {
                CardDirectoryIndexStore.scanDirectory(dir)
            }.value

            guard !Task.isCancelled, self.refreshGeneration == generation, self.currentDirectory == dir else { return }
            let previousPairIDs = self.pairs.map(\.id)
            self.applyDirectoryIndex(index, pruneMetadata: true)
            Task.detached(priority: .background) { CardDirectoryIndexStore.saveCacheIfNeeded(for: index) }
            if !silent {
                self.log("Loaded \(index.pairs.count) pair(s) from \(dir.lastPathComponent).")
            } else if previousPairIDs != index.pairs.map(\.id) {
                self.log("Updated \(index.pairs.count) pair(s) from \(dir.lastPathComponent).")
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

                CardPreviewView.invalidateCache(for: previewURL)
                self.previewRevision += 1
                log("Rotated \(previewURL.lastPathComponent)")
                isBusy = false
                refreshImages(silent: true)
            } catch {
                log("Rotate failed: \(error.localizedDescription)")
                isBusy = false
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
                if !output.isEmpty { log(output) }
                log("Downloaded PSA cert \(trimmedCert) to \(destination.lastPathComponent)")
                isBusy = false
                refreshImages()
            } catch {
                log("PSA download failed: \(error.localizedDescription)")
                isBusy = false
            }
        }
    }

    // MARK: - Selection

    func selectAll() { selectedIDs = Set(visiblePairs.map(\.id)) }
    func selectNone() { selectedIDs = [] }

    func deleteSelectedCard() {
        deleteSelectedPairs()
    }

    func deleteSelectedPairs() {
        deleteCards(selectedPairs)
    }

    func deleteCards(_ targets: [CardPair]) {
        deletePairs(targets)
        selectedIDs = []
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
            listingStore.removeListing(for: pair)
        }
        let pairWord = toDelete.count == 1 ? "pair" : "pairs"
        log("Moved \(moved) file(s) to Trash from \(toDelete.count) \(pairWord)" + (failed.isEmpty ? "" : "; \(failed.count) failed"))
        for (name, err) in failed { log("  ✗ \(name): \(err)") }
    }
    func toggleTrait(_ trait: CardTrait) {
        guard let selectedPair else { return }
        let newValue = !metadataStore.traits(for: selectedPair).contains(trait)
        metadataStore.set(trait, to: newValue, for: selectedPair)
        updateSelectionMetadata()
        updateVisiblePairs()
        syncSelectionWithVisiblePairs()
    }

    func clearFilters() {
        filterText = ""
    }

    // MARK: - Listing records

    func listing(for pair: CardPair) -> CardListing {
        listingStore.listing(for: pair)
    }

    /// Marks or unmarks cards as listed. A mixed selection lands on one
    /// consistent state, taken from the first card in the group.
    func toggleListed(_ targets: [CardPair]) {
        guard let first = targets.first else { return }
        let newValue = !listingStore.listing(for: first).listed
        for pair in targets {
            listingStore.setListed(newValue, for: pair)
        }
        updateListedPairs()
        updateSelectionMetadata()
        updateVisiblePairs()
        syncSelectionWithVisiblePairs()
        let cardWord = targets.count == 1 ? "card" : "cards"
        log("\(newValue ? "Marked" : "Unmarked") \(targets.count) \(cardWord) as listed")
    }

    // MARK: - Move cards

    func movePairsToSales(_ targets: [CardPair]) {
        movePairs(targets, to: SettingsStore.shared.currentSalesDirectory)
    }

    func movePairsToCollection(_ targets: [CardPair]) {
        movePairs(targets, to: SettingsStore.shared.existingCardsDirectory)
    }

    private func movePairs(_ targets: [CardPair], to dest: URL) {
        try? FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
        var moved = 0
        var skipped = 0
        for pair in targets {
            var movedFilesForPair = 0
            for source in [pair.front, pair.back] {
                let target = dest.appendingPathComponent(source.lastPathComponent)
                if source.standardized == target.standardized { skipped += 1; continue }
                do {
                    if FileManager.default.fileExists(atPath: target.path) {
                        try FileManager.default.removeItem(at: target)
                    }
                    try FileManager.default.moveItem(at: source, to: target)
                    moved += 1
                    movedFilesForPair += 1
                } catch {
                    log("Move failed for \(source.lastPathComponent): \(error.localizedDescription)")
                    skipped += 1
                }
            }
            // Both halves landed, so the card's traits and listing record follow it.
            if movedFilesForPair == 2 {
                metadataStore.moveMetadata(for: pair, to: dest)
                listingStore.moveListing(for: pair, to: dest)
            }
        }
        log("Move complete: \(moved) moved, \(skipped) skipped → \(dest.lastPathComponent)")
        refreshImages()
    }

    private func syncSelectionWithVisiblePairs() {
        let validIDs = selectedIDs.filter { id in visiblePairs.contains(where: { $0.id == id }) }
        if !validIDs.isEmpty {
            if validIDs != selectedIDs { selectedIDs = validIDs }
        } else {
            selectedIDs = visiblePairs.first.map { [$0.id] } ?? []
            showingBack = false
        }
    }

    private func updateVisiblePairs() {
        visiblePairs = filteredAndSortedPairs()
    }

    private func updateSelectionMetadata() {
        guard let selectedPair else {
            selectedTraits = CardTraits()
            selectedListing = CardListing()
            return
        }
        selectedTraits = metadataStore.traits(for: selectedPair)
        selectedListing = listingStore.listing(for: selectedPair)

        // Show the rules this card's title was generated under. Cards with no
        // recorded category leave the picker where the user last put it.
        if let storedCategory = selectedListing.ebayCategory, storedCategory != category {
            category = storedCategory
        }
    }

    private func updateListedPairs() {
        listedPairs = listingStore.listedBaseNames
    }

    private func filteredAndSortedPairs() -> [CardPair] {
        let text = filterText.lowercased()

        let matchingPairs = pairs.filter { pair in
            (text.isEmpty || pair.displayName.lowercased().contains(text)) &&
            (!hideListed  || !listedPairs.contains(pair.baseName))
        }

        return matchingPairs.sorted { lhs, rhs in
            switch sortOrder {
            case .ascending:
                return sortLess(lhs, rhs)
            case .descending:
                return sortLess(rhs, lhs)
            }
        }
    }

    private func scheduleFilterUpdate() {
        filterDebounceTask?.cancel()
        filterDebounceTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 150_000_000)
            guard !Task.isCancelled else { return }
            self.updateVisiblePairs()
            self.syncSelectionWithVisiblePairs()
        }
    }

    private func sortLess(_ lhs: CardPair, _ rhs: CardPair) -> Bool {
        switch sortField {
        case .name:
            return CardDirectoryIndexStore.naturalSortLess(lhs.displayName, rhs.displayName)
        case .modificationDate:
            let lhsDate = lhs.modificationDate
            let rhsDate = rhs.modificationDate
            if lhsDate != rhsDate { return lhsDate < rhsDate }
            return CardDirectoryIndexStore.naturalSortLess(lhs.displayName, rhs.displayName)
        }
    }

    private func applyDirectoryIndex(_ index: CardDirectoryIndex, pruneMetadata: Bool) {
        let prevIDs = selectedIDs
        childDirectories = index.childDirectories
        pairs = index.pairs
        pairsByID = Dictionary(uniqueKeysWithValues: pairs.map { ($0.id, $0) })
        pairsByFrontName = Dictionary(pairs.map { ($0.front.lastPathComponent, $0) }) { first, _ in first }
        if pruneMetadata {
            let validBaseNames = Set(index.pairs.map(\.baseName))
            metadataStore.pruneMetadata(keepingBaseNames: validBaseNames)
            listingStore.pruneListings(keepingBaseNames: validBaseNames)
        }
        updateListedPairs()
        updateVisiblePairs()

        let stillValid = prevIDs.filter { pairsByID[$0] != nil }
        if stillValid != selectedIDs { selectedIDs = stillValid }
        syncSelectionWithVisiblePairs()
    }

    // MARK: - Generate

    func generateTitles() {
        let targets = selectedPairs
        guard !isBusy, !targets.isEmpty else { return }
        // One card is an inline edit of that card, not a batch: it updates the
        // title shown in the detail pane and leaves the results window alone.
        let isSingleCard = targets.count == 1
        isBusy = true
        progress = 0
        if !isSingleCard { results = [] }
        log("Starting \(category.rawValue) — \(targets.count) pair(s)…")

        let cat = category
        // Overrides only exist for categories whose UI shows them; otherwise a
        // value left over from a previous category would leak into the prompt.
        let setOvr = cat.showsOverrides ? setOverride.trimmingCharacters(in: .whitespaces) : ""
        let varOvr = cat.showsOverrides ? varietyOverride.trimmingCharacters(in: .whitespaces) : ""
        let extraRules = supplementalRules.trimmingCharacters(in: .whitespacesAndNewlines)
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
                    varietyOverride: varOvr.isEmpty ? nil : varOvr,
                    supplementalRules: extraRules.isEmpty ? nil : extraRules
                )
                let correctedTitle = cat == .sportsCards
                    ? BasketballRookieLookup.bundled.correctingRookieMarker(in: title)
                    : title
                return (index, EbayTitleResult(frontName: frontName, title: correctedTitle))
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
                    log("→ \(pair.front.lastPathComponent)")
                    group.addTask { await runOne(index, pair) }
                }

                // Drain: as each completes, log it and launch the next one
                while let (idx, r) = await group.next() {
                    indexed.append((idx, r))
                    storeTitle(r, category: cat)
                    completed += 1
                    progress = Double(completed) / total
                    log("[\(completed)/\(Int(total))] \(r.frontName): \(r.title)")
                    if let (index, pair) = iterator.next() {
                        log("→ \(pair.front.lastPathComponent)")
                        group.addTask { await runOne(index, pair) }
                    }
                }
            }

            let sorted = indexed.sorted { $0.0 > $1.0 }.map(\.1)
            isBusy = false
            progress = 1.0
            // The title is already saved against the card by `storeTitle`, so
            // description.csv only needs the matching rows merged in.
            upsertCSVRows(sorted)
            if !isSingleCard {
                results = sorted
                NotificationCenter.default.post(name: .showEbayResultsWindow, object: nil)
            }
        }
    }

    /// Commits a hand-edited title for the selected card.
    func updateSelectedTitle(_ title: String) {
        guard let pair = selectedPair else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed != (listingStore.listing(for: pair).title ?? "") else { return }

        listingStore.setTitle(trimmed, for: pair)
        updateSelectionMetadata()
        if !trimmed.isEmpty {
            upsertCSVRows([EbayTitleResult(frontName: pair.front.lastPathComponent, title: trimmed)])
        }
        log("Updated title for \(pair.front.lastPathComponent)")
    }

    func displaySavedTitles() {
        guard hasSavedTitles else { return }
        do {
            let loaded = try loadCSVResults()
            results = loaded
            // Migrate titles written before listing records existed.
            var backfilled = 0
            for result in loaded {
                guard let pair = pairsByFrontName[result.frontName],
                      listingStore.listing(for: pair).title == nil else { continue }
                storeTitle(result)
                backfilled += 1
            }
            log("Loaded \(loaded.count) title(s) from description.csv"
                + (backfilled > 0 ? "; saved \(backfilled) to \(CardListingStore.fileName)" : ""))
            NotificationCenter.default.post(name: .showEbayResultsWindow, object: nil)
        } catch {
            log("Could not load description.csv: \(error.localizedDescription)")
        }
    }

    func saveEditedTitles() {
        for result in results { storeTitle(result) }
        upsertCSVRows(results)
    }

    /// Merges rows into description.csv instead of replacing the file. Writing
    /// it wholesale meant generating or editing a handful of cards discarded the
    /// titles already saved for every other card in the folder.
    private func upsertCSVRows(_ rows: [EbayTitleResult]) {
        let usableRows = rows.filter { !$0.title.hasPrefix("ERROR:") }
        guard !usableRows.isEmpty else { return }

        var merged = (try? loadCSVResults()) ?? []
        for row in usableRows {
            if let index = merged.firstIndex(where: { $0.frontName == row.frontName }) {
                merged[index].title = row.title
            } else {
                merged.insert(row, at: 0)
            }
        }
        saveCSV(merged)
    }

    /// Persists a generated or hand-edited title against its card. Errors are
    /// already surfaced in the log and aren't worth recording.
    private func storeTitle(_ result: EbayTitleResult, category: EbayCategory? = nil) {
        guard !result.title.hasPrefix("ERROR:"),
              let pair = pairsByFrontName[result.frontName] else { return }
        listingStore.setTitle(result.title, category: category, for: pair)
        updateSelectionMetadata()
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
            log("description.csv updated — \(rows.count) title(s) total")
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
        // DirectoryWatcher's DispatchSource is created with `queue: .main`, so this
        // handler always arrives on the main thread.
        watcher.onChange = { [weak self] in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.debounceTask?.cancel()
                self.debounceTask = Task { [weak self] in
                    try? await Task.sleep(nanoseconds: 400_000_000)
                    guard !Task.isCancelled else { return }
                    self?.refreshImages(silent: true)
                }
            }
        }
    }

}
