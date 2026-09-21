import Foundation
import AppKit
import Observation

/// A `CardPair`'s sort-relevant fields, parsed once from the filename up
/// front so the sort comparator itself only ever does cheap field comparisons.
private struct CardSortKey {
    let pair: CardPair
    let year: String
    let player: String
    let set: String
    let number: Int?
    let variation: String

    init(pair: CardPair) {
        self.pair = pair
        let components = pair.nameComponents
        self.year = pair.parsedYear
        self.player = pair.parsedPlayer
        self.set = "\(components.manufacturer) \(components.series)"
        self.number = components.number
        self.variation = components.variation
    }
}

@Observable
@MainActor
final class CardNamerViewModel {

    // Directory state
    var directoryPath: String = SettingsStore.shared.incomingDirectory.path
    var childDirectories: [URL] = []
    var pairs: [CardPair] = []
    var visiblePairs: [CardPair] = []
    private var pairsByID: [CardPair.ID: CardPair] = [:]
    var filterText: String = "" { didSet { scheduleFilterUpdate() } }
    var filterPlayer: String = "" { didSet { scheduleFilterUpdate() } }
    var filterYear: String = "" { didSet { scheduleFilterUpdate() } }
    var filterSet: String = "" { didSet { scheduleFilterUpdate() } }
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
    var selectedTraitFilters: Set<CardTrait> = [] {
        didSet {
            updateVisiblePairs()
            syncSelectionWithVisiblePairs()
        }
    }
    var pairsWithTraits: Set<String> = []
    var listedPairs: Set<String> = []
    var hideListed: Bool = SettingsStore.shared.hideListedCardNamer {
        didSet {
            SettingsStore.shared.hideListedCardNamer = hideListed
            updateVisiblePairs()
            syncSelectionWithVisiblePairs()
        }
    }

    var selectedIDs: Set<CardPair.ID> = [] {
        didSet {
            proposedName = selectedPair?.baseName ?? ""
            updateSelectionMetadata()
        }
    }

    var selectedPairID: CardPair.ID? {
        get { selectedIDs.count == 1 ? selectedIDs.first : nil }
        set {
            if let newValue { selectedIDs = [newValue] }
            else { selectedIDs = [] }
        }
    }

    // Naming
    var proposedName: String = ""
    var isBusy: Bool = false

    // Log
    var logText: String = ""

    // Preview
    var showingBack: Bool = false
    var selectedTraits: CardTraits = CardTraits()
    var selectedListing: CardListing = CardListing()
    var previewRevision: Int = 0

    private let watcher = DirectoryWatcher()
    private let metadataStore = CardMetadataStore(directoryURL: SettingsStore.shared.incomingDirectory)
    private let listingStore = CardListingStore(directoryURL: SettingsStore.shared.incomingDirectory)
    private var debounceTask: Task<Void, Never>?
    private var refreshTask: Task<Void, Never>?
    private var refreshGeneration = 0
    private var pendingSelectionID: CardPair.ID?

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

    var currentDirectory: URL {
        URL(fileURLWithPath: directoryPath).standardized
    }

    var parentDirectory: URL? {
        let parent = currentDirectory.deletingLastPathComponent().standardized
        return parent == currentDirectory ? nil : parent
    }

    var selectedPair: CardPair? {
        guard let selectedPairID else { return nil }
        return pairsByID[selectedPairID]
    }

    var selectedPairs: [CardPair] {
        visiblePairs.filter { selectedIDs.contains($0.id) }
    }

    var hasActiveFilter: Bool {
        !filterText.isEmpty || !filterPlayer.isEmpty || !filterYear.isEmpty || !filterSet.isEmpty
            || !selectedTraitFilters.isEmpty || hideListed
    }

    var hiddenListedCount: Int {
        pairs.reduce(into: 0) { count, pair in
            if listedPairs.contains(pair.baseName) { count += 1 }
        }
    }

    var previewURL: URL? {
        guard let pair = selectedPair else { return nil }
        return showingBack ? pair.back : pair.front
    }

    var isExistingCardsDirectory: Bool {
        currentDirectory == SettingsStore.shared.existingCardsDirectory
    }

    // MARK: - Directory management

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
        proposedName = ""
        selectedPairID = nil
        showingBack = false
        refreshImages()
    }

    func chooseDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = currentDirectory
        if panel.runModal() == .OK, let url = panel.url {
            directoryPath = url.path
            refreshImages()
        }
    }

// MARK: - Image loading

    private func filteredAndSortedPairs() -> [CardPair] {
        let text = filterText.lowercased()
        let player = filterPlayer.lowercased()
        let year = filterYear
        let set = filterSet.lowercased()

        let matchingPairs = pairs.filter { pair in
            (text.isEmpty   || pair.displayName.lowercased().contains(text)) &&
            (player.isEmpty || pair.parsedPlayer.contains(player)) &&
            (year.isEmpty   || pair.parsedYear.contains(year)) &&
            (set.isEmpty    || pair.parsedSetText.contains(set))
        }

        let metadataMatchingPairs = matchingPairs.filter {
            metadataStore.matches($0, selectedTraits: selectedTraitFilters)
                && (!hideListed || !listedPairs.contains($0.baseName))
        }

        // Parse each pair's sort fields once up front (O(n)) rather than inside
        // the comparator, which would re-parse the filename on every pairwise
        // comparison (O(n log n) parses) — costly once a folder has 1,000+ cards.
        let keyed = metadataMatchingPairs.map { CardSortKey(pair: $0) }
        let ordered = keyed.sorted { lhs, rhs in
            switch sortOrder {
            case .ascending:
                return sortLess(lhs, rhs)
            case .descending:
                return sortLess(rhs, lhs)
            }
        }
        return ordered.map(\.pair)
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

    private func sortLess(_ lhs: CardSortKey, _ rhs: CardSortKey) -> Bool {
        switch sortField {
        case .name:
            return cardNameLess(lhs, rhs)
        case .modificationDate:
            let lhsDate = lhs.pair.modificationDate
            let rhsDate = rhs.pair.modificationDate
            if lhsDate != rhsDate { return lhsDate < rhsDate }
            return cardNameLess(lhs, rhs)
        }
    }

    /// Orders by year, player, set (manufacturer + series), then card number
    /// (numeric, so 2 < 25 < 206) with variation only as a final tiebreaker —
    /// this keeps same-numbered cards grouped together regardless of variation.
    /// Operates on precomputed `CardSortKey`s so no filename parsing happens here.
    private func cardNameLess(_ lhs: CardSortKey, _ rhs: CardSortKey) -> Bool {
        if lhs.year != rhs.year {
            return CardDirectoryIndexStore.naturalSortLess(lhs.year, rhs.year)
        }
        if lhs.player != rhs.player {
            return CardDirectoryIndexStore.naturalSortLess(lhs.player, rhs.player)
        }
        if lhs.set != rhs.set {
            return CardDirectoryIndexStore.naturalSortLess(lhs.set, rhs.set)
        }
        if lhs.number != rhs.number {
            guard let lhsNumber = lhs.number else { return false }
            guard let rhsNumber = rhs.number else { return true }
            return lhsNumber < rhsNumber
        }
        if lhs.variation != rhs.variation {
            return CardDirectoryIndexStore.naturalSortLess(lhs.variation, rhs.variation)
        }
        return CardDirectoryIndexStore.naturalSortLess(lhs.pair.displayName, rhs.pair.displayName)
    }

    func refreshImages(silent: Bool = false) {
        refreshTask?.cancel()
        refreshGeneration += 1
        let generation = refreshGeneration
        let dir = currentDirectory
        guard FileManager.default.fileExists(atPath: dir.path) else {
            if !silent { log("Directory does not exist: \(dir.path)") }
            childDirectories = []
            pairs = []
            visiblePairs = []
            return
        }
        metadataStore.load(directoryURL: dir)
        listingStore.load(directoryURL: dir)
        startWatching()

        if let cachedIndex = CardDirectoryIndexStore.cachedIndex(for: dir) {
            applyDirectoryIndex(cachedIndex, pruneMetadata: false)
            if !silent {
                log("Loaded \(cachedIndex.pairs.count) cached card pair(s); verifying folder...")
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
                if index.imageFileCount == 0 {
                    self.log("No image files found.")
                } else {
                    self.log("Loaded \(index.pairs.count) card pair(s) from \(dir.lastPathComponent).")
                }
            } else if previousPairIDs != index.pairs.map(\.id) {
                self.log("Updated \(index.pairs.count) card pair(s) from \(dir.lastPathComponent).")
            }
        }
    }

    // MARK: - Naming

    func startNaming() {
        guard let pair = selectedPair, !isBusy else { return }
        isBusy = true
        proposedName = ""
        log("Naming \(pair.front.lastPathComponent) + \(pair.back.lastPathComponent) ...")

        Task {
            do {
                let name = try await namePair(pair)
                proposedName = name
                log("Suggested name: \(name)")
                isBusy = false
            } catch {
                log("Naming failed: \(error.localizedDescription)")
                isBusy = false
            }
        }
    }

    private func namePair(_ pair: CardPair) async throws -> String {
        async let ocrFront = OCRService.recognize(imageURL: pair.front)
        async let ocrBack = OCRService.recognize(imageURL: pair.back)
        async let ocrBackBottom = OCRService.recognize(imageURL: pair.back, cropBottom: true)

        let (front, back, bottom) = await (ocrFront, ocrBack, ocrBackBottom)
        var details = try await OpenAIService.identifyCard(
            frontURL: pair.front,
            backURL: pair.back,
            ocrFront: front,
            ocrBack: back,
            ocrBackBottom: bottom
        )
        CardNameBuilder.refineYear(details: &details, ocrFront: front, ocrBack: back, ocrBackBottom: bottom)
        return CardNameBuilder.buildBaseName(from: details)
    }

    // MARK: - Accept name (rename files)

    func acceptName() {
        guard let pair = selectedPair else { return }
        let sanitized = CardNameBuilder.sanitize(proposedName)
        guard !sanitized.isEmpty else { return }
        if sanitized != proposedName { proposedName = sanitized }

        guard let renamed = renameFiles(for: pair, to: sanitized) else { return }
        log("Renamed to \(renamed.front.lastPathComponent) and \(renamed.back.lastPathComponent)")
        proposedName = ""
        pendingSelectionID = "\(renamed.front.standardized.path)|\(renamed.back.standardized.path)"
        refreshImages()
    }

    /// Renames the file pair to `desiredBase`, resolving collisions by appending a
    /// numeric suffix. Returns the new front/back URLs, or nil on failure.
    @discardableResult
    private func renameFiles(for pair: CardPair, to desiredBase: String) -> (front: URL, back: URL)? {
        let sanitized = CardNameBuilder.sanitize(desiredBase)
        guard !sanitized.isEmpty else { return nil }

        let dir = pair.front.deletingLastPathComponent()
        let ext = pair.front.pathExtension.lowercased()
        let backExt = pair.back.pathExtension.lowercased()
        let finalBase = uniqueBaseName(sanitized, in: dir, ext: ext, backExt: backExt, excluding: pair)
        let newFront = dir.appendingPathComponent("\(finalBase).\(ext)")
        let newBack  = dir.appendingPathComponent("\(finalBase)_b.\(backExt)")

        do {
            if pair.front.standardized != newFront.standardized {
                CardPreviewView.invalidateCache(for: pair.front)
                try FileManager.default.moveItem(at: pair.front, to: newFront)
            }
            if pair.back.standardized != newBack.standardized {
                CardPreviewView.invalidateCache(for: pair.back)
                try FileManager.default.moveItem(at: pair.back, to: newBack)
            }
            metadataStore.moveMetadata(from: pair.baseName, to: finalBase)
            listingStore.moveListing(from: pair.baseName, to: finalBase)
            return (newFront, newBack)
        } catch {
            log("Rename failed for \(pair.displayName): \(error.localizedDescription)")
            return nil
        }
    }

    /// Finds a base name whose front/back files don't already exist (ignoring the pair's own files).
    private func uniqueBaseName(_ base: String, in dir: URL, ext: String, backExt: String, excluding pair: CardPair) -> String {
        let fm = FileManager.default
        func taken(_ candidate: String) -> Bool {
            let f = dir.appendingPathComponent("\(candidate).\(ext)")
            let b = dir.appendingPathComponent("\(candidate)_b.\(backExt)")
            let fExists = fm.fileExists(atPath: f.path) && f.standardized != pair.front.standardized
            let bExists = fm.fileExists(atPath: b.path) && b.standardized != pair.back.standardized
            return fExists || bExists
        }
        if !taken(base) { return base }
        var i = 2
        while taken("\(base)_\(i)") { i += 1 }
        return "\(base)_\(i)"
    }

    // MARK: - Quick Name (batch AI naming)

    func quickNameSelected() {
        quickName(selectedPairs)
    }

    /// Names each pair via the AI route and applies the returned name automatically.
    func quickName(_ targets: [CardPair]) {
        guard !isBusy, !targets.isEmpty else { return }
        isBusy = true
        log("Quick Name: processing \(targets.count) card\(targets.count == 1 ? "" : "s") ...")

        Task {
            var succeeded = 0
            var failed = 0
            for pair in targets {
                do {
                    let name = try await namePair(pair)
                    if let renamed = renameFiles(for: pair, to: name) {
                        log("Renamed \(pair.displayName) → \(renamed.front.lastPathComponent)")
                        succeeded += 1
                    } else {
                        failed += 1
                    }
                } catch {
                    log("Quick Name failed for \(pair.displayName): \(error.localizedDescription)")
                    failed += 1
                }
            }
            log("Quick Name complete: \(succeeded) named, \(failed) failed")
            isBusy = false
            refreshImages()
        }
    }

    // MARK: - Delete card

    func deleteSelectedCard() {
        deleteCards(selectedPairs)
    }

    func deleteCard(_ pair: CardPair) {
        deleteCards([pair])
    }

    func deleteCards(_ targets: [CardPair]) {
        for pair in targets {
            do {
                try FileManager.default.trashItem(at: pair.front, resultingItemURL: nil)
                try FileManager.default.trashItem(at: pair.back, resultingItemURL: nil)
                metadataStore.removeMetadata(for: pair)
                listingStore.removeListing(for: pair)
                log("Moved to Trash: \(pair.front.lastPathComponent) + \(pair.back.lastPathComponent)")
            } catch {
                log("Delete failed: \(error.localizedDescription)")
            }
        }
        refreshImages()
    }

    // MARK: - Move cards

    func moveSelectedCard() {
        guard let pair = selectedPair else { return }
        movePairs([pair], to: SettingsStore.shared.existingCardsDirectory)
    }

    func moveAllCards() {
        movePairs(visiblePairs, to: SettingsStore.shared.existingCardsDirectory)
    }

    func moveCardToSales(_ pair: CardPair) {
        movePairs([pair], to: SettingsStore.shared.currentSalesDirectory)
    }

    func moveCardToCollection(_ pair: CardPair) {
        movePairs([pair], to: SettingsStore.shared.existingCardsDirectory)
    }

    func moveSelectedToSales() {
        movePairs(selectedPairs, to: SettingsStore.shared.currentSalesDirectory)
    }

    func moveSelectedToCollection() {
        movePairs(selectedPairs, to: SettingsStore.shared.existingCardsDirectory)
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
            if movedFilesForPair == 2 {
                metadataStore.moveMetadata(for: pair, to: dest)
                listingStore.moveListing(for: pair, to: dest)
            }
        }
        log("Move complete: \(moved) moved, \(skipped) skipped → \(dest.lastPathComponent)")
        refreshImages()
    }


    // MARK: - Merge fronts

    /// Combines the selected cards' front images into one grid image, named
    /// after the first card in the current sort order.
    func mergeFronts(_ targets: [CardPair]) {
        guard !isBusy,
              targets.count >= CardMergeService.minimumImages,
              targets.count <= CardMergeService.maximumImages else { return }

        isBusy = true
        let fronts = targets.map(\.front)
        let backs = targets.map(\.back)
        let destination = currentDirectory
        log("Merging \(fronts.count) card(s)...")

        Task {
            do {
                let output = try await Task.detached(priority: .userInitiated) {
                    try CardMergeService.merge(fronts: fronts, backs: backs, in: destination)
                }.value
                // A re-merge overwrites in place, so the previously decoded
                // image for those URLs has to go or the stale one keeps showing.
                CardPreviewView.invalidateCache(for: output.front)
                CardPreviewView.invalidateCache(for: output.back)
                previewRevision += 1
                log("Merged \(fronts.count) cards -> \(output.front.lastPathComponent) + \(output.back.lastPathComponent)")
                isBusy = false
                refreshImages()
            } catch {
                log("Merge failed: \(error.localizedDescription)")
                isBusy = false
            }
        }
    }

    // MARK: - Search

    func openTCDB() {
        guard let url = CardNameBuilder.tcdbURL(fromBaseName: proposedName) else { return }
        NSWorkspace.shared.open(url)
        copyToClipboard(url.absoluteString)
        log("Opened TCDB: \(url.absoluteString)")
    }

    func openEbay() {
        guard let url = CardNameBuilder.ebayURL(fromBaseName: proposedName) else { return }
        NSWorkspace.shared.open(url)
        copyToClipboard(url.absoluteString)
        log("Opened eBay: \(url.absoluteString)")
    }

    // MARK: - Helpers

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

    func toggleTrait(_ trait: CardTrait) {
        guard let selectedPair else { return }
        let newValue = !metadataStore.traits(for: selectedPair).contains(trait)
        metadataStore.set(trait, to: newValue, for: selectedPair)
        updateSelectionMetadata()
        updatePairsWithTraits()
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
        filterPlayer = ""
        filterYear = ""
        filterSet = ""
        selectedTraitFilters = []
    }

    func log(_ message: String) {
        let line = "[\(timeString())] \(message)"
        if logText.isEmpty {
            logText = line
        } else {
            logText += "\n" + line
        }
    }

    private func timeString() -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f.string(from: Date())
    }

    private func copyToClipboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
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

    private func updatePairsWithTraits() {
        pairsWithTraits = Set(pairs.compactMap { pair in
            metadataStore.traits(for: pair).hasAnyTrait ? pair.baseName : nil
        })
    }

    private func updateSelectionMetadata() {
        guard let selectedPair else {
            selectedTraits = CardTraits()
            selectedListing = CardListing()
            return
        }
        selectedTraits = metadataStore.traits(for: selectedPair)
        selectedListing = listingStore.listing(for: selectedPair)
    }

    private func updateListedPairs() {
        listedPairs = listingStore.listedBaseNames
    }

    private func applyDirectoryIndex(_ index: CardDirectoryIndex, pruneMetadata: Bool) {
        let prevIDs = selectedIDs
        childDirectories = index.childDirectories
        pairs = index.pairs
        pairsByID = Dictionary(uniqueKeysWithValues: pairs.map { ($0.id, $0) })
        if pruneMetadata {
            let validBaseNames = Set(index.pairs.map(\.baseName))
            metadataStore.pruneMetadata(keepingBaseNames: validBaseNames)
            listingStore.pruneListings(keepingBaseNames: validBaseNames)
        }
        updatePairsWithTraits()
        updateListedPairs()
        updateVisiblePairs()

        if let pending = pendingSelectionID, pairsByID[pending] != nil {
            pendingSelectionID = nil
            selectedIDs = [pending]
        } else {
            let stillValid = prevIDs.filter { pairsByID[$0] != nil }
            if stillValid != selectedIDs { selectedIDs = stillValid }
            syncSelectionWithVisiblePairs()
        }
    }

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
