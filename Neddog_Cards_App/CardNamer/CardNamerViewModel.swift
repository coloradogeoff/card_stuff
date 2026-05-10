import Foundation
import AppKit
import Observation

@Observable
final class CardNamerViewModel {

    // Directory state
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
    var selectedPairID: CardPair.ID? {
        didSet {
            proposedName = selectedPair?.baseName ?? ""
            updateSelectedTraits()
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

    var currentDirectory: URL {
        URL(fileURLWithPath: directoryPath).standardized
    }

    var parentDirectory: URL? {
        let parent = currentDirectory.deletingLastPathComponent().standardized
        return parent == currentDirectory ? nil : parent
    }

    var selectedPair: CardPair? {
        pairs.first { $0.id == selectedPairID }
    }

    var hasActiveFilter: Bool {
        !filterText.isEmpty || !selectedTraitFilters.isEmpty
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
                let scannedIndex = CardDirectoryIndexStore.scanDirectory(dir)
                CardDirectoryIndexStore.saveCacheIfNeeded(for: scannedIndex)
                return scannedIndex
            }.value

            await MainActor.run {
                guard !Task.isCancelled, self.refreshGeneration == generation, self.currentDirectory == dir else { return }
                let previousPairIDs = self.pairs.map(\.id)
                self.applyDirectoryIndex(index, pruneMetadata: true)

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
                await MainActor.run {
                    proposedName = name
                    log("Suggested name: \(name)")
                    isBusy = false
                }
            } catch {
                await MainActor.run {
                    log("Naming failed: \(error.localizedDescription)")
                    isBusy = false
                }
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

        let dir = pair.front.deletingLastPathComponent()
        let ext = pair.front.pathExtension.lowercased()
        let backExt = pair.back.pathExtension.lowercased()
        let newFront = dir.appendingPathComponent("\(sanitized).\(ext)")
        let newBack  = dir.appendingPathComponent("\(sanitized)_b.\(backExt)")

        do {
            if pair.front.standardized != newFront.standardized {
                try FileManager.default.moveItem(at: pair.front, to: newFront)
            }
            if pair.back.standardized != newBack.standardized {
                try FileManager.default.moveItem(at: pair.back, to: newBack)
            }
            metadataStore.moveMetadata(from: pair.baseName, to: sanitized)
            log("Renamed to \(newFront.lastPathComponent) and \(newBack.lastPathComponent)")
            proposedName = ""
            refreshImages()
        } catch {
            log("Rename failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Delete card

    func deleteSelectedCard() {
        guard let selectedPair else { return }
        deleteCard(selectedPair)
    }

    func deleteCard(_ pair: CardPair) {
        do {
            try FileManager.default.trashItem(at: pair.front, resultingItemURL: nil)
            try FileManager.default.trashItem(at: pair.back, resultingItemURL: nil)
            metadataStore.removeMetadata(for: pair)
            log("Moved to Trash: \(pair.front.lastPathComponent) + \(pair.back.lastPathComponent)")
        } catch {
            log("Delete failed: \(error.localizedDescription)")
        }
        refreshImages()
    }

    // MARK: - Move cards

    func moveSelectedCard() {
        guard let pair = selectedPair else { return }
        movePairs([pair])
    }

    func moveAllCards() {
        movePairs(visiblePairs)
    }

    private func movePairs(_ targets: [CardPair]) {
        let dest = SettingsStore.shared.existingCardsDirectory
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
            }
        }
        log("Move complete: \(moved) moved, \(skipped) skipped → \(dest.lastPathComponent)")
        refreshImages()
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

    private func applyDirectoryIndex(_ index: CardDirectoryIndex, pruneMetadata: Bool) {
        let prevID = selectedPairID
        childDirectories = index.childDirectories
        pairs = index.pairs
        if pruneMetadata {
            metadataStore.pruneMetadata(keepingBaseNames: Set(index.pairs.map(\.baseName)))
        }
        updateVisiblePairs()

        if let prevID, pairs.contains(where: { $0.id == prevID }) {
            selectedPairID = prevID
        } else {
            selectedPairID = nil
        }

        syncSelectionWithVisiblePairs()
    }

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

}
