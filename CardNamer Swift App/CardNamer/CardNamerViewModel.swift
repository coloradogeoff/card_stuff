import Foundation
import AppKit
import Observation

@Observable
final class CardNamerViewModel {

    // Directory state
    var directoryPath: String = SettingsStore.shared.quickDirectories.first?.path ?? "/Users/geoff/incoming cards"
    var pairs: [CardPair] = []
    var selectedPairID: CardPair.ID? {
        didSet { proposedName = selectedPair?.baseName ?? "" }
    }

    // Naming
    var proposedName: String = ""
    var isBusy: Bool = false

    // Log
    var logText: String = ""

    // Preview
    var showingBack: Bool = false

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
    }

    // MARK: - Computed

    var currentDirectory: URL {
        URL(fileURLWithPath: directoryPath).standardized
    }

    var selectedPair: CardPair? {
        pairs.first { $0.id == selectedPairID }
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

    func refreshImages(silent: Bool = false) {
        let dir = currentDirectory
        guard FileManager.default.fileExists(atPath: dir.path) else {
            if !silent { log("Directory does not exist: \(dir.path)") }
            pairs = []
            return
        }

        let imageExts: Set<String> = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey]))?.filter {
            imageExts.contains($0.pathExtension.lowercased())
        }.sorted {
            naturalSortLess($0.lastPathComponent, $1.lastPathComponent)
        } ?? []

        let newPairs = buildPairs(from: files)

        let prevID = selectedPairID
        pairs = newPairs

        // Try to keep selection on same pair (by file names)
        if let prev = prevID, pairs.contains(where: { $0.id == prev }) {
            // still exists — keep selection
        } else if !pairs.isEmpty {
            selectedPairID = pairs[0].id
            showingBack = false
        }

        startWatching()

        if !silent {
            if files.isEmpty {
                log("No image files found.")
            } else {
                log("Loaded \(newPairs.count) card pair(s) from \(dir.lastPathComponent).")
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

        let (front, back, bottom) = try await (ocrFront, ocrBack, ocrBackBottom)
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
            log("Renamed to \(newFront.lastPathComponent) and \(newBack.lastPathComponent)")
            proposedName = ""
            refreshImages()
        } catch {
            log("Rename failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Delete card

    func deleteCard(_ pair: CardPair) {
        do {
            try FileManager.default.trashItem(at: pair.front, resultingItemURL: nil)
            try FileManager.default.trashItem(at: pair.back, resultingItemURL: nil)
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
        movePairs(pairs)
    }

    private func movePairs(_ targets: [CardPair]) {
        let dest = SettingsStore.shared.existingCardsDirectory
        try? FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
        var moved = 0
        var skipped = 0
        for pair in targets {
            for source in [pair.front, pair.back] {
                let target = dest.appendingPathComponent(source.lastPathComponent)
                if source.standardized == target.standardized { skipped += 1; continue }
                do {
                    if FileManager.default.fileExists(atPath: target.path) {
                        try FileManager.default.removeItem(at: target)
                    }
                    try FileManager.default.moveItem(at: source, to: target)
                    moved += 1
                } catch {
                    log("Move failed for \(source.lastPathComponent): \(error.localizedDescription)")
                    skipped += 1
                }
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

    // MARK: - Pair building

    private func buildPairs(from files: [URL]) -> [CardPair] {
        let isBack = { (url: URL) -> Bool in url.deletingPathExtension().lastPathComponent.lowercased().hasSuffix("_b") }

        var fronts: [String: [URL]] = [:]
        for f in files where !isBack(f) {
            let stem = f.deletingPathExtension().lastPathComponent.lowercased()
            fronts[stem, default: []].append(f)
        }
        let backs = files.filter(isBack)

        var used = Set<URL>()
        var pairs: [CardPair] = []

        for back in backs {
            let backStem = back.deletingPathExtension().lastPathComponent.lowercased()
            let frontStem = String(backStem.dropLast(2))
            if let candidates = fronts[frontStem], let front = candidates.first(where: { !used.contains($0) }) {
                pairs.append(CardPair(front: front, back: back))
                used.insert(front); used.insert(back)
            }
        }

        let remaining = files.filter { !used.contains($0) }
        for i in stride(from: 0, to: remaining.count - 1, by: 2) {
            var first = remaining[i], second = remaining[i + 1]
            if isBack(first) && !isBack(second) { swap(&first, &second) }
            pairs.append(CardPair(front: first, back: second))
        }

        return pairs.sorted {
            let m0 = (try? $0.front.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? Date.distantPast
            let m1 = (try? $1.front.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? Date.distantPast
            return m0 > m1
        }
    }

    private func naturalSortLess(_ a: String, _ b: String) -> Bool {
        let re = try? NSRegularExpression(pattern: #"(\d+)|(\D+)"#)
        func tokens(_ s: String) -> [(Int?, String)] {
            let range = NSRange(s.startIndex..., in: s)
            return (re?.matches(in: s, range: range) ?? []).compactMap { m -> (Int?, String)? in
                guard let r = Range(m.range, in: s) else { return nil }
                let t = String(s[r])
                return (Int(t), t)
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
