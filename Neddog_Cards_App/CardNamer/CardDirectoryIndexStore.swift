import Foundation

struct CardDirectoryIndex {
    let directoryURL: URL
    let childDirectories: [URL]
    let imageFileCount: Int
    let pairs: [CardPair]
}

enum CardDirectoryIndexStore {
    static let cacheThreshold = 200

    private static let cacheVersion = 1
    private static let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]

    // Cache lives outside the watched directory to prevent the atomic write
    // from triggering the kqueue watcher and causing an infinite refresh loop.
    private static func cacheURL(for directoryURL: URL) -> URL {
        let appCaches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let cacheDir = appCaches.appendingPathComponent("NeddogCards/DirectoryIndex")
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let encoded = Data(directoryURL.path.utf8).base64EncodedString()
            .replacingOccurrences(of: "+", with: ".")
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: "=", with: "")
        return cacheDir.appendingPathComponent(encoded + ".cache")
    }

    static func cachedIndex(for directoryURL: URL) -> CardDirectoryIndex? {
        let directoryURL = directoryURL.standardized
        let cacheURL = cacheURL(for: directoryURL)
        guard FileManager.default.fileExists(atPath: cacheURL.path) else { return nil }

        do {
            let data = try Data(contentsOf: cacheURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let cache = try decoder.decode(CardDirectoryCacheFile.self, from: data)
            guard cache.version == cacheVersion else { return nil }

            let pairs = cache.pairs.map { p -> CardPair in
                let fd = p.frontModificationDate ?? .distantPast
                let bd = p.backModificationDate ?? .distantPast
                return CardPair(
                    front: directoryURL.appendingPathComponent(p.front),
                    back: directoryURL.appendingPathComponent(p.back),
                    modificationDate: max(fd, bd)
                )
            }
            let childDirectories = cache.childDirectoryNames.map {
                directoryURL.appendingPathComponent($0)
            }
            return CardDirectoryIndex(
                directoryURL: directoryURL,
                childDirectories: childDirectories,
                imageFileCount: cache.imageFileCount,
                pairs: pairs
            )
        } catch {
            return nil
        }
    }

    static func scanDirectory(_ directoryURL: URL) -> CardDirectoryIndex {
        let directoryURL = directoryURL.standardized
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        let childDirectories = contents.filter { url in
            (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }.sorted {
            naturalSortLess($0.lastPathComponent, $1.lastPathComponent)
        }

        let files = contents
            .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { naturalSortLess($0.lastPathComponent, $1.lastPathComponent) }

        return CardDirectoryIndex(
            directoryURL: directoryURL,
            childDirectories: childDirectories,
            imageFileCount: files.count,
            pairs: buildPairs(from: files)
        )
    }

    static func saveCacheIfNeeded(for index: CardDirectoryIndex) {
        let cacheURL = cacheURL(for: index.directoryURL)
        guard index.imageFileCount > cacheThreshold || FileManager.default.fileExists(atPath: cacheURL.path) else {
            return
        }

        let newPairFronts = index.pairs.map { $0.front.lastPathComponent }
        let newChildNames = index.childDirectories.map(\.lastPathComponent).sorted()

        if let existing = try? Data(contentsOf: cacheURL),
           let existingCache = try? JSONDecoder().decode(CardDirectoryCacheFile.self, from: existing),
           existingCache.version == cacheVersion,
           existingCache.imageFileCount == index.imageFileCount,
           existingCache.pairs.map(\.front) == newPairFronts,
           existingCache.childDirectoryNames.sorted() == newChildNames {
            return
        }

        let cache = CardDirectoryCacheFile(
            version: cacheVersion,
            generatedAt: Date(),
            imageFileCount: index.imageFileCount,
            pairs: index.pairs.map { pair in
                CardDirectoryCachePair(
                    front: pair.front.lastPathComponent,
                    back: pair.back.lastPathComponent,
                    frontSize: fileSize(pair.front),
                    backSize: fileSize(pair.back),
                    frontModificationDate: modificationDate(pair.front),
                    backModificationDate: modificationDate(pair.back)
                )
            },
            childDirectoryNames: index.childDirectories.map(\.lastPathComponent)
        )

        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(cache)
            try data.write(to: cacheURL, options: .atomic)
        } catch {
            // The cache is only a performance optimization; scan results remain usable.
        }
    }

    static func naturalSortLess(_ a: String, _ b: String) -> Bool {
        a.localizedStandardCompare(b) == .orderedAscending
    }

    private static func buildPairs(from files: [URL]) -> [CardPair] {
        let isBack = { (url: URL) -> Bool in
            url.deletingPathExtension().lastPathComponent.lowercased().hasSuffix("_b")
        }

        var fronts: [String: [URL]] = [:]
        for file in files where !isBack(file) {
            let stem = file.deletingPathExtension().lastPathComponent.lowercased()
            fronts[stem, default: []].append(file)
        }

        var used = Set<URL>()
        var pairs: [CardPair] = []
        for back in files.filter(isBack) {
            let backStem = back.deletingPathExtension().lastPathComponent.lowercased()
            let frontStem = String(backStem.dropLast(2))
            if let candidates = fronts[frontStem], let front = candidates.first(where: { !used.contains($0) }) {
                pairs.append(CardPair(front: front, back: back))
                used.insert(front)
                used.insert(back)
            }
        }

        let stemRegex = try? NSRegularExpression(pattern: #"_\d+$"#)
        let fallbackStem = { (url: URL) -> String in
            let base = url.deletingPathExtension().lastPathComponent
            let range = NSRange(base.startIndex..., in: base)
            if let stemRegex,
               let match = stemRegex.firstMatch(in: base, range: range),
               let matchRange = Range(match.range, in: base) {
                return String(base[..<matchRange.lowerBound])
            }
            return base
        }

        let remaining = files.filter { !used.contains($0) }
        var groups: [String: [URL]] = [:]
        var orderedStems: [String] = []
        for file in remaining {
            let stem = fallbackStem(file)
            if groups[stem] == nil { orderedStems.append(stem) }
            groups[stem, default: []].append(file)
        }

        for stem in orderedStems {
            let group = groups[stem] ?? []
            for index in stride(from: 0, to: group.count - 1, by: 2) {
                var front = group[index]
                var back = group[index + 1]
                if isBack(front) && !isBack(back) { swap(&front, &back) }
                pairs.append(CardPair(front: front, back: back))
            }
        }

        return pairs.sorted {
            let m0 = modificationDate($0.front) ?? .distantPast
            let m1 = modificationDate($1.front) ?? .distantPast
            return m0 > m1
        }
    }

    private static func fileSize(_ url: URL) -> Int? {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize)
    }

    private static func modificationDate(_ url: URL) -> Date? {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
    }
}

private struct CardDirectoryCacheFile: Codable {
    var version: Int
    var generatedAt: Date
    var imageFileCount: Int
    var pairs: [CardDirectoryCachePair]
    var childDirectoryNames: [String] = []
}

private struct CardDirectoryCachePair: Codable {
    var front: String
    var back: String
    var frontSize: Int?
    var backSize: Int?
    var frontModificationDate: Date?
    var backModificationDate: Date?
}
