import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Combines card fronts into one grid image, the way `card_merge.py` and the
/// `card_merge` Rust binary do.
///
/// The layout below deliberately mirrors those tools so a merge made here looks
/// like one made from the shell. It is reimplemented rather than shelled out to
/// because the CLI cannot serve this caller: it names its output `merged001.jpg`
/// in the working directory with no way to ask for another name, it rewrites the
/// modification time of every input file, and — decisively — it silently drops
/// any file whose name does not end in digits before `.jpg`, which would quietly
/// omit cards like `2026-ONeal-Panini-90s-Pop-Culture-AW-SN.jpg` from the grid.
enum CardMergeService {

    /// Fewer than two images is not a merge.
    static let minimumImages = 2
    /// The grid stays legible up to two rows of four.
    static let maximumImages = 8

    enum MergeError: LocalizedError {
        case tooFew(Int)
        case tooMany(Int)
        case unreadable(URL)
        case mismatchedSides
        case renderFailed

        var errorDescription: String? {
            switch self {
            case .tooFew(let n):
                "Select at least \(CardMergeService.minimumImages) cards to merge (got \(n))."
            case .tooMany(let n):
                "Select at most \(CardMergeService.maximumImages) cards to merge (got \(n))."
            case .unreadable(let url):
                "Could not read \(url.lastPathComponent)."
            case .mismatchedSides:
                "Each front needs a matching back to merge."
            case .renderFailed:
                "Could not render the merged image."
            }
        }
    }

    /// Merges the fronts into one grid and the matching backs into another,
    /// written side by side as `<name>.jpg` and `<name>_b.jpg`.
    ///
    /// The back companion is not decoration: a lone merged file has no partner,
    /// so `CardDirectoryIndexStore.buildPairs` drops it and it never appears in
    /// the app. Naming the pair explicitly also gets it matched by the `_b` pass
    /// that runs before the adjacency fallback, which keeps two merges from the
    /// same day from pairing with each other.
    @discardableResult
    static func merge(fronts: [URL], backs: [URL], in directory: URL) throws -> (front: URL, back: URL) {
        guard fronts.count >= minimumImages else { throw MergeError.tooFew(fronts.count) }
        guard fronts.count <= maximumImages else { throw MergeError.tooMany(fronts.count) }
        guard backs.count == fronts.count else { throw MergeError.mismatchedSides }

        // Render both before writing either, so a failure never leaves a lone
        // front on disk — the exact state this pairing is meant to avoid.
        let frontImage = try renderGrid(from: fronts)
        let backImage = try renderGrid(from: backs)

        let output = outputURLs(for: fronts[0], count: fronts.count, in: directory)
        let backExisted = FileManager.default.fileExists(atPath: output.back.path)
        try write(frontImage, to: output.front)
        do {
            try write(backImage, to: output.back)
        } catch {
            // Don't strand an unpairable front if the second write fails. Only
            // clean up a front this call created; a pre-existing pair is left
            // as it was rather than half-deleted.
            if !backExisted { try? FileManager.default.removeItem(at: output.front) }
            throw error
        }
        return output
    }

    /// Lays the images out and draws them into one grid image.
    private static func renderGrid(from urls: [URL]) throws -> CGImage {
        let images = try urls.map { url -> CGImage in
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw MergeError.unreadable(url)
            }
            return image
        }

        // Scanner output can vary by a few pixels. Size each grid cell to the
        // largest scan so no card edge is clipped — same as the CLI.
        let cellWidth = images.map(\.width).max() ?? 0
        let cellHeight = images.map(\.height).max() ?? 0
        guard cellWidth > 0, cellHeight > 0 else { throw MergeError.renderFailed }

        let plan = layout(count: images.count, cellWidth: cellWidth, cellHeight: cellHeight)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil,
                width: plan.canvasWidth,
                height: plan.canvasHeight,
                bitsPerComponent: 8,
                bytesPerRow: 0,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ) else {
            throw MergeError.renderFailed
        }

        // The 5-up layout leaves gaps beside the centred bottom row; the CLI
        // fills those with the average edge colour of the scans so the seam is
        // less obvious. Every other layout keeps the default black.
        if plan.fillsBackground {
            let fill = averagePerimeterColor(of: images)
            context.setFillColor(red: fill.red, green: fill.green, blue: fill.blue, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: plan.canvasWidth, height: plan.canvasHeight))
        }

        for (image, topLeft) in zip(images, plan.positions) {
            // Core Graphics draws from a bottom-left origin, but the layout is
            // expressed in top-left terms like the CLI's, so flip the y here.
            let rect = CGRect(
                x: topLeft.x,
                y: CGFloat(plan.canvasHeight) - topLeft.y - CGFloat(image.height),
                width: CGFloat(image.width),
                height: CGFloat(image.height)
            )
            context.draw(image, in: rect)
        }

        guard let merged = context.makeImage() else { throw MergeError.renderFailed }
        return merged
    }

    private static func write(_ image: CGImage, to url: URL) throws {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data, UTType.jpeg.identifier as CFString, 1, nil
        ) else {
            throw MergeError.renderFailed
        }
        CGImageDestinationAddImage(
            destination, image, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary
        )
        guard CGImageDestinationFinalize(destination) else { throw MergeError.renderFailed }
        try (data as Data).write(to: url, options: .atomic)
    }

    /// `merge_<YYYYMMDD>_<number of cards>.jpg` plus its `_b` companion, dated
    /// from the first image.
    ///
    /// The name is deterministic and an existing pair is overwritten: re-merging
    /// after one of the source cards is rotated or rescanned should refresh the
    /// merge in place, not leave a stale original beside a suffixed copy. The
    /// trade-off is that two *different* merges of the same card count on the
    /// same scan date share a name, so the second replaces the first.
    static func outputURLs(for firstFront: URL, count: Int, in directory: URL) -> (front: URL, back: URL) {
        let base = "merge_\(dateStamp(for: firstFront))_\(count)"
        return (
            front: directory.appendingPathComponent("\(base).jpg"),
            back: directory.appendingPathComponent("\(base)_b.jpg")
        )
    }

    // MARK: - Date stamp

    /// The date to name a merge after, preferring what the scan itself records.
    /// Filenames are the most reliable source here because the scanner writes
    /// the date into them (`card_20260919_0005.jpg`); EXIF and the file's own
    /// modification date are fallbacks for cards renamed out of that pattern.
    static func dateStamp(for url: URL, now: Date = Date()) -> String {
        if let fromName = dateInFilename(url) { return fromName }
        if let fromExif = captureDate(of: url) { return format(fromExif) }

        let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate
        return format(modified ?? now)
    }

    /// First plausible YYYYMMDD run of digits in the filename. A card named
    /// `2023-Murray-Panini-Select-166` has a year but no such run, and correctly
    /// falls through to the other sources.
    private static func dateInFilename(_ url: URL) -> String? {
        let stem = url.deletingPathExtension().lastPathComponent
        let digits = Array(stem)
        var index = 0

        while index + 8 <= digits.count {
            let window = digits[index..<(index + 8)]
            if window.allSatisfy(\.isNumber) {
                let text = String(window)
                let isBoundedLeft = index == 0 || !digits[index - 1].isNumber
                let isBoundedRight = index + 8 == digits.count || !digits[index + 8].isNumber
                if isBoundedLeft, isBoundedRight, isPlausibleDate(text) {
                    return text
                }
            }
            index += 1
        }
        return nil
    }

    private static func isPlausibleDate(_ yyyymmdd: String) -> Bool {
        guard yyyymmdd.count == 8,
              let year = Int(yyyymmdd.prefix(4)),
              let month = Int(yyyymmdd.dropFirst(4).prefix(2)),
              let day = Int(yyyymmdd.suffix(2)) else { return false }
        return (1900...2999).contains(year) && (1...12).contains(month) && (1...31).contains(day)
    }

    private static func captureDate(of url: URL) -> Date? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        else { return nil }

        let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any]
        let tiff = properties[kCGImagePropertyTIFFDictionary] as? [CFString: Any]
        let raw = (exif?[kCGImagePropertyExifDateTimeOriginal] as? String)
            ?? (exif?[kCGImagePropertyExifDateTimeDigitized] as? String)
            ?? (tiff?[kCGImagePropertyTIFFDateTime] as? String)
        guard let raw else { return nil }

        let parser = DateFormatter()
        parser.locale = Locale(identifier: "en_US_POSIX")
        parser.dateFormat = "yyyy:MM:dd HH:mm:ss"
        return parser.date(from: raw)
    }

    private static func format(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd"
        return formatter.string(from: date)
    }

    // MARK: - Layout

    struct Plan {
        let canvasWidth: Int
        let canvasHeight: Int
        /// Top-left corner of each image, in the order supplied.
        let positions: [CGPoint]
        let fillsBackground: Bool
    }

    /// Mirrors `merge_images` in card_merge.py / main.rs, including the special
    /// cases for 2, 3 and 5 images.
    static func layout(count: Int, cellWidth: Int, cellHeight: Int) -> Plan {
        let w = CGFloat(cellWidth)
        let h = CGFloat(cellHeight)

        switch count {
        case 2, 3:
            // A single row.
            return Plan(
                canvasWidth: cellWidth * count,
                canvasHeight: cellHeight,
                positions: (0..<count).map { CGPoint(x: CGFloat($0) * w, y: 0) },
                fillsBackground: false
            )

        case 5:
            // Three across the top, two centred beneath them.
            var positions = (0..<3).map { CGPoint(x: CGFloat($0) * w, y: 0) }
            positions += (0..<2).map { CGPoint(x: CGFloat($0) * w + w / 2, y: h) }
            return Plan(
                canvasWidth: cellWidth * 3,
                canvasHeight: cellHeight * 2,
                positions: positions,
                fillsBackground: true
            )

        default:
            // Two rows, filled left to right. `rounded()` matches Rust's
            // `f64::round` and Python's behaviour for these counts, so 7 cards
            // lay out as 4 columns with the last slot left empty.
            let columns = max(1, Int((Double(count) / 2.0).rounded()))
            let positions = (0..<count).map { index in
                CGPoint(
                    x: CGFloat(index % columns) * w,
                    y: CGFloat(index / columns) * h
                )
            }
            return Plan(
                canvasWidth: cellWidth * columns,
                canvasHeight: cellHeight * 2,
                positions: positions,
                fillsBackground: false
            )
        }
    }

    // MARK: - Fill colour

    /// Average colour of every image's outer edge. Corners are counted twice,
    /// matching the CLI implementations rather than "correcting" them.
    private static func averagePerimeterColor(of images: [CGImage]) -> (red: CGFloat, green: CGFloat, blue: CGFloat) {
        var totals = (r: 0.0, g: 0.0, b: 0.0)
        var samples = 0.0

        for image in images {
            let width = image.width
            let height = image.height
            guard width > 0, height > 0,
                  let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { continue }

            let bytesPerRow = width * 4
            var buffer = [UInt8](repeating: 0, count: bytesPerRow * height)
            let drawn = buffer.withUnsafeMutableBytes { raw -> Bool in
                guard let context = CGContext(
                    data: raw.baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: bytesPerRow,
                    space: colorSpace,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                ) else { return false }
                context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
                return true
            }
            guard drawn else { continue }

            func sample(x: Int, y: Int) {
                let offset = y * bytesPerRow + x * 4
                totals.r += Double(buffer[offset])
                totals.g += Double(buffer[offset + 1])
                totals.b += Double(buffer[offset + 2])
                samples += 1
            }

            for x in 0..<width {
                sample(x: x, y: 0)
                sample(x: x, y: height - 1)
            }
            for y in 0..<height {
                sample(x: 0, y: y)
                sample(x: width - 1, y: y)
            }
        }

        guard samples > 0 else { return (0, 0, 0) }
        return (
            red: CGFloat(totals.r / samples / 255.0),
            green: CGFloat(totals.g / samples / 255.0),
            blue: CGFloat(totals.b / samples / 255.0)
        )
    }
}
