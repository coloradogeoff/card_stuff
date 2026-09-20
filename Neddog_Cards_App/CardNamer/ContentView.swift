import SwiftUI
import AppKit

enum AppMode: String, CaseIterable {
    case cardNamer   = "Card Namer"
    case ebayTitles  = "eBay Titles"
}

struct ContentView: View {
    @Environment(\.openWindow) private var openWindow

    let cardVM: CardNamerViewModel
    let ebayVM: EbayTitlesViewModel
    @State private var showSettings = false
    @State private var appMode: AppMode = .cardNamer
    @State private var showDeleteCardConfirmation = false
    @State private var showPSAPrompt = false
    @State private var psaCertNumber = ""

    var body: some View {
        NavigationSplitView(columnVisibility: .constant(.all)) {
            sidebar
                .toolbar {
                    ToolbarItemGroup(placement: .primaryAction) {
                        directoryToolbarButtons
                    }
                }
        } detail: {
            detail
                .navigationTitle("")
        }
        .navigationSplitViewStyle(.balanced)
        .sheet(isPresented: $showSettings) { SettingsView() }
        .sheet(isPresented: $showPSAPrompt) {
            PSACertPromptSheet(
                certNumber: $psaCertNumber,
                onCancel: {
                    psaCertNumber = ""
                    showPSAPrompt = false
                },
                onDownload: {
                    let cert = psaCertNumber
                    psaCertNumber = ""
                    showPSAPrompt = false
                    downloadPSACard(certNumber: cert)
                }
            )
        }
        .toolbar {
            ToolbarItem(placement: .navigation) { psaButton }
            ToolbarItem(placement: .navigation) { rotateButton }
            ToolbarItem(placement: .navigation) { deleteButton }
            ToolbarItem(placement: .principal) { modePicker }
            ToolbarItem(placement: .primaryAction) {
                Button { showSettings = true } label: { Image(systemName: "gear") }
                    .help("OpenAI / PSA Settings")
            }
        }
        .confirmationDialog(
            activeDeleteMessage,
            isPresented: $showDeleteCardConfirmation,
            titleVisibility: .visible
        ) {
            Button("Move to Trash", role: .destructive) {
                deleteSelectedCard()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Both the front and back image files for the selected card will be moved to the Trash.")
        }
        .onReceive(NotificationCenter.default.publisher(for: .showEbayResultsWindow)) { _ in
            openWindow(id: SceneID.ebayResults)
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            let previewURL = appMode == .cardNamer ? cardVM.previewURL : ebayVM.previewURL
            if let url = previewURL {
                CardPreviewView.invalidateCache(for: url)
            }
            if appMode == .cardNamer { cardVM.previewRevision += 1 }
            else { ebayVM.previewRevision += 1 }
        }
    }

    @ViewBuilder
    private var sidebar: some View {
        if appMode == .cardNamer {
            CardNamerSidebar(vm: cardVM)
        } else {
            EbayTitlesSidebar(vm: ebayVM)
        }
    }

    @ViewBuilder
    private var detail: some View {
        if appMode == .cardNamer {
            CardNamerDetail(vm: cardVM)
        } else {
            EbayTitlesDetail(vm: ebayVM)
        }
    }

    // MARK: - Toolbar items

    private var modePicker: some View {
        Picker("", selection: $appMode) {
            ForEach(AppMode.allCases, id: \.self) { mode in
                Text(mode.rawValue).tag(mode)
            }
        }
        .pickerStyle(.segmented)
        .frame(width: 220)
    }

    private var currentDirectoryPath: String {
        appMode == .cardNamer ? cardVM.directoryPath : ebayVM.directoryPath
    }

    private func switchDirectory(_ dir: QuickDirectory) {
        if appMode == .cardNamer { cardVM.switchTo(dir) }
        else { ebayVM.switchTo(dir) }
    }

    private func chooseDirectory() {
        if appMode == .cardNamer { cardVM.chooseDirectory() }
        else { ebayVM.chooseDirectory() }
    }

    private func refreshImages() {
        if appMode == .cardNamer { cardVM.refreshImages() }
        else { ebayVM.refreshImages() }
    }

    private func rotatePreviewImageClockwise() {
        if appMode == .cardNamer { cardVM.rotatePreviewImageClockwise() }
        else { ebayVM.rotatePreviewImageClockwise() }
    }

    private func downloadPSACard(certNumber: String) {
        if appMode == .cardNamer { cardVM.downloadPSACard(certNumber: certNumber) }
        else { ebayVM.downloadPSACard(certNumber: certNumber) }
    }

    private func deleteSelectedCard() {
        if appMode == .cardNamer { cardVM.deleteSelectedCard() }
        else { ebayVM.deleteSelectedCard() }
    }

    private var canRotatePreviewImage: Bool {
        if appMode == .cardNamer {
            return cardVM.previewURL != nil && !cardVM.isBusy
        }
        return ebayVM.previewURL != nil && !ebayVM.isBusy
    }

    private var canDeleteSelectedCard: Bool {
        if appMode == .cardNamer {
            return !cardVM.selectedIDs.isEmpty && !cardVM.isBusy
        }
        return !ebayVM.selectedIDs.isEmpty && !ebayVM.isBusy
    }

    private var canDownloadPSA: Bool {
        if appMode == .cardNamer {
            return !cardVM.isBusy
        }
        return !ebayVM.isBusy
    }

    private var activeDeleteMessage: String {
        let n = appMode == .cardNamer ? cardVM.selectedIDs.count : ebayVM.selectedIDs.count
        return n > 1 ? "Move \(n) cards to the Trash?" : "Move selected card to the Trash?"
    }

    @ViewBuilder
    private var directoryToolbarButtons: some View {
        let dirs = SettingsStore.shared.quickDirectories
        Button("Incoming") { switchDirectory(dirs[0]) }
            .disabled(!dirs[0].isAvailable)
            .help("Incoming cards  (\(dirs[0].path))  ⌘1")
        Button("Collection") { switchDirectory(dirs[1]) }
            .disabled(!dirs[1].isAvailable)
            .help("Collection  (\(dirs[1].path))  ⌘2")
        Button("Sales") { switchDirectory(dirs[2]) }
            .disabled(!dirs[2].isAvailable)
            .help("Current sales folder  (\(dirs[2].path))  ⌘3")
        Button("Browse…") { chooseDirectory() }
            .help("Choose a custom folder")
    }

    private var rotateButton: some View {
        Button {
            rotatePreviewImageClockwise()
        } label: {
            Label("Rotate", systemImage: "rotate.right")
        }
        .disabled(!canRotatePreviewImage)
        .help("Rotate the currently displayed card image 90 degrees clockwise")
    }

    private var psaButton: some View {
        Button {
            showPSAPrompt = true
        } label: {
            Label("PSA", systemImage: "arrow.down.circle")
        }
        .disabled(!canDownloadPSA)
        .help("Download card images from PSA by certification number")
    }

    private var deleteButton: some View {
        Button(role: .destructive) {
            showDeleteCardConfirmation = true
        } label: {
            Label("Delete Card", systemImage: "trash")
        }
        .disabled(!canDeleteSelectedCard)
        .help("Delete the selected card's front and back image files")
    }

}

// MARK: - Trait controls

struct TraitFilterBar: View {
    let selectedTraits: Set<CardTrait>
    let onToggle: (CardTrait) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Trait filters")
                .font(.caption2)
                .foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 86), spacing: 6)], alignment: .leading, spacing: 6) {
                ForEach(CardTrait.allCases) { trait in
                    TraitChip(
                        trait: trait,
                        isOn: selectedTraits.contains(trait),
                        isDisabled: false
                    ) {
                        onToggle(trait)
                    }
                }
            }
        }
    }
}

struct CardTraitEditor: View {
    let title: String
    let traits: CardTraits
    let isDisabled: Bool
    let onToggle: (CardTrait) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 104), spacing: 6)], alignment: .leading, spacing: 6) {
                ForEach(CardTrait.allCases) { trait in
                    TraitChip(
                        trait: trait,
                        isOn: traits.contains(trait),
                        isDisabled: isDisabled
                    ) {
                        onToggle(trait)
                    }
                }
            }
        }
    }
}

struct TraitChip: View {
    let trait: CardTrait
    let isOn: Bool
    let isDisabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(trait.label, systemImage: isOn ? "checkmark.circle.fill" : trait.systemImage)
                .font(.caption)
                .lineLimit(1)
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .tint(isOn ? .accentColor : .secondary)
        .disabled(isDisabled)
    }
}

// MARK: - Sidebar search/filter cluster

/// Shared control language for the sidebar's search field, filter group, and
/// clear button so they read as one cluster instead of stacked widgets.
private enum FilterControlMetrics {
    static let height: CGFloat = 26
    static let radius: CGFloat = 8
}

/// Native-style search field: leading magnifying glass, pill shape.
struct SidebarSearchField: View {
    @Binding var text: String
    let placeholder: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
        }
        .padding(.horizontal, 10)
        .frame(height: FilterControlMetrics.height)
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(Capsule())
        .overlay {
            Capsule().strokeBorder(Color.secondary.opacity(0.25))
        }
    }
}

/// Player / Year / Set filters as one bordered group with hairline dividers,
/// instead of three separate boxes.
struct SidebarFilterFieldGroup: View {
    @Binding var player: String
    @Binding var year: String
    @Binding var set: String

    var body: some View {
        HStack(spacing: 0) {
            TextField("Player", text: $player)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .padding(.horizontal, 8)
                .frame(maxWidth: .infinity)

            Divider().frame(height: 14)

            TextField("Year", text: $year)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 4)
                .frame(width: 40)

            Divider().frame(height: 14)

            TextField("Set / Series", text: $set)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .padding(.horizontal, 8)
                .frame(maxWidth: .infinity)
        }
        .frame(height: FilterControlMetrics.height)
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: FilterControlMetrics.radius))
        .overlay {
            RoundedRectangle(cornerRadius: FilterControlMetrics.radius)
                .strokeBorder(Color.secondary.opacity(0.25))
        }
    }
}

/// Small round icon button matching the field height, replacing the
/// full-size "Clear" push button that used to dwarf the compact fields.
struct SidebarClearFilterButton: View {
    let isEnabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: "xmark")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(isEnabled ? Color.accentColor : Color.secondary.opacity(0.35))
                .frame(width: FilterControlMetrics.height, height: FilterControlMetrics.height)
        }
        .buttonStyle(.plain)
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(Circle())
        .overlay {
            Circle().strokeBorder(Color.secondary.opacity(0.25))
        }
        .disabled(!isEnabled)
        .help("Clear filters")
    }
}

/// Hides cards already marked as listed. Shares the filter cluster's control
/// language so it reads as part of the same row.
struct SidebarListedToggle: View {
    @Binding var isOn: Bool
    let hiddenCount: Int

    var body: some View {
        Button {
            isOn.toggle()
        } label: {
            Label(
                isOn ? "Listed Hidden (\(hiddenCount))" : "Hide Listed (\(hiddenCount))",
                systemImage: isOn ? "eye.slash.fill" : "eye.slash"
            )
            .font(.caption)
            .lineLimit(1)
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .tint(isOn ? .accentColor : .secondary)
        .disabled(hiddenCount == 0 && !isOn)
        .help(isOn ? "Show cards marked as listed" : "Hide cards marked as listed")
    }
}

// MARK: - Listing record

/// Tooltip text for a sidebar row backed by a listing record.
func listingRowTooltip(_ listing: CardListing) -> String? {
    guard !listing.isEmpty else { return nil }
    var lines: [String] = []
    if let title = listing.title { lines.append(title) }
    if listing.listed {
        if let listedAt = listing.listedAt {
            lines.append("Listed \(ListingDateFormat.medium.string(from: listedAt))")
        } else {
            lines.append("Listed")
        }
    }
    return lines.isEmpty ? nil : lines.joined(separator: "\n")
}

func listedMenuTitle(isListed: Bool, count: Int) -> String {
    let verb = isListed ? "Unmark" : "Mark"
    guard count > 1 else { return "\(verb) as Listed" }
    return "\(verb) \(count) Cards as Listed"
}

enum ListingDateFormat {
    static let medium: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter
    }()
}

/// The stored eBay title for the selected card, with when it was listed.
/// Renders nothing for a card that has no record yet, so the detail pane is
/// unchanged for unprocessed cards.
struct ListingInfoBar: View {
    /// Fixed so `CardNamerDetail`, which lays its panes out by arithmetic
    /// rather than by flexing, can reserve exactly the right space.
    static let height: CGFloat = 62

    /// eBay rejects titles longer than this.
    static let maxTitleLength = EbayTitleLimit.maxCharacters

    let listing: CardListing
    /// Card Namer shows the title read-only; the eBay Titles pane owns editing,
    /// so description.csv is only ever rewritten from one place.
    var isEditable: Bool = false
    var onCommitTitle: ((String) -> Void)? = nil

    @State private var draft: String
    @FocusState private var titleIsFocused: Bool

    init(listing: CardListing, isEditable: Bool = false, onCommitTitle: ((String) -> Void)? = nil) {
        self.listing = listing
        self.isEditable = isEditable
        self.onCommitTitle = onCommitTitle
        _draft = State(initialValue: listing.title ?? "")
    }

    private var isOverLimit: Bool { draft.count > Self.maxTitleLength }

    @ViewBuilder
    var body: some View {
        if isEditable || !listing.isEmpty {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    if isEditable {
                        TextField("eBay title", text: $draft)
                            .textFieldStyle(.plain)
                            .font(.system(size: 12))
                            .focused($titleIsFocused)
                            .onSubmit { commit() }
                            .onChange(of: titleIsFocused) { _, focused in
                                if !focused { commit() }
                            }
                    } else if let title = listing.title {
                        Text(title)
                            .font(.system(size: 12))
                            .lineLimit(2)
                            .textSelection(.enabled)
                    }

                    HStack(spacing: 8) {
                        status
                        characterCount
                    }
                }

                Spacer(minLength: 0)

                Button {
                    let title = isEditable ? draft : (listing.title ?? "")
                    guard !title.isEmpty else { return }
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(title, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless)
                .disabled(currentTitle.isEmpty)
                .help("Copy title")

                Button {
                    guard let url = CardNameBuilder.ebayURL(fromBaseName: currentTitle) else { return }
                    NSWorkspace.shared.open(url)
                } label: {
                    Image(systemName: "magnifyingglass")
                }
                .buttonStyle(.borderless)
                .disabled(currentTitle.isEmpty)
                .help("Search eBay for this title")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, minHeight: Self.height, maxHeight: Self.height, alignment: .top)
            .background(Color(nsColor: .controlBackgroundColor))
            .overlay(alignment: .top) { Divider() }
            // A title generated while this card stays selected has to reach the
            // field; the guard keeps it from stomping on text being typed.
            .onChange(of: listing) { _, newValue in
                let stored = newValue.title ?? ""
                if !titleIsFocused, stored != draft { draft = stored }
            }
        }
    }

    private var currentTitle: String {
        isEditable ? draft : (listing.title ?? "")
    }

    private func commit() {
        onCommitTitle?(draft)
    }

    @ViewBuilder
    private var characterCount: some View {
        if currentTitle.isEmpty {
            EmptyView()
        } else if isOverLimit {
            Label(
                "\(draft.count)/\(Self.maxTitleLength) · over eBay limit",
                systemImage: "exclamationmark.triangle.fill"
            )
            .font(.caption)
            .foregroundStyle(.red)
        } else {
            Text("\(currentTitle.count)/\(Self.maxTitleLength)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var status: some View {
        if listing.listed {
            Label(
                listing.listedAt.map { "Listed \(ListingDateFormat.medium.string(from: $0))" } ?? "Listed",
                systemImage: "checkmark.seal.fill"
            )
            .font(.caption)
            .foregroundStyle(.green)
        } else if listing.title != nil {
            Label("Title saved · not yet listed", systemImage: "tag")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            Label("No title yet", systemImage: "tag")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }
}

// MARK: - Sidebar folder selection

struct SidebarDirectoryMenu: View {
    let directoryPath: String
    let switchDirectory: (QuickDirectory) -> Void
    let chooseDirectory: () -> Void
    let refreshImages: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Folder")
                .font(.caption)
                .foregroundStyle(.secondary)

            Menu {
                ForEach(SettingsStore.shared.quickDirectories) { directory in
                    Button {
                        switchDirectory(directory)
                    } label: {
                        Label {
                            Text(directory.name) +
                            Text(directory.isAvailable ? "" : "  (unavailable)")
                                .foregroundStyle(.secondary)
                        } icon: {
                            Image(systemName: directory.isAvailable ? "folder.fill" : "folder.badge.questionmark")
                        }
                    }
                    .disabled(!directory.isAvailable)
                }
                Divider()
                Button("Browse…", action: chooseDirectory)
                Divider()
                Button("Refresh", action: refreshImages)
            } label: {
                HStack {
                    Label {
                        Text(URL(fileURLWithPath: directoryPath).lastPathComponent)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    } icon: {
                        Image(systemName: "folder")
                    }
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .menuStyle(.borderlessButton)
            .help(directoryPath)
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor))
    }
}

// MARK: - Card Namer Sidebar

struct CardNamerSidebar: View {
    @Bindable var vm: CardNamerViewModel
    @State private var pendingDeletePair: CardPair?

    private var hasMultipleSelectedCards: Bool {
        vm.selectedIDs.count > 1
    }

    private func contextPairs(for pair: CardPair) -> [CardPair] {
        hasMultipleSelectedCards && vm.selectedIDs.contains(pair.id) ? vm.selectedPairs : [pair]
    }

    private func copyCardNames(_ pairs: [CardPair]) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(
            pairs.map(\.displayName).joined(separator: "\n"),
            forType: .string
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            sourceHeader
            List(selection: $vm.selectedIDs) {
                if vm.parentDirectory != nil || !vm.childDirectories.isEmpty {
                    Section {
                        if vm.parentDirectory != nil {
                            Button {
                                vm.navigateToParentDirectory()
                            } label: {
                                directoryRowLabel(name: "..", icon: "arrowshape.turn.up.left")
                            }
                            .buttonStyle(.plain)
                        }

                        ForEach(vm.childDirectories, id: \.path) { directory in
                            Button {
                                vm.navigateToDirectory(directory)
                            } label: {
                                directoryRowLabel(name: directory.lastPathComponent, icon: "folder")
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }

                Section {
                    ForEach(vm.visiblePairs) { pair in
                        let isListed = vm.listedPairs.contains(pair.baseName)
                        Label {
                            HStack(spacing: 4) {
                                Text(pair.displayName)
                                    .font(.system(size: 12))
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                    .strikethrough(isListed)
                                if vm.pairsWithTraits.contains(pair.baseName) {
                                    Image(systemName: "tag.fill")
                                        .font(.system(size: 9))
                                        .foregroundStyle(.secondary)
                                }
                                if isListed {
                                    Image(systemName: "checkmark.seal.fill")
                                        .font(.system(size: 9))
                                        .foregroundStyle(.green)
                                }
                            }
                        } icon: {
                            Image(systemName: "photo.on.rectangle")
                                .foregroundStyle(.secondary)
                        }
                        .tag(pair.id)
                        .help(listingRowTooltip(vm.listing(for: pair)) ?? pair.displayName)
                        .contextMenu {
                            Button {
                                // If this pair is part of a multi-selection, name all selected; else just this pair
                                if hasMultipleSelectedCards && vm.selectedIDs.contains(pair.id) {
                                    vm.quickNameSelected()
                                } else {
                                    vm.quickName([pair])
                                }
                            } label: {
                                Label(
                                    hasMultipleSelectedCards && vm.selectedIDs.contains(pair.id)
                                        ? "Quick Name \(vm.selectedIDs.count) Cards"
                                        : "Quick Name",
                                    systemImage: "sparkles"
                                )
                            }
                            .disabled(vm.isBusy)

                            Divider()

                            Button {
                                vm.toggleListed(contextPairs(for: pair))
                            } label: {
                                Label(
                                    listedMenuTitle(
                                        isListed: vm.listing(for: pair).listed,
                                        count: contextPairs(for: pair).count
                                    ),
                                    systemImage: vm.listing(for: pair).listed ? "xmark.seal" : "checkmark.seal"
                                )
                            }

                            Divider()

                            Button {
                                // If this pair is part of a multi-selection, move all selected; else just this pair
                                if hasMultipleSelectedCards && vm.selectedIDs.contains(pair.id) {
                                    vm.moveSelectedToSales()
                                } else {
                                    vm.moveCardToSales(pair)
                                }
                            } label: {
                                Label("Move to Sales", systemImage: "cart")
                            }
                            .disabled(vm.isBusy)

                            Button {
                                if hasMultipleSelectedCards && vm.selectedIDs.contains(pair.id) {
                                    vm.moveSelectedToCollection()
                                } else {
                                    vm.moveCardToCollection(pair)
                                }
                            } label: {
                                Label("Move to Collection", systemImage: "archivebox")
                            }
                            .disabled(vm.isBusy)

                            Divider()

                            Button {
                                if let gcURL = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.lemkesoft.graphicconverter12") {
                                    NSWorkspace.shared.open([pair.front], withApplicationAt: gcURL, configuration: NSWorkspace.OpenConfiguration())
                                }
                            } label: {
                                Label("Open in GraphicConverter", systemImage: "photo")
                            }
                            .disabled(hasMultipleSelectedCards)

                            Button {
                                if let image = NSImage(contentsOf: pair.front) {
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.writeObjects([image])
                                }
                            } label: {
                                Label("Copy to Clipboard", systemImage: "doc.on.clipboard")
                            }
                            .disabled(hasMultipleSelectedCards)

                            Button {
                                copyCardNames(contextPairs(for: pair))
                            } label: {
                                Label("Copy Card Names", systemImage: "list.clipboard")
                            }

                            Divider()

                            Button(role: .destructive) {
                                pendingDeletePair = pair
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            .confirmationDialog(
                pendingDeletePair.map { pair in
                    vm.selectedIDs.count > 1 && vm.selectedIDs.contains(pair.id)
                        ? "Delete \(vm.selectedIDs.count) cards?"
                        : "Delete \(pair.displayName)?"
                } ?? "",
                isPresented: Binding(get: { pendingDeletePair != nil }, set: { if !$0 { pendingDeletePair = nil } }),
                titleVisibility: .visible
            ) {
                if let pair = pendingDeletePair {
                    Button("Move to Trash", role: .destructive) {
                        if vm.selectedIDs.count > 1 && vm.selectedIDs.contains(pair.id) {
                            vm.deleteCards(vm.selectedPairs)
                        } else {
                            vm.deleteCard(pair)
                        }
                        pendingDeletePair = nil
                    }
                }
                Button("Cancel", role: .cancel) { pendingDeletePair = nil }
            } message: {
                Text("Both the front and back image files will be moved to the Trash.")
            }

            if !vm.visiblePairs.isEmpty && !vm.isExistingCardsDirectory {
                Divider()
                HStack(spacing: 8) {
                    Button {
                        vm.moveSelectedCard()
                    } label: {
                        Label("Move Card", systemImage: "arrow.right.square")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(vm.selectedPair == nil || vm.isBusy)

                    Button {
                        vm.moveAllCards()
                    } label: {
                        Label(vm.hasActiveFilter ? "Move Results" : "Move All", systemImage: "arrow.right.square.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(vm.isBusy)
                }
                .buttonStyle(.borderless)
                .controlSize(.small)
                .padding(10)
            }
        }
        .navigationSplitViewColumnWidth(min: 240, ideal: 300, max: 420)
    }

    private func directoryRowLabel(name: String, icon: String) -> some View {
        Label {
            Text(name)
                .font(.system(size: 12))
                .lineLimit(1)
                .truncationMode(.middle)
        } icon: {
            Image(systemName: icon)
                .foregroundStyle(.blue)
        }
    }

    private var sourceHeader: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Source")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(URL(fileURLWithPath: vm.directoryPath).lastPathComponent)
                .font(.headline)
                .lineLimit(1)
                .truncationMode(.middle)
            Text(vm.directoryPath)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .truncationMode(.middle)
            SidebarSearchField(text: $vm.filterText, placeholder: "Search filenames")
                .padding(.top, 8)
            HStack(spacing: 6) {
                SidebarFilterFieldGroup(player: $vm.filterPlayer, year: $vm.filterYear, set: $vm.filterSet)
                SidebarClearFilterButton(isEnabled: vm.hasActiveFilter) {
                    vm.clearFilters()
                }
            }
            .padding(.top, 6)
            HStack(spacing: 8) {
                Picker("Sort By", selection: $vm.sortField) {
                    ForEach(CardPairSortField.allCases) { field in
                        Text(field.rawValue).tag(field)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)

                Picker("Sort Order", selection: $vm.sortOrder) {
                    ForEach(CardPairSortOrder.allCases) { order in
                        Text(order.rawValue).tag(order)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)

                Text("\(vm.visiblePairs.count)/\(vm.pairs.count)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            .controlSize(.small)
            .padding(.top, 8)
            HStack(spacing: 8) {
                SidebarListedToggle(isOn: $vm.hideListed, hiddenCount: vm.hiddenListedCount)
                Spacer(minLength: 0)
            }
            .padding(.top, 6)
            TraitFilterBar(
                selectedTraits: vm.selectedTraitFilters,
                onToggle: vm.toggleTraitFilter
            )
            .padding(.top, 8)
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
        .padding(.bottom, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor))
        .overlay(alignment: .bottom) { Divider() }
    }
}

// MARK: - Card Namer Detail

struct CardNamerDetail: View {
    private enum Layout {
        static let actionHeight: CGFloat = 282
        static let minimumPreviewHeight: CGFloat = 240
        static let previewBadgeBottomPadding: CGFloat = 28
    }

    @Bindable var vm: CardNamerViewModel

    private var showsListingBar: Bool {
        vm.selectedIDs.count <= 1 && !vm.selectedListing.isEmpty
    }

    var body: some View {
        GeometryReader { proxy in
            let listingBarHeight = showsListingBar ? ListingInfoBar.height : 0
            let previewHeight = max(
                proxy.size.height - Layout.actionHeight - listingBarHeight,
                Layout.minimumPreviewHeight
            )

            VStack(spacing: 0) {
                Color(nsColor: .windowBackgroundColor)
                    .overlay {
                        if vm.selectedIDs.count > 1 {
                            MultiSelectionThumbnailGrid(pairs: vm.selectedPairs)
                        } else {
                            CardPreviewView(imageURL: vm.previewURL, reloadID: vm.previewRevision)
                                .onTapGesture { vm.togglePreviewSide() }
                        }
                    }
                    .overlay(alignment: .bottom) {
                        if vm.previewURL != nil {
                            Text(vm.showingBack ? "Back  •  tap to flip" : "Front  •  tap to flip")
                                .font(.caption2)
                                .foregroundStyle(.white)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(.black.opacity(0.45))
                                .clipShape(Capsule())
                                .padding(.bottom, Layout.previewBadgeBottomPadding)
                        }
                    }
                    .contextMenu {
                        if let url = vm.previewURL {
                            Button {
                                if let image = NSImage(contentsOf: url) {
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.writeObjects([image])
                                }
                            } label: {
                                Label("Copy to Clipboard", systemImage: "doc.on.clipboard")
                            }
                            Button {
                                if let gcURL = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.lemkesoft.graphicconverter12") {
                                    NSWorkspace.shared.open([url], withApplicationAt: gcURL, configuration: NSWorkspace.OpenConfiguration())
                                }
                            } label: {
                                Label("Open in GraphicConverter", systemImage: "photo")
                            }
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: previewHeight)
                    .clipped()

                if showsListingBar {
                    ListingInfoBar(listing: vm.selectedListing)
                }

                actionArea
                    .frame(height: Layout.actionHeight)
            }
        }
    }

    private var actionArea: some View {
        VStack(spacing: 12) {
            Text("Click Identify Card to generate a name using OpenAI. Edit the name if needed, then press Return or click Rename to apply it. Use the TCDB and eBay buttons to search the web using the card's name.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 8) {
                Text("Name")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(width: 44, alignment: .trailing)
                TextField("Select a card to begin", text: $vm.proposedName)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
                    .onSubmit {
                        guard !vm.isBusy else { return }
                        vm.acceptName()
                    }
            }

            CardTraitEditor(
                title: "Traits",
                traits: vm.selectedTraits,
                isDisabled: vm.selectedPair == nil || vm.isBusy,
                onToggle: vm.toggleTrait
            )

            HStack(spacing: 8) {
                Button {
                    vm.startNaming()
                } label: {
                    Label(vm.isBusy ? "Identifying…" : "Identify Card", systemImage: "sparkles")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.indigo)
                .controlSize(.large)
                .disabled(vm.selectedPair == nil || vm.isBusy)

                Button {
                    vm.acceptName()
                } label: {
                    Label("Rename", systemImage: "checkmark")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                .controlSize(.large)
                .disabled(vm.selectedPair == nil || vm.proposedName.trimmingCharacters(in: .whitespaces).isEmpty || vm.isBusy)
            }

            HStack(spacing: 8) {
                Button {
                    vm.openTCDB()
                } label: {
                    Label("TCDB", systemImage: "magnifyingglass")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(vm.proposedName.isEmpty || vm.isBusy)

                Button {
                    vm.openEbay()
                } label: {
                    Label("eBay", systemImage: "tag")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(vm.proposedName.isEmpty || vm.isBusy)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    Text(vm.logText.isEmpty ? "Ready." : vm.logText)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(vm.logText.isEmpty ? .tertiary : .secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .id("log")
                }
                .frame(height: 64)
                .padding(8)
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.primary.opacity(0.08)))
                .onChange(of: vm.logText) {
                    proxy.scrollTo("log", anchor: .bottom)
                }
            }
        }
        .padding(16)
        .background(Color(nsColor: .controlBackgroundColor))
        .overlay(alignment: .top) { Divider() }
    }
}

// MARK: - eBay Titles Sidebar

struct EbayTitlesSidebar: View {
    @Bindable var vm: EbayTitlesViewModel
    @State private var showDeleteConfirmation = false
    @State private var pendingDeletePair: CardPair?

    private var hasMultipleSelectedCards: Bool {
        vm.selectedIDs.count > 1
    }

    /// Right-clicking a row that's part of the current selection acts on the
    /// whole selection; right-clicking outside it acts on just that row.
    private func contextPairs(for pair: CardPair) -> [CardPair] {
        hasMultipleSelectedCards && vm.selectedIDs.contains(pair.id) ? vm.selectedPairs : [pair]
    }

    private func copyCardNames(_ pairs: [CardPair]) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(
            pairs.map(\.displayName).joined(separator: "\n"),
            forType: .string
        )
    }

    @ViewBuilder
    private func listedMenuItem(for pair: CardPair) -> some View {
        let targets = contextPairs(for: pair)
        let isListed = vm.listing(for: pair).listed
        Button {
            vm.toggleListed(targets)
        } label: {
            Label(
                listedMenuTitle(isListed: isListed, count: targets.count),
                systemImage: isListed ? "xmark.seal" : "checkmark.seal"
            )
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            sourceHeader
            List(selection: $vm.selectedIDs) {
                if vm.parentDirectory != nil || !vm.childDirectories.isEmpty {
                    Section {
                        if vm.parentDirectory != nil {
                            Button {
                                vm.navigateToParentDirectory()
                            } label: {
                                directoryRowLabel(name: "..", icon: "arrowshape.turn.up.left")
                            }
                            .buttonStyle(.plain)
                        }

                        ForEach(vm.childDirectories, id: \.path) { directory in
                            Button {
                                vm.navigateToDirectory(directory)
                            } label: {
                                directoryRowLabel(name: directory.lastPathComponent, icon: "folder")
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }

                Section {
                    ForEach(vm.visiblePairs) { pair in
                        let isListed = vm.listedPairs.contains(pair.baseName)
                        Label {
                            HStack(spacing: 4) {
                                Text(pair.displayName)
                                    .font(.system(size: 12))
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                    .strikethrough(isListed)
                                if isListed {
                                    Image(systemName: "checkmark.seal.fill")
                                        .font(.system(size: 9))
                                        .foregroundStyle(.green)
                                }
                            }
                        } icon: {
                            Image(systemName: "photo.on.rectangle")
                                .foregroundStyle(.secondary)
                        }
                        .tag(pair.id)
                        .help(listingRowTooltip(vm.listing(for: pair)) ?? pair.displayName)
                        .contextMenu {
                            listedMenuItem(for: pair)

                            Divider()

                            Button {
                                vm.movePairsToSales(contextPairs(for: pair))
                            } label: {
                                Label("Move to Sales", systemImage: "cart")
                            }
                            .disabled(vm.isBusy)

                            Button {
                                vm.movePairsToCollection(contextPairs(for: pair))
                            } label: {
                                Label("Move to Collection", systemImage: "archivebox")
                            }
                            .disabled(vm.isBusy)

                            Divider()

                            Button {
                                if let gcURL = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.lemkesoft.graphicconverter12") {
                                    NSWorkspace.shared.open([pair.front], withApplicationAt: gcURL, configuration: NSWorkspace.OpenConfiguration())
                                }
                            } label: {
                                Label("Open in GraphicConverter", systemImage: "photo")
                            }
                            .disabled(hasMultipleSelectedCards)

                            Button {
                                if let image = NSImage(contentsOf: pair.front) {
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.writeObjects([image])
                                }
                            } label: {
                                Label("Copy to Clipboard", systemImage: "doc.on.clipboard")
                            }
                            .disabled(hasMultipleSelectedCards)

                            Button {
                                copyCardNames(contextPairs(for: pair))
                            } label: {
                                Label("Copy Card Names", systemImage: "list.clipboard")
                            }

                            Button {
                                guard let title = vm.listing(for: pair).title else { return }
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(title, forType: .string)
                            } label: {
                                Label("Copy Title", systemImage: "doc.on.doc")
                            }
                            .disabled(vm.listing(for: pair).title == nil)

                            Divider()

                            Button(role: .destructive) {
                                pendingDeletePair = pair
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            .confirmationDialog(
                pendingDeletePair.map { pair in
                    contextPairs(for: pair).count > 1
                        ? "Delete \(contextPairs(for: pair).count) cards?"
                        : "Delete \(pair.displayName)?"
                } ?? "",
                isPresented: Binding(get: { pendingDeletePair != nil }, set: { if !$0 { pendingDeletePair = nil } }),
                titleVisibility: .visible
            ) {
                if let pair = pendingDeletePair {
                    Button("Move to Trash", role: .destructive) {
                        vm.deleteCards(contextPairs(for: pair))
                        pendingDeletePair = nil
                    }
                }
                Button("Cancel", role: .cancel) { pendingDeletePair = nil }
            } message: {
                Text("Both the front and back image files will be moved to the Trash.")
            }

            Divider()
            HStack(spacing: 8) {
                Button("All")  { vm.selectAll() }
                Button("None") { vm.selectNone() }
                Spacer()
                Text("\(vm.selectedIDs.count) selected")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 4)

            Button(role: .destructive) {
                showDeleteConfirmation = true
            } label: {
                Label("Delete Selected", systemImage: "trash")
                    .frame(maxWidth: .infinity)
            }
            .disabled(vm.selectedIDs.isEmpty)
            .padding(.horizontal, 12)
            .padding(.bottom, 8)
            .confirmationDialog(
                "Move \(vm.selectedIDs.count) \(vm.selectedIDs.count == 1 ? "pair" : "pairs") to the Trash?",
                isPresented: $showDeleteConfirmation,
                titleVisibility: .visible
            ) {
                Button("Move to Trash", role: .destructive) {
                    vm.deleteSelectedPairs()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Both front and back files for each selected pair will be moved to the Trash.")
            }
        }
        .navigationSplitViewColumnWidth(min: 240, ideal: 300, max: 420)
    }

    private func directoryRowLabel(name: String, icon: String) -> some View {
        Label {
            Text(name)
                .font(.system(size: 12))
                .lineLimit(1)
                .truncationMode(.middle)
        } icon: {
            Image(systemName: icon)
                .foregroundStyle(.blue)
        }
    }

    private var sourceHeader: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Source")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(URL(fileURLWithPath: vm.directoryPath).lastPathComponent)
                .font(.headline)
                .lineLimit(1)
                .truncationMode(.middle)
            Text(vm.directoryPath)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .truncationMode(.middle)
            // No player/year/set filters here: eBay Titles runs on raw scans,
            // and those filters parse a naming convention this input lacks.
            SidebarSearchField(text: $vm.filterText, placeholder: "Search filenames")
                .padding(.top, 8)
            HStack(spacing: 8) {
                Picker("Sort By", selection: $vm.sortField) {
                    ForEach(CardPairSortField.allCases) { field in
                        Text(field.rawValue).tag(field)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)

                Picker("Sort Order", selection: $vm.sortOrder) {
                    ForEach(CardPairSortOrder.allCases) { order in
                        Text(order.rawValue).tag(order)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
            }
            .controlSize(.small)
            .padding(.top, 8)
            HStack(spacing: 8) {
                SidebarListedToggle(isOn: $vm.hideListed, hiddenCount: vm.hiddenListedCount)
                Spacer(minLength: 0)
                Text("\(vm.visiblePairs.count)/\(vm.pairs.count)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            .padding(.top, 6)
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
        .padding(.bottom, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor))
        .overlay(alignment: .bottom) { Divider() }
    }
}

// MARK: - eBay Titles Detail

struct EbayTitlesDetail: View {
    @Bindable var vm: EbayTitlesViewModel

    var body: some View {
        VStack(spacing: 0) {
            Color(nsColor: .windowBackgroundColor)
                .overlay {
                    // Matches Card Namer: a multi-selection previews as a grid,
                    // a single card as the full-size front/back view.
                    if vm.selectedIDs.count > 1 {
                        MultiSelectionThumbnailGrid(pairs: vm.selectedPairs)
                    } else {
                        CardPreviewView(imageURL: vm.previewURL, reloadID: vm.previewRevision)
                            .onTapGesture { vm.togglePreviewSide() }
                    }
                }
                .overlay(alignment: .bottom) {
                    if vm.previewURL != nil {
                        Text(vm.showingBack ? "Back  •  tap to flip" : "Front  •  tap to flip")
                            .font(.caption2)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(.black.opacity(0.45))
                            .clipShape(Capsule())
                            .padding(.bottom, 10)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipped()

            if vm.selectedPair != nil {
                ListingInfoBar(
                    listing: vm.selectedListing,
                    isEditable: true,
                    onCommitTitle: vm.updateSelectedTitle
                )
                // Rebuild per card so an untouched draft can't follow the
                // selection onto the next card.
                .id(vm.selectedPairID)
            }


            actionArea
        }
    }

    /// Singular for one card, since that path updates the title in place rather
    /// than opening the results window.
    private var generateButtonTitle: String {
        if vm.isBusy { return "Generating…" }
        return vm.selectedIDs.count == 1 ? "Generate Title" : "Generate Titles"
    }

    private var actionArea: some View {
        VStack(spacing: 12) {
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Category").font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $vm.category) {
                        ForEach(EbayCategory.allCases) { cat in
                            Text(cat.rawValue).tag(cat)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 160)
                }

                if vm.category.showsOverrides {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Set Override").font(.caption).foregroundStyle(.secondary)
                        TextField("e.g. 2024-25 Panini Select", text: $vm.setOverride)
                            .textFieldStyle(.roundedBorder)
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Variety Override").font(.caption).foregroundStyle(.secondary)
                        TextField("e.g. Silver Prizm", text: $vm.varietyOverride)
                            .textFieldStyle(.roundedBorder)
                    }
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Supplemental Rules").font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $vm.supplementalRules)
                    .font(.system(size: 12))
                    .scrollContentBackground(.hidden)
                    .padding(5)
                    .frame(minHeight: 58, maxHeight: 90)
                    .background(Color(nsColor: .textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.primary.opacity(0.15)))
                Text("Saved automatically and added to every generated-title prompt.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            CardTraitEditor(
                title: "Traits",
                traits: vm.selectedTraits,
                isDisabled: vm.selectedPair == nil || vm.isBusy,
                onToggle: vm.toggleTrait
            )

            if vm.isBusy {
                ProgressView(value: vm.progress).tint(.orange)
            }

            HStack(spacing: 8) {
                Button {
                    vm.generateTitles()
                } label: {
                    Label(generateButtonTitle, systemImage: "sparkles")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .controlSize(.large)
                .disabled(vm.selectedIDs.isEmpty || vm.isBusy)

                if vm.hasSavedTitles {
                    Button("Display Titles") {
                        vm.displaySavedTitles()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .disabled(vm.isBusy)
                }
            }

            ScrollViewReader { proxy in
                ScrollView {
                    Text(vm.logText.isEmpty ? "Ready." : vm.logText)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(vm.logText.isEmpty ? .tertiary : .secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .id("log")
                }
                .frame(height: 52)
                .padding(8)
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.primary.opacity(0.08)))
                .onChange(of: vm.logText) { proxy.scrollTo("log", anchor: .bottom) }
            }
        }
        .padding(16)
        .background(Color(nsColor: .controlBackgroundColor))
        .overlay(alignment: .top) { Divider() }
    }
}

struct EbayTitlesResultsWindow: View {
    @Bindable var vm: EbayTitlesViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Generated Titles")
                        .font(.title3.bold())
                    Text("\(vm.results.count) result(s)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Descr to Clip") {
                    EbayListingDescription.copyToPasteboard()
                }
                .help("Copy the formatted eBay description")

                Button("Date to Clip") {
                    let formatter = DateFormatter()
                    formatter.dateFormat = "yyyyMMdd"
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(formatter.string(from: Date()), forType: .string)
                }
                .help("Copy today's date as a YYYYMMDD SKU")

                Button("Copy All") {
                    let text = vm.results.map(\.title).joined(separator: "\n")
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                }
                .disabled(vm.results.isEmpty)
            }

            if vm.results.isEmpty {
                ContentUnavailableView(
                    "No Results Yet",
                    systemImage: "text.page",
                    description: Text("Run Generate Titles to open results here.")
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(spacing: 8) {
                        ForEach($vm.results) { $result in
                            EbayTitleResultRow(result: $result) {
                                vm.saveEditedTitles()
                            }
                        }
                    }
                }
            }
        }
        .padding(16)
    }
}

private struct EbayTitleResultRow: View {
    @Binding var result: EbayTitleResult
    let save: () -> Void

    @FocusState private var titleIsFocused: Bool
    @State private var lastSavedTitle: String

    init(result: Binding<EbayTitleResult>, save: @escaping () -> Void) {
        _result = result
        self.save = save
        _lastSavedTitle = State(initialValue: result.wrappedValue.title)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(result.frontName)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                TextField("Title", text: $result.title)
                    .font(.system(size: 13))
                    .textFieldStyle(.plain)
                    .focused($titleIsFocused)
                    .onSubmit {
                        commitTitleChange()
                    }
                    .onChange(of: titleIsFocused) { _, isFocused in
                        if !isFocused {
                            commitTitleChange()
                        }
                    }
            }
            Spacer()
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(result.title, forType: .string)
            } label: {
                Image(systemName: "doc.on.doc")
            }
            .buttonStyle(.borderless)
            .help("Copy title")

            Button {
                if let url = CardNameBuilder.ebayURL(fromBaseName: result.title) {
                    NSWorkspace.shared.open(url)
                }
            } label: {
                Image(systemName: "magnifyingglass")
            }
            .buttonStyle(.borderless)
            .help("Search eBay for this title")
        }
        .padding(10)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func commitTitleChange() {
        guard result.title != lastSavedTitle else { return }
        save()
        lastSavedTitle = result.title
    }
}

private enum EbayListingDescription {
    static let html = """
    <div style="font-family: Arial, Helvetica, sans-serif; font-size: 11pt; color: #000000; line-height: 1.4;">
      <p><strong>SHIPPING</strong>: I use a PWE (Plain White Envelope) to send 1 to 8 cards or philatelic items valued under $20. I typically use postage stamps and a letter-tracking service called LetterTrackPro, which provides a tracking number after the item is posted with the U.S. Postal Service. For items thicker than a standard card or weighing more than 3 ounces, I ship via USPS Ground Advantage for $5.</p>
      <ul>
        <li>$1 for the first item; 35 cents each additional item up to $5.</li>
        <li>If you are charged more than $5, <u>I will refund the overpayment</u>.</li>
      </ul>
      <p><strong>INTERNATIONAL</strong>: I offer very reasonable international shipping rates. The low price is possible because I send cards as letters through the postal service without tracking, in plain white envelopes.</p>
      <p><strong>SAVE $$$</strong>: I combine shipping whenever possible. Check out my other lots.</p>
      <p><strong>QUESTIONS</strong>: Welcomed!</p>
    </div>
    """

    static let plainText = """
    SHIPPING: I use a PWE (Plain White Envelope) to send 1 to 8 cards or philatelic items valued under $20. I typically use postage stamps and a letter-tracking service called LetterTrackPro, which provides a tracking number after the item is posted with the U.S. Postal Service. For items thicker than a standard card or weighing more than 3 ounces, I ship via USPS Ground Advantage for $5.

    • $1 for the first item; 35 cents each additional item up to $5.
    • If you are charged more than $5, I will refund the overpayment.

    INTERNATIONAL: I offer very reasonable international shipping rates. The low price is possible because I send cards as letters through the postal service without tracking, in plain white envelopes.

    SAVE $$$: I combine shipping whenever possible. Check out my other lots.

    QUESTIONS: Welcomed!
    """

    static func copyToPasteboard() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.declareTypes([.html, .string], owner: nil)
        pasteboard.setString(html, forType: .html)
        pasteboard.setString(plainText, forType: .string)
    }
}

// MARK: - Card preview

/// Shared by `CardPreviewView` and `MultiSelectionThumbnailGrid` so a card's image
/// is only ever decoded once regardless of which view requests it.
@MainActor
fileprivate enum CardImageCache {
    static let shared = NSCache<NSURL, NSImage>()
}

struct CardPreviewView: View {
    let imageURL: URL?
    let reloadID: Int
    @State private var loadedURL: URL?
    @State private var loadedImage: NSImage?

    private static var imageCache: NSCache<NSURL, NSImage> { CardImageCache.shared }

    static func invalidateCache(for imageURL: URL) {
        imageCache.removeObject(forKey: imageURL as NSURL)
    }

    var body: some View {
        GeometryReader { proxy in
            Group {
                if let loadedImage, loadedURL == imageURL {
                    Image(nsImage: loadedImage)
                        .resizable()
                        .scaledToFit()
                        .frame(
                            width: max(proxy.size.width - 24, 0),
                            height: max(proxy.size.height - 24, 0)
                        )
                        .position(x: proxy.size.width / 2, y: proxy.size.height / 2)
                } else if imageURL != nil {
                    ProgressView()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 40))
                            .foregroundStyle(.tertiary)
                        Text("Select a card pair")
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .task(id: "\(imageURL?.path ?? ""):\(reloadID)") {
            await loadImage()
        }
    }

    private func loadImage() async {
        guard let imageURL else {
            loadedURL = nil
            loadedImage = nil
            return
        }

        let cacheKey = imageURL as NSURL
        if let cached = Self.imageCache.object(forKey: cacheKey) {
            loadedURL = imageURL
            loadedImage = cached
            return
        }

        loadedURL = imageURL
        loadedImage = nil

        let image = await Task.detached(priority: .userInitiated) {
            NSImage(contentsOf: imageURL)
        }.value

        guard !Task.isCancelled, loadedURL == imageURL else { return }
        if let image {
            Self.imageCache.setObject(image, forKey: cacheKey)
        }
        loadedImage = image
    }
}

// MARK: - Multi-selection thumbnail grid

/// Snapshot view shown in place of the single-card preview when more than one
/// card is selected: a grid of front-only thumbnails. The column count and
/// thumbnail size are computed from the pane's actual size so that, whenever
/// possible, every selected card fits on screen at once (shrinking thumbnails
/// as the selection grows) instead of always requiring a scroll.
struct MultiSelectionThumbnailGrid: View {
    let pairs: [CardPair]

    @State private var expandedIndex: Int?

    private let spacing: CGFloat = 10
    private let outerPadding: CGFloat = 14
    private let labelHeight: CGFloat = 18
    private let aspectRatio: CGFloat = 2.5 / 3.5 // width / height
    private let minItemWidth: CGFloat = 90
    private let maxItemWidth: CGFloat = 260

    var body: some View {
        GeometryReader { proxy in
            let available = CGSize(
                width: max(proxy.size.width - outerPadding * 2, 0),
                height: max(proxy.size.height - outerPadding * 2, 0)
            )
            let layout = computeLayout(count: pairs.count, in: available)

            ScrollView {
                LazyVGrid(
                    columns: Array(repeating: GridItem(.fixed(layout.itemWidth), spacing: spacing), count: layout.columns),
                    spacing: spacing
                ) {
                    ForEach(Array(pairs.enumerated()), id: \.element.id) { index, pair in
                        VStack(spacing: 4) {
                            CardThumbnailView(imageURL: pair.front)
                                .aspectRatio(aspectRatio, contentMode: .fit)
                                .frame(width: layout.itemWidth)
                                .background(Color(nsColor: .textBackgroundColor))
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                                .overlay {
                                    RoundedRectangle(cornerRadius: 6)
                                        .strokeBorder(Color.secondary.opacity(0.25))
                                }
                                .contentShape(Rectangle())
                                .onTapGesture {
                                    withAnimation(.easeOut(duration: 0.16)) {
                                        expandedIndex = index
                                    }
                                }
                            Text(pair.displayName)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .frame(width: layout.itemWidth)
                        }
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(outerPadding)
            }
        }
        .overlay(alignment: .topTrailing) {
            Text("\(pairs.count) selected")
                .font(.caption2)
                .foregroundStyle(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.black.opacity(0.45))
                .clipShape(Capsule())
                .padding(8)
        }
        .overlay {
            if let expandedIndex, pairs.indices.contains(expandedIndex) {
                CardLightboxView(
                    pairs: pairs,
                    index: expandedIndex,
                    onNavigate: { self.expandedIndex = $0 },
                    onClose: {
                        withAnimation(.easeOut(duration: 0.16)) {
                            self.expandedIndex = nil
                        }
                    }
                )
                .transition(.opacity)
            }
        }
    }

    /// Finds the column count that yields the largest thumbnail (capped at
    /// `maxItemWidth`) whose full grid still fits in `available` without
    /// scrolling; ties favor more columns (a wider, flatter layout). Falls
    /// back to a squarish grid — with scrolling — only when nothing fits.
    private func computeLayout(count: Int, in available: CGSize) -> (columns: Int, itemWidth: CGFloat) {
        guard count > 0, available.width > 0 else { return (1, minItemWidth) }

        var best: (columns: Int, itemWidth: CGFloat)?
        for columns in 1...count {
            let rawItemWidth = (available.width - spacing * CGFloat(columns - 1)) / CGFloat(columns)
            guard rawItemWidth >= minItemWidth else { continue }
            let itemWidth = min(maxItemWidth, rawItemWidth)

            let rows = Int(ceil(Double(count) / Double(columns)))
            let itemHeight = itemWidth / aspectRatio + labelHeight
            let totalHeight = CGFloat(rows) * itemHeight + spacing * CGFloat(max(0, rows - 1))

            if totalHeight <= available.height, best == nil || itemWidth >= best!.itemWidth {
                best = (columns, itemWidth)
            }
        }
        if let best { return best }

        let fallbackColumns = max(1, Int(ceil(sqrt(Double(count)))))
        let itemWidth = max(minItemWidth, min(maxItemWidth, (available.width - spacing * CGFloat(fallbackColumns - 1)) / CGFloat(fallbackColumns)))
        return (fallbackColumns, itemWidth)
    }
}

/// Full-size lightbox overlay opened by tapping a thumbnail in
/// `MultiSelectionThumbnailGrid`. Tapping the dimmed background (or the close
/// button, or Escape) dismisses it; the chevrons and left/right arrow keys
/// step through the rest of the current selection without closing.
private struct CardLightboxView: View {
    let pairs: [CardPair]
    let index: Int
    let onNavigate: (Int) -> Void
    let onClose: () -> Void

    @FocusState private var isFocused: Bool
    @State private var showingBack = false

    private var pair: CardPair? {
        pairs.indices.contains(index) ? pairs[index] : nil
    }

    private var imageURL: URL? {
        guard let pair else { return nil }
        return showingBack ? pair.back : pair.front
    }

    var body: some View {
        ZStack {
            Color.black.opacity(0.72)
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture { onClose() }

            if let pair {
                VStack(spacing: 8) {
                    CardPreviewView(imageURL: imageURL, reloadID: 0)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            withAnimation(.easeOut(duration: 0.12)) {
                                showingBack.toggle()
                            }
                        }
                    Text(showingBack ? "Back  •  tap to flip" : "Front  •  tap to flip")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.8))
                    Text(pair.displayName)
                        .font(.callout)
                        .foregroundStyle(.white)
                        .padding(.bottom, 16)
                }
                .padding(48)
            }

            HStack {
                navButton(systemImage: "chevron.left") { step(by: -1) }
                    .disabled(pairs.count <= 1)
                Spacer()
                navButton(systemImage: "chevron.right") { step(by: 1) }
                    .disabled(pairs.count <= 1)
            }
            .padding(.horizontal, 24)

            VStack {
                HStack {
                    Spacer()
                    Button(action: onClose) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 26))
                            .foregroundStyle(.white, .black.opacity(0.4))
                    }
                    .buttonStyle(.plain)
                    .padding(16)
                }
                Spacer()
            }
        }
        .focusable()
        .focused($isFocused)
        .onAppear { isFocused = true }
        .onChange(of: index) { showingBack = false }
        .onKeyPress(.escape) { onClose(); return .handled }
        .onKeyPress(.leftArrow) { step(by: -1); return .handled }
        .onKeyPress(.rightArrow) { step(by: 1); return .handled }
    }

    private func step(by delta: Int) {
        guard !pairs.isEmpty else { return }
        let next = (index + delta + pairs.count) % pairs.count
        onNavigate(next)
    }

    private func navButton(systemImage: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(.white)
                .padding(14)
                .background(.black.opacity(0.35), in: Circle())
        }
        .buttonStyle(.plain)
    }
}

/// A single lightweight thumbnail (front image only) for `MultiSelectionThumbnailGrid`.
private struct CardThumbnailView: View {
    let imageURL: URL
    @State private var loadedURL: URL?
    @State private var image: NSImage?

    var body: some View {
        Group {
            if let image, loadedURL == imageURL {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
            } else {
                ProgressView()
                    .controlSize(.small)
            }
        }
        .task(id: imageURL) {
            await load()
        }
    }

    private func load() async {
        let cacheKey = imageURL as NSURL
        if let cached = CardImageCache.shared.object(forKey: cacheKey) {
            loadedURL = imageURL
            image = cached
            return
        }

        loadedURL = imageURL
        image = nil

        let loaded = await Task.detached(priority: .utility) {
            NSImage(contentsOf: imageURL)
        }.value

        guard !Task.isCancelled, loadedURL == imageURL else { return }
        if let loaded {
            CardImageCache.shared.setObject(loaded, forKey: cacheKey)
        }
        image = loaded
    }
}

struct PSACertPromptSheet: View {
    @Binding var certNumber: String
    let onCancel: () -> Void
    let onDownload: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Download from PSA")
                .font(.title3.bold())

            VStack(alignment: .leading, spacing: 6) {
                Text("Certification Number")
                    .font(.headline)
                TextField("Enter PSA cert number", text: $certNumber)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
            }

            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                    .keyboardShortcut(.escape)
                Button("Download", action: onDownload)
                    .keyboardShortcut(.return)
                    .buttonStyle(.borderedProminent)
                    .disabled(certNumber.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(24)
        .frame(width: 360)
    }
}

// MARK: - Settings sheet

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var apiKey: String = SettingsStore.shared.openAIKey
    @State private var psaToken: String = SettingsStore.shared.psaToken
    @State private var selectedModel: String = SettingsStore.shared.selectedModel
    @State private var availableModels: [String] = []
    @State private var loadingModels = false

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {

            Text("Settings")
                .font(.title2.bold())

            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Label("OpenAI API Key", systemImage: "key.horizontal")
                        .font(.headline)
                    Text("Get your key at platform.openai.com → API Keys.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextField("sk-…", text: $apiKey)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.body, design: .monospaced))
                }
                .padding(4)
            }

            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Label("PSA API Token", systemImage: "key.horizontal.fill")
                        .font(.headline)
                    Text("Used by the PSA download button to fetch front and back images by cert number.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextField("Paste PSA bearer token", text: $psaToken)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.body, design: .monospaced))
                }
                .padding(4)
            }

            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label("Model", systemImage: "cpu")
                            .font(.headline)
                        Spacer()
                        if loadingModels {
                            ProgressView().controlSize(.small)
                        } else {
                            Button("Refresh List") { fetchModels() }
                                .controlSize(.small)
                        }
                    }
                    Text("Fetched live from your OpenAI account.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    if availableModels.isEmpty {
                        Text(loadingModels ? "Loading models…" : "Save a valid API key, then tap Refresh List.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding(.vertical, 6)
                    } else {
                        Picker("Model", selection: $selectedModel) {
                            ForEach(availableModels, id: \.self) { model in
                                Text(model).tag(model)
                            }
                        }
                        .labelsHidden()
                    }

                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .font(.caption)
                        Text(selectedModel)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(4)
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.escape)
                Button("Save") {
                    SettingsStore.shared.openAIKey = apiKey
                    SettingsStore.shared.psaToken = psaToken
                    SettingsStore.shared.selectedModel = selectedModel
                    dismiss()
                }
                .keyboardShortcut(.return)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(24)
        .frame(width: 420)
        .task { fetchModels() }
    }

    private func fetchModels() {
        guard !apiKey.isEmpty else { return }
        loadingModels = true
        Task {
            let orig = SettingsStore.shared.openAIKey
            SettingsStore.shared.openAIKey = apiKey
            let models = await OpenAIService.fetchChatModels()
            SettingsStore.shared.openAIKey = orig
            await MainActor.run {
                availableModels = models
                if !models.contains(selectedModel), let first = models.first {
                    selectedModel = first
                }
                loadingModels = false
            }
        }
    }
}
