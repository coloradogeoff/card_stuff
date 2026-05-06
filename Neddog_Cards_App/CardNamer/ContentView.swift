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

    var body: some View {
        NavigationSplitView(columnVisibility: .constant(.all)) {
            sidebar
        } detail: {
            detail
        }
        .navigationSplitViewStyle(.balanced)
        .sheet(isPresented: $showSettings) { SettingsView() }
        .toolbar {
            ToolbarItem(placement: .navigation) { directoryMenu }
            ToolbarItem(placement: .principal) { modePicker }
            ToolbarItem(placement: .primaryAction) {
                Button { showSettings = true } label: { Image(systemName: "gear") }
                    .help("API Key & Model Settings")
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .showEbayResultsWindow)) { _ in
            openWindow(id: SceneID.ebayResults)
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

    private var directoryMenu: some View {
        Menu {
            ForEach(SettingsStore.shared.quickDirectories) { dir in
                Button {
                    switchDirectory(dir)
                } label: {
                    Label {
                        Text(dir.name) +
                        Text(dir.isAvailable ? "" : "  (unavailable)")
                            .foregroundStyle(.secondary)
                    } icon: {
                        Image(systemName: dir.isAvailable ? "folder.fill" : "folder.badge.questionmark")
                    }
                }
                .disabled(!dir.isAvailable)
            }
            Divider()
            Button("Browse…") { chooseDirectory() }
            Divider()
            Button("Refresh") { refreshImages() }
        } label: {
            Label {
                Text(URL(fileURLWithPath: currentDirectoryPath).lastPathComponent)
                    .lineLimit(1)
            } icon: {
                Image(systemName: "folder")
            }
        }
        .menuStyle(.borderlessButton)
        .help(currentDirectoryPath)
    }
}

// MARK: - Card Namer Sidebar

struct CardNamerSidebar: View {
    @Bindable var vm: CardNamerViewModel

    var body: some View {
        VStack(spacing: 0) {
            sourceHeader
            List(vm.pairs, selection: $vm.selectedPairID) { pair in
                Label {
                    Text(pair.displayName)
                        .font(.system(size: 12))
                        .lineLimit(1)
                        .truncationMode(.middle)
                } icon: {
                    Image(systemName: "photo.on.rectangle")
                        .foregroundStyle(.secondary)
                }
                .tag(pair.id)
                .contextMenu {
                    Button(role: .destructive) {
                        vm.deleteCard(pair)
                    } label: {
                        Label("Delete Card", systemImage: "trash")
                    }
                }
            }
            .listStyle(.sidebar)

            if !vm.pairs.isEmpty && !vm.isExistingCardsDirectory {
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
                        Label("Move All", systemImage: "arrow.right.square.fill")
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
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
        .padding(.bottom, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor))
        .overlay(alignment: .bottom) { Divider() }
    }
}

// MARK: - Card Namer Detail

struct CardNamerDetail: View {
    private enum Layout {
        static let actionHeight: CGFloat = 230
        static let minimumPreviewHeight: CGFloat = 240
        static let previewBadgeBottomPadding: CGFloat = 28
    }

    @Bindable var vm: CardNamerViewModel

    var body: some View {
        GeometryReader { proxy in
            let previewHeight = max(
                proxy.size.height - Layout.actionHeight,
                Layout.minimumPreviewHeight
            )

            VStack(spacing: 0) {
                Color(nsColor: .windowBackgroundColor)
                    .overlay {
                        CardPreviewView(imageURL: vm.previewURL)
                            .onTapGesture { vm.togglePreviewSide() }
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
                    .frame(maxWidth: .infinity)
                    .frame(height: previewHeight)
                    .clipped()

                actionArea
                    .frame(height: Layout.actionHeight)
            }
        }
    }

    private var actionArea: some View {
        VStack(spacing: 12) {
            Text("Click Identify Card to generate a name using OpenAI. Edit the name if needed, then click Rename to apply it. Use the TCDB and eBay buttons to search the web using the card's name.")
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
            }

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

    var body: some View {
        VStack(spacing: 0) {
            sourceHeader
            List(vm.pairs, selection: $vm.selectedPairID) { pair in
                HStack(spacing: 6) {
                    Image(systemName: vm.checkedIDs.contains(pair.id) ? "checkmark.square.fill" : "square")
                        .foregroundStyle(vm.checkedIDs.contains(pair.id) ? Color.accentColor : .secondary)
                        .onTapGesture { vm.toggleCheck(pair) }
                    Text(pair.displayName)
                        .font(.system(size: 12))
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                .tag(pair.id)
            }
            .listStyle(.sidebar)

            Divider()
            HStack(spacing: 8) {
                Button("All")  { vm.selectAll() }
                Button("None") { vm.selectNone() }
                Spacer()
                Text("\(vm.checkedIDs.count) selected")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .navigationSplitViewColumnWidth(min: 240, ideal: 300, max: 420)
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
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
        .padding(.bottom, 8)
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
                    CardPreviewView(imageURL: vm.previewURL)
                        .onTapGesture { vm.togglePreviewSide() }
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

            actionArea
        }
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

            if vm.isBusy {
                ProgressView(value: vm.progress).tint(.orange)
            }

            HStack(spacing: 8) {
                Button {
                    vm.generateTitles()
                } label: {
                    Label(vm.isBusy ? "Generating…" : "Generate Titles", systemImage: "sparkles")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .controlSize(.large)
                .disabled(vm.checkedIDs.isEmpty || vm.isBusy)

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
                        ForEach(vm.results) { result in
                            HStack(alignment: .top, spacing: 8) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(result.frontName)
                                        .font(.system(size: 11))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                    Text(result.title)
                                        .font(.system(size: 13))
                                        .textSelection(.enabled)
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
                    }
                }
            }
        }
        .padding(16)
    }
}

// MARK: - Card preview

struct CardPreviewView: View {
    let imageURL: URL?

    var body: some View {
        GeometryReader { proxy in
            Group {
                if let url = imageURL, let nsImage = NSImage(contentsOf: url) {
                    Image(nsImage: nsImage)
                        .resizable()
                        .scaledToFit()
                        .frame(
                            width: max(proxy.size.width - 24, 0),
                            height: max(proxy.size.height - 24, 0)
                        )
                        .position(x: proxy.size.width / 2, y: proxy.size.height / 2)
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
    }
}

// MARK: - Settings sheet

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var apiKey: String = SettingsStore.shared.openAIKey
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
