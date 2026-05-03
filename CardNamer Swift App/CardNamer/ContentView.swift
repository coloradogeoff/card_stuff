import SwiftUI
import AppKit

struct ContentView: View {
    @State private var vm = CardNamerViewModel()
    @State private var showSettings = false

    var body: some View {
        NavigationSplitView(columnVisibility: .constant(.all)) {
            sidebarPanel
        } detail: {
            detailPanel
        }
        .navigationSplitViewStyle(.balanced)
        .sheet(isPresented: $showSettings) {
            SettingsView()
        }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                directoryMenu
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "gear")
                }
                .help("API Key & Model Settings")
            }
        }
    }

    // MARK: - Sidebar

    private var sidebarPanel: some View {
        VStack(spacing: 0) {
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

    // MARK: - Detail

    @State private var actionAreaHeight: CGFloat = 220

    private var detailPanel: some View {
        GeometryReader { geo in
            VStack(spacing: 0) {
                // Card preview — fills everything above the action area
                ZStack {
                    Color(nsColor: .windowBackgroundColor)
                    CardPreviewView(imageURL: vm.previewURL)
                        .onTapGesture { vm.togglePreviewSide() }

                    if vm.previewURL != nil {
                        VStack {
                            Spacer()
                            Text(vm.showingBack ? "Back  •  tap to flip" : "Front  •  tap to flip")
                                .font(.caption2)
                                .foregroundStyle(.white)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(.black.opacity(0.45))
                                .clipShape(Capsule())
                                .padding(.bottom, 10)
                        }
                    }
                }
                .frame(width: geo.size.width, height: max(100, geo.size.height - actionAreaHeight))
                .clipped()

                Divider()

            // Action area
            VStack(spacing: 12) {
                // Instructions
                Text("Click Identify Card to generate a name using OpenAI. Edit the name if needed, then click Rename to apply it. Use the TCDB and eBay buttons to search the web using the card's name.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                // Proposed name
                HStack(spacing: 8) {
                    Text("Name")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .frame(width: 44, alignment: .trailing)
                    TextField("Select a card to begin", text: $vm.proposedName)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.body, design: .monospaced))
                }

                // Primary actions
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

                // Secondary actions
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

                // Log
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
            .overlay(GeometryReader { ap in
                Color.clear.preference(key: ActionAreaHeightKey.self, value: ap.size.height)
            })
        }
        .onPreferenceChange(ActionAreaHeightKey.self) { actionAreaHeight = $0 }
        }
    }

    // MARK: - Directory menu

    private var directoryMenu: some View {
        Menu {
            let dirs = SettingsStore.shared.quickDirectories
            ForEach(dirs) { dir in
                Button {
                    vm.switchTo(dir)
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
            Button("Browse…") { vm.chooseDirectory() }
            Divider()
            Button("Refresh") { vm.refreshImages() }
        } label: {
            Label {
                Text(URL(fileURLWithPath: vm.directoryPath).lastPathComponent)
                    .lineLimit(1)
            } icon: {
                Image(systemName: "folder")
            }
        }
        .menuStyle(.borderlessButton)
        .help(vm.directoryPath)
    }
}

// MARK: - Card preview

struct CardPreviewView: View {
    let imageURL: URL?

    var body: some View {
        Group {
            if let url = imageURL, let nsImage = NSImage(contentsOf: url) {
                Image(nsImage: nsImage)
                    .resizable()
                    .scaledToFit()
                    .padding(12)
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

            // API Key
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

            // Model picker
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

private struct ActionAreaHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 220
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}
