import AppKit
import SwiftUI

struct PostageView: View {
    @StateObject private var store = PostageStore()
    @State private var selectedRateID: PostageRateID = .firstClassLetter1Ounce
    @State private var toleranceCents = 1
    @State private var requiredDenominations: Set<Int> = []
    @State private var showingManagementSheet = false

    private var selectedRate: PostageRate {
        store.settings.rates.first(where: { $0.id == selectedRateID })
            ?? PostageRate(id: selectedRateID, cents: selectedRateID.defaultCents)
    }

    private var selection: StampSelection? {
        StampCombinationSolver.solve(
            targetCents: selectedRate.cents,
            denominations: store.settings.stampDenominations,
            toleranceCents: toleranceCents,
            requiredDenominations: requiredDenominations
        )
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                rateSection
                toleranceSection
                requiredDenominationSection
                resultSection
                Button {
                    showingManagementSheet = true
                } label: {
                    Label("Manage Rates & Stamps", systemImage: "slider.horizontal.3")
                        .frame(maxWidth: .infinity)
                }
                .controlSize(.large)
            }
            .padding(16)
        }
        .sheet(isPresented: $showingManagementSheet) {
            PostageManagementSheet(store: store)
        }
        .onChange(of: store.settings.stampDenominations) {
            requiredDenominations.formIntersection(store.settings.stampDenominations)
        }
    }

    private var rateSection: some View {
        GroupBox("Postal Rate") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("Rate", selection: $selectedRateID) {
                    ForEach(store.settings.rates) { rate in
                        Text("\(rate.id.label) — \(currency(rate.cents))")
                            .tag(rate.id)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: .infinity)

                HStack {
                    Text(selectedRate.id.label)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(currency(selectedRate.cents))
                        .font(.title2.bold())
                        .monospacedDigit()
                }
            }
            .padding(.vertical, 4)
        }
    }

    private var toleranceSection: some View {
        GroupBox("Allowed Overpayment") {
            Picker("Allowed Overpayment", selection: $toleranceCents) {
                Text("0¢").tag(0)
                Text("1¢").tag(1)
                Text("2¢").tag(2)
                Text("3¢").tag(3)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.vertical, 4)
        }
    }

    private var requiredDenominationSection: some View {
        GroupBox("Required Denomination") {
            VStack(alignment: .leading, spacing: 8) {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 66), spacing: 6)],
                    alignment: .leading,
                    spacing: 6
                ) {
                    ForEach(store.settings.stampDenominations.sorted(), id: \.self) { denomination in
                        Toggle(
                            stampValue(denomination),
                            isOn: requiredBinding(for: denomination)
                        )
                        .toggleStyle(.checkbox)
                        .monospacedDigit()
                    }
                }

                Text("Each checked denomination must appear at least once.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
        }
    }

    private var resultSection: some View {
        GroupBox("Recommended Stamps") {
            VStack(alignment: .leading, spacing: 10) {
                if let selection {
                    ForEach(selection.counts.keys.sorted(by: >), id: \.self) { denomination in
                        HStack {
                            Text("\(selection.counts[denomination, default: 0]) × \(stampValue(denomination))")
                            Spacer()
                            Text(currency(selection.counts[denomination, default: 0] * denomination))
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                    }

                    Divider()

                    resultRow("Stamps", value: "\(selection.stampCount)")
                    resultRow("Postage", value: currency(selection.totalCents))
                    resultRow(
                        "Overpayment",
                        value: selection.overpaymentCents == 0
                            ? "Exact"
                            : "\(selection.overpaymentCents)¢"
                    )

                    if selection.exceedsTolerance {
                        Label(
                            "No combination fits the selected tolerance. Showing the closest available overpayment.",
                            systemImage: "exclamationmark.triangle.fill"
                        )
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                    }
                } else {
                    Text("Add at least one stamp denomination to calculate postage.")
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 4)
        }
    }

    private func resultRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.semibold)
                .monospacedDigit()
        }
    }

    private func requiredBinding(for denomination: Int) -> Binding<Bool> {
        Binding(
            get: { requiredDenominations.contains(denomination) },
            set: { isRequired in
                if isRequired {
                    requiredDenominations.insert(denomination)
                } else {
                    requiredDenominations.remove(denomination)
                }
            }
        )
    }
}

private struct EditableRate: Identifiable {
    let id: PostageRateID
    var dollars: Double
}

private struct PostageManagementSheet: View {
    @ObservedObject var store: PostageStore
    @Environment(\.dismiss) private var dismiss

    @State private var rates: [EditableRate]
    @State private var denominations: [Int]
    @State private var newDenomination: Int?
    @State private var errorMessage: String?

    private let notice123URL = URL(string: "https://pe.usps.com/text/dmm300/notice123.htm")!

    init(store: PostageStore) {
        self.store = store
        _rates = State(initialValue: store.settings.rates.map {
            EditableRate(id: $0.id, dollars: Double($0.cents) / 100)
        })
        _denominations = State(initialValue: store.settings.stampDenominations)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Manage Rates & Stamps")
                    .font(.title2.bold())
                Spacer()
                Button {
                    NSWorkspace.shared.open(notice123URL)
                } label: {
                    Label("Open USPS Notice 123", systemImage: "arrow.up.right.square")
                }
            }

            GroupBox("Postal Rates") {
                VStack(spacing: 7) {
                    ForEach($rates) { $rate in
                        HStack {
                            Text(rate.id.label)
                                .lineLimit(1)
                            Spacer()
                            Text("$")
                                .foregroundStyle(.secondary)
                            TextField(
                                "0.00",
                                value: $rate.dollars,
                                format: .number.precision(.fractionLength(2))
                            )
                            .multilineTextAlignment(.trailing)
                            .monospacedDigit()
                            .frame(width: 70)
                        }
                    }
                }
                .padding(.vertical, 4)
            }

            GroupBox("Available Stamp Denominations") {
                VStack(alignment: .leading, spacing: 8) {
                    ScrollView {
                        LazyVGrid(
                            columns: [GridItem(.adaptive(minimum: 72), spacing: 6)],
                            alignment: .leading,
                            spacing: 6
                        ) {
                            ForEach(denominations.sorted(), id: \.self) { denomination in
                                HStack(spacing: 4) {
                                    Text(stampValue(denomination))
                                        .monospacedDigit()
                                    Spacer(minLength: 2)
                                    Button {
                                        denominations.removeAll { $0 == denomination }
                                    } label: {
                                        Image(systemName: "xmark.circle.fill")
                                            .foregroundStyle(.secondary)
                                    }
                                    .buttonStyle(.plain)
                                    .help("Remove \(stampValue(denomination)) stamp")
                                }
                                .padding(.horizontal, 7)
                                .padding(.vertical, 5)
                                .background(Color.secondary.opacity(0.12))
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                            }
                        }
                    }
                    .frame(minHeight: 80, maxHeight: 130)

                    HStack {
                        TextField("Value in cents", value: $newDenomination, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 130)
                            .onSubmit(addDenomination)
                        Button("Add", action: addDenomination)
                    }
                }
                .padding(.vertical, 4)
            }

            if let errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            HStack {
                Spacer()
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
                Button("Save", action: save)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(18)
        .frame(width: 520)
    }

    private func addDenomination() {
        errorMessage = nil
        guard let value = newDenomination, value > 0 else {
            errorMessage = "Enter a stamp value greater than zero."
            return
        }
        guard !denominations.contains(value) else {
            errorMessage = "\(stampValue(value)) is already in the list."
            return
        }
        denominations.append(value)
        denominations.sort()
        newDenomination = nil
    }

    private func save() {
        errorMessage = nil
        guard rates.allSatisfy({ $0.dollars > 0 && $0.dollars.isFinite }) else {
            errorMessage = "Every postal rate must be greater than zero."
            return
        }
        guard !denominations.isEmpty else {
            errorMessage = "Keep at least one stamp denomination."
            return
        }

        let settings = PostageSettings(
            rates: rates.map {
                PostageRate(id: $0.id, cents: Int(($0.dollars * 100).rounded()))
            },
            stampDenominations: denominations
        )

        do {
            try store.save(settings)
            dismiss()
        } catch {
            errorMessage = "Could not save settings: \(error.localizedDescription)"
        }
    }
}

private func currency(_ cents: Int) -> String {
    (Double(cents) / 100).formatted(.currency(code: "USD"))
}

private func stampValue(_ cents: Int) -> String {
    cents < 100 ? "\(cents)¢" : currency(cents)
}

#Preview {
    PostageView()
        .frame(width: 320, height: 640)
}
