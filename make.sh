#!/usr/bin/env bash
# Build a Release version of one of the Swift apps and copy the .app to ~/Desktop.
# Usage: ./make.sh cards
#        ./make.sh mail

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP="$HOME/Desktop"

usage() {
    echo "Usage: $0 {cards|mail}" >&2
    exit 1
}

build() {
    local target_name="$1"     # human label
    local project_path="$2"    # absolute path to .xcodeproj
    local scheme="$3"          # Xcode scheme name
    local product="$4"         # .app filename
    local build_dir="$5"       # temp DerivedData dir

    echo "→ Building $target_name (Release)…"
    /usr/bin/arch -arm64 xcodebuild \
        -project "$project_path" \
        -scheme "$scheme" \
        -quiet \
        -configuration Release \
        -destination "platform=macOS,arch=arm64" \
        -derivedDataPath "$build_dir" \
        build

    rm -rf "$DESKTOP/$product"
    cp -R "$build_dir/Build/Products/Release/$product" "$DESKTOP/"
    echo "✓ $DESKTOP/$product"
}

case "${1:-}" in
    cards)
        build \
            "Neddog Cards" \
            "$REPO_ROOT/Neddog_Cards_App/Neddog Cards.xcodeproj" \
            "Neddog Cards" \
            "Neddog Cards.app" \
            "/tmp/neddog-build"
        ;;
    mail)
        build \
            "Ned Mail" \
            "$REPO_ROOT/Ned_Mail/NedMail.xcodeproj" \
            "Ned Mail" \
            "Ned Mail.app" \
            "/tmp/nedmail-build"
        ;;
    *)
        usage
        ;;
esac
