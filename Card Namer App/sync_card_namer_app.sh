#!/bin/zsh
set -euo pipefail

APP_DIR="/Users/geoff/code/card_stuff/Card Namer App/Card Namer.app"
WRAPPER="/Users/geoff/code/card_stuff/Card Namer App/card_namer_wrapper.sh"
APP_SCRIPT="$APP_DIR/Contents/Resources/script"
INFO_PLIST="$APP_DIR/Contents/Info.plist"
USAGE_TEXT="Card Namer needs access to removable drives to read and organize card images."

if [ ! -d "$APP_DIR" ]; then
  echo "App bundle not found: $APP_DIR" >&2
  exit 1
fi

if [ ! -f "$WRAPPER" ]; then
  echo "Wrapper script not found: $WRAPPER" >&2
  exit 1
fi

cp "$WRAPPER" "$APP_SCRIPT"
chmod +x "$APP_SCRIPT"

/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier org.geoff.CardNamer" "$INFO_PLIST" \
  || /usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string org.geoff.CardNamer" "$INFO_PLIST"

/usr/libexec/PlistBuddy -c "Set :NSRemovableVolumesUsageDescription $USAGE_TEXT" "$INFO_PLIST" \
  || /usr/libexec/PlistBuddy -c "Add :NSRemovableVolumesUsageDescription string $USAGE_TEXT" "$INFO_PLIST"

plutil -p "$INFO_PLIST" | rg 'CFBundleIdentifier|NSRemovableVolumesUsageDescription'
