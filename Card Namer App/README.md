# Card Namer

Developer notes for the `Card Namer` macOS app and its Python runtime.

## Overview

`Card Namer` is a Platypus-wrapped macOS app that launches a Python script for both GUI and CLI workflows.

Main responsibilities:

- load front/back sports card image pairs from a directory
- OCR card images and send card-identification prompts to OpenAI
- propose normalized filenames
- rename images in place
- move renamed images into a saved library directory
- open TCDB and eBay searches for the current card

## Source Layout

- [Card Namer.app](/Users/geoff/code/card_stuff/Card%20Namer%20App/Card%20Namer.app): generated Platypus app bundle
- [card_namer.py](/Users/geoff/code/card_stuff/Card%20Namer%20App/card_namer.py): main application code
- [card_namer_wrapper.sh](/Users/geoff/code/card_stuff/Card%20Namer%20App/card_namer_wrapper.sh): wrapper script used by the Platypus bundle
- [sync_card_namer_app.sh](/Users/geoff/code/card_stuff/Card%20Namer%20App/sync_card_namer_app.sh): post-rebuild sync step for bundle metadata and launcher script

## Entrypoints

### App Bundle

The Platypus bundle runs:

- `Card Namer.app/Contents/Resources/script`

That script is expected to stay in sync with [card_namer_wrapper.sh](/Users/geoff/code/card_stuff/Card%20Namer%20App/card_namer_wrapper.sh).

The wrapper currently does two things:

- loads `OPENAI_API_KEY` from `/Users/geoff/code/card_stuff/.openai-api-key.txt` if it is not already set
- launches `/Users/geoff/opt/anaconda3/bin/python` against `card_namer.py`

### Python Script

[card_namer.py](/Users/geoff/code/card_stuff/Card%20Namer%20App/card_namer.py) has two entry modes:

- default: launches the PyQt GUI
- `--cli`: uses the Typer CLI

Current dispatch logic:

```bash
/Users/geoff/opt/anaconda3/bin/python "/Users/geoff/code/card_stuff/Card Namer App/card_namer.py"
/Users/geoff/opt/anaconda3/bin/python "/Users/geoff/code/card_stuff/Card Namer App/card_namer.py" --cli --help
```

## Runtime Dependencies

This project is currently coupled to the local machine.

Assumed runtime pieces:

- macOS
- `/Users/geoff/opt/anaconda3/bin/python`
- PyQt5
- Pillow
- pytesseract
- typer
- openai
- Tesseract installed and discoverable by `pytesseract`

The app is not currently packaged as a self-contained Python runtime. The Platypus app delegates to an external Anaconda interpreter.

## GUI Behavior

The GUI is implemented in `CardNamerGui`.

Key behaviors:

- directory selection via `Incoming Cards`, `Existing Card Images`, and `Browse`
- image pairing based on front/back filenames
- preview toggle between front and back image
- background naming work via `QThread`
- rename in place
- move selected or all pairs to the saved existing-cards directory
- open TCDB or eBay searches for the proposed name

By default, launching the script without arguments starts the GUI.

## CLI Behavior

The CLI supports:

- dry-run naming
- in-place rename
- CSV-driven rename via `--usecsv`
- single-card processing via `--images`
- TCDB URL generation via `--tcdb`

The CLI still expects a filesystem directory and paired image structure consistent with the GUI workflow.

## Persistence

### App Support Directory

The app writes its local settings under:

```bash
~/Library/Application Support/Card Namer/
```

Current persisted file:

- `settings.json`

### Existing Cards Directory

`existing_cards_directory` is stored in `settings.json`.

This replaced the previous hardcoded default of `/Volumes/Dutton 2TB/Cards/Mix`.

Current behavior:

- if a saved `existing_cards_directory` exists and is valid, use it
- otherwise fall back to `~/Library/Application Support/Card Namer/Cards`
- when the user picks `Existing Card Images`, save that selection for future runs

## Removable Drive Access

The app had been prompting repeatedly for removable-drive access because it was directly targeting `/Volumes/Dutton 2TB/Cards/Mix`.

The current fix is intentionally simple:

- stop hardcoding the removable drive as the default existing-cards directory
- require the user to select the existing-cards folder through the system folder picker
- persist the chosen path in app settings
- include a removable-volumes usage string in the app bundle

Bundle values that matter here:

- `CFBundleIdentifier = org.geoff.CardNamer`
- `NSRemovableVolumesUsageDescription = Card Namer needs access to removable drives to read and organize card images.`

This is not a full security-scoped bookmark implementation. It is a practical TCC-friendly fix for the current architecture.

## Bundle Sync After Rebuild

`Card Namer.app` is a generated artifact. If Platypus rebuilds it, bundle-local changes can be lost.

Run this after a rebuild:

```bash
zsh "/Users/geoff/code/card_stuff/Card Namer App/sync_card_namer_app.sh"
```

That script currently:

- copies [card_namer_wrapper.sh](/Users/geoff/code/card_stuff/Card%20Namer%20App/card_namer_wrapper.sh) to `Card Namer.app/Contents/Resources/script`
- ensures `CFBundleIdentifier` is `org.geoff.CardNamer`
- ensures `NSRemovableVolumesUsageDescription` is present

## Bundle State

Observed current bundle state:

- Platypus-generated
- not code signed
- uses `/bin/zsh` as the Platypus interpreter
- launches the real app logic through the wrapper script

## Known Constraints

- The app depends on a machine-specific Anaconda interpreter path.
- The app bundle is not self-contained.
- The app bundle is not code signed.
- The current removable-drive solution is not bookmark-based persistent authorization.
- The current Python runtime does not include PyObjC, which blocks a straightforward in-process security-scoped bookmark implementation.

## Likely Future Cleanup

If this app needs to be more portable or robust, the next engineering steps are:

- remove hardcoded absolute paths where possible
- define and document environment/bootstrap setup for the Python runtime
- decide whether to keep Platypus or move to a more self-contained macOS packaging path
- package a runtime that can support a true security-scoped bookmark implementation if removable-drive persistence becomes a recurring problem
