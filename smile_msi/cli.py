"""SMILE MSI entry point — launches the desktop application.

    smile-msi                 open the app
    smile-msi data.imzML      open the app and load this dataset
    smile-msi chat            plain-language analysis chat in your browser (see --help)

The app is the whole interface; analysis is interactive (no command-line pipeline).
"""
from __future__ import annotations

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["chat"]:
        from .agent.server import main as chat_main
        return chat_main(argv[1:])
    open_path = None
    for a in argv:
        if not a.startswith("-") and a.lower().endswith(".imzml"):
            open_path = a
    try:
        from .gui import run
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"GUI dependencies missing ({exc}).\n"
                 "Install them with:  uv pip install -e '.[gui]'")
    return run(open_path=open_path)


if __name__ == "__main__":
    main()
