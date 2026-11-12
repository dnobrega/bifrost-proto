"""Discover a Bifrost run root from the ``snapname`` parameter in IDL files.

This module is private to the constructor. No snapshot arrays are read.
"""

import re
from collections import Counter
from pathlib import Path

_SNAPNAME = re.compile(
    r"^\s*snapname\s*=\s*(?:([\"'])(.*?)\1|([^\s;#]+))",
    re.IGNORECASE,
)


def _snapname_from_idl(path):
    """Read only as far as the first ``snapname`` entry in one IDL file."""
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = _SNAPNAME.match(line)
            if match:
                value = match.group(2) if match.group(1) else match.group(3)
                return value.strip() or None
    return None


def detect_snapname(folder):
    """Find one run root in local ``*.idl`` files, or ask which one to use.

    Multiple runs are listed by decreasing number of matching IDL files.
    An explicit ``file_root``/``idlparam`` bypasses this detection entirely.
    """
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError(f"Simulation folder does not exist: {folder}")

    counts = Counter()
    nfiles = 0
    for path in folder.glob("*.idl"):
        if not path.is_file():
            continue
        nfiles += 1
        name = _snapname_from_idl(path)
        if name:
            counts[name] += 1

    if not counts:
        raise ValueError(
            f"No snapname found in {nfiles} *.idl files in {folder}. "
            "Specify file_root (or idlparam) explicitly."
        )
    if len(counts) == 1:
        return next(iter(counts))

    choices = sorted(counts, key=lambda name: (-counts[name], name))
    print(f"Multiple Bifrost simulations found in {folder}:")
    for i, name in enumerate(choices, start=1):
        print(f"  [{i}] {name} ({counts[name]} .idl files)")

    while True:
        try:
            choice = input(f"Select simulation [1-{len(choices)}]: ").strip()
        except (EOFError, OSError) as exc:
            raise ValueError(
                "Multiple snapnames found but interactive selection is "
                f"unavailable. Specify file_root or idlparam explicitly: {choices}"
            ) from exc
        except KeyboardInterrupt as exc:
            raise ValueError(
                "Simulation selection cancelled; specify file_root explicitly"
            ) from exc

        if choice in counts:
            return choice
        if choice.isdecimal() and 1 <= int(choice) <= len(choices):
            return choices[int(choice) - 1]
        print("Invalid selection; enter a number or the exact snapname.")
