"""
tools/check_locked_env.py — is the running interpreter exactly the lock?

    python tools/check_locked_env.py requirements.txt [requirements-evidence.txt ...]

Exits 1, naming each package, when an installed version differs from the pin
in any lock file given, or when a pinned package that applies to this platform
is missing. Run by the workflows right after installing, and suggested as the
tail of Render's build command, because a hash-checked `pip install` guarantees
what it installed and says nothing about what a LATER install moved: the
weekly job installs the unpinned torch/transformers scorer too.

Standard library plus `packaging`, which the serving lock pins and pip vendors.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)(?:\s*;\s*([^\\]+?))?\s*\\?$")


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(path: Path) -> dict[str, tuple[str, str]]:
    """{package: (version, marker)} for every exact pin in a uv/pip-compile lock."""
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "--")):
            continue
        m = _PIN.match(line)
        if not m:
            raise SystemExit(f"{path}: not an exact pin: {line!r}")
        out[canon(m.group(1))] = (m.group(2), (m.group(3) or "").strip())
    return out


def applies(marker: str) -> bool:
    if not marker:
        return True
    try:
        from packaging.markers import Marker
    except ImportError:                                         # pragma: no cover
        from pip._vendor.packaging.markers import Marker
    return Marker(marker).evaluate()


def check(paths: list[Path]) -> list[str]:
    problems = []
    for path in paths:
        for pkg, (want, marker) in pins(path).items():
            if not applies(marker):
                continue
            try:
                have = version(pkg)
            except PackageNotFoundError:
                problems.append(f"{pkg}: missing (locked {want} in {path.name})")
                continue
            if have != want:
                problems.append(f"{pkg}: {have} installed, locked {want} in {path.name}")
    return problems


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    paths = [Path(a) for a in argv]
    problems = check(paths)
    if problems:
        print("THE ENVIRONMENT IS NOT THE LOCK:\n  " + "\n  ".join(problems))
        return 1
    n = sum(len(pins(p)) for p in paths)
    print(f"environment matches {', '.join(a.name for a in paths)} ({n} pins)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
