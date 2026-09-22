"""
The dependency lock (2026-09-21).

`requirements*.in` are what a person edits; `requirements*.txt` are GENERATED
by `uv pip compile --universal --python-version 3.12 --generate-hashes` and pin
the whole tree, transitive packages included. These tests hold the structure
that makes the numbers reproducible:

  * every line of every lock is an exact pin carrying a hash, so pip installs
    in hash-checking mode and refuses any package the lock does not name;
  * the serving lock, which Render installs, carries none of the research or
    grading packages;
  * a package shared between two locks has ONE version;
  * the interpreter running this suite IS the locked environment - so a local
    upgrade shows up here before it shows up as a moved result;
  * the stored h=30 baseline reproduces EXACTLY (slow; opt-in, see below).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LOCKS = ("requirements.txt", "requirements-evidence.txt", "requirements-research.txt")
SERVING_MUST_NOT_CARRY = ("arch", "linearmodels", "statsmodels", "torch",
                          "transformers", "pyarrow", "chronos-forecasting")

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)(?:\s*;\s*([^\\]+?))?\s*\\?$")


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path: Path) -> dict[str, dict]:
    """{package: {"version", "marker", "hashes"}} for every pin in a lock."""
    out: dict[str, dict] = {}
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--hash="):
            assert current is not None, f"{path.name}: a hash before any pin"
            out[current]["hashes"] += 1
            continue
        m = _PIN.match(line)
        assert m, f"{path.name}: not an exact pin: {line!r}"
        current = _canon(m.group(1))
        out[current] = {"version": m.group(2), "marker": (m.group(3) or "").strip(),
                        "hashes": 0}
    return out


def _direct_requirements(path: Path) -> set[str]:
    names = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(_canon(re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0]))
    return names


@pytest.mark.parametrize("lock", LOCKS)
def test_every_lock_line_is_an_exact_pin_with_a_hash(lock):
    pins = parse_lock(REPO / lock)
    assert pins, f"{lock} is empty"
    unhashed = [p for p, v in pins.items() if v["hashes"] == 0]
    assert not unhashed, (
        f"{lock}: {unhashed} carry no hash. Regenerate with --generate-hashes: a "
        f"lock without hashes lets pip resolve anything it does not name.")


@pytest.mark.parametrize("lock", LOCKS)
def test_every_input_requirement_is_in_its_lock(lock):
    source = REPO / lock.replace(".txt", ".in")
    missing = _direct_requirements(source) - set(parse_lock(REPO / lock))
    assert not missing, f"{source.name} names {missing}, which {lock} does not pin"


def test_the_serving_lock_carries_nothing_render_does_not_import():
    pins = parse_lock(REPO / "requirements.txt")
    present = [p for p in SERVING_MUST_NOT_CARRY if p in pins]
    assert not present, (
        f"requirements.txt now pins {present}. Render installs that file and "
        f"only that file; the grading and research packages live in their own "
        f"locks (see the torch landmine in CLAUDE.md).")


def test_the_evidence_lock_carries_the_grading_packages():
    pins = parse_lock(REPO / "requirements-evidence.txt")
    assert {"arch", "linearmodels"} <= set(pins)


def test_a_package_shared_between_locks_has_one_version():
    locks = {name: parse_lock(REPO / name) for name in LOCKS}
    seen: dict[str, tuple[str, str]] = {}
    clashes = []
    for name, pins in locks.items():
        for pkg, v in pins.items():
            if pkg in seen and seen[pkg][1] != v["version"]:
                clashes.append(f"{pkg}: {seen[pkg][0]}={seen[pkg][1]}, "
                               f"{name}={v['version']}")
            seen.setdefault(pkg, (name, v["version"]))
    assert not clashes, "; ".join(clashes)


def test_the_running_environment_is_the_locked_one():
    """
    Every locked package that applies here is installed at its locked version,
    checked by the SAME tool the workflows run after installing.

    The serving lock must be complete. The evidence and research locks are
    checked only when their headline package is present, since a serving
    machine legitimately lacks them - but a DIFFERENT version is always a
    failure: that is how a floating transitive dependency moves a result with
    nothing in the repository changing.
    """
    from tools.check_locked_env import check

    paths = [REPO / "requirements.txt"]
    for lock, marker_pkg in (("requirements-evidence.txt", "arch"),
                             ("requirements-research.txt", "pyarrow")):
        try:
            version(marker_pkg)
            paths.append(REPO / lock)
        except PackageNotFoundError:
            pass
    problems = check(paths)
    assert not problems, (
        "this environment is not the locked one: " + "; ".join(problems)
        + ". Reinstall with `pip install -r requirements.txt -r "
          "requirements-evidence.txt -r requirements-research.txt`.")


def test_the_environment_check_bites(tmp_path):
    """A lock pinning a version that is not installed must be reported, or the
    test above proves nothing."""
    from tools.check_locked_env import check

    fake = tmp_path / "fake.txt"
    fake.write_text(f"numpy==0.0.1 \\\n    --hash=sha256:{'0' * 64}\n", encoding="utf-8")
    assert any("numpy" in p and "0.0.1" in p for p in check([fake]))
    absent = tmp_path / "absent.txt"
    absent.write_text("no-such-package-asf==1.0\n", encoding="utf-8")
    assert any("missing" in p for p in check([absent]))


# ── the stored baseline, reproduced exactly ─────────────────────────────────

#: The exact-reproduction checks. Each takes minutes, so they run only when
#: asked: `ASF_REPRO=1 python -m pytest tests/test_dependency_lock.py`.
#: They are REFERENCE-PLATFORM checks: XGBoost's row and column subsampling
#: draws a different sample from the same seed under MSVC's C++ library than
#: under GCC's, so the stored predictions - produced on Windows - cannot be
#: reproduced bit for bit on Linux. See the platform landmine in CLAUDE.md.
REPRO_CASES = [
    ("pre-fix panel, hygiene baseline", "panel_cache.parquet",
     "hygiene_repin_oos.npz", "new_pinned"),
    ("calendar-clean panel, the post-fix baseline", "panel_cache_clean.parquet",
     "baseline_clean_oos.npz", "new_pinned"),
]


@pytest.mark.parametrize("label,panel,npz,arm", REPRO_CASES,
                         ids=[c[0] for c in REPRO_CASES])
def test_the_stored_baseline_reproduces_exactly(label, panel, npz, arm):
    if os.environ.get("ASF_REPRO") != "1":
        pytest.skip("slow (~6 min each); set ASF_REPRO=1 to run")
    if sys.platform != "win32":
        pytest.skip("the stored predictions were produced on Windows; see the "
                    "platform landmine")
    for f in (panel, npz):
        if not (REPO / f).exists():
            pytest.skip(f"{f} is not present in this checkout (gitignored)")
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "baseline_repro.py"),
         "--panel-cache", str(REPO / panel), "--against", str(REPO / npz),
         "--arm", arm],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0 and "EXACT REPRODUCTION: YES" in r.stdout, (
        f"{label} did not reproduce exactly:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
