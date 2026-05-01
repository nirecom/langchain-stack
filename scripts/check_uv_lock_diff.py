"""Verify uv.lock regeneration did not add, remove, or change package versions.

Wheel and sdist binary selections legitimately differ when the Python range
changes (e.g. >=3.12 → >=3.12,<3.13), so only `name@version` pairs are
compared — not hashes, wheels, sources, or markers.

Compares uv.lock.bak vs uv.lock; non-zero exit if any package was
added, removed, or upgraded/downgraded.

Usage:
    uv run python scripts/check_uv_lock_diff.py
"""
import sys
import tomllib
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"


def name_version_set(p: Path) -> set[str]:
    """Return {name@version, ...} for every package in the lock file."""
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    return {f"{pkg['name']}@{pkg['version']}" for pkg in data.get("package", [])}


def main() -> None:
    bak = APP / "uv.lock.bak"
    cur = APP / "uv.lock"
    if not bak.exists():
        print(f"SKIP: {bak} not found — nothing to compare.")
        return

    old = name_version_set(bak)
    new = name_version_set(cur)
    added = sorted(set(new) - old)
    removed = sorted(old - set(new))

    if added or removed:
        print("FAIL: package set changed during lock regeneration.")
        print(f"  added:   {added}")
        print(f"  removed: {removed}")
        print("If these are intentional, review each change and re-run after deleting uv.lock.bak.")
        sys.exit(1)
    print(f"OK: {len(new)} packages unchanged (name@version). "
          f"Binary wheel/marker changes are expected when Python range narrows.")


if __name__ == "__main__":
    main()
