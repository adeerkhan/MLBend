#!/usr/bin/env python3
"""Fix the versions of the sibling imports of the packages.

A package that imports a sibling of this repository by name holds that
name and that version. When the sibling's version moves, the import must
follow it, or the program reads the published sibling rather than the one
beside it.

    scripts/fix_versions.py          # say what would change
    scripts/fix_versions.py --write  # change it

The version of a sibling is the one in its own bend.toml, which is the
version it is released under.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRITE = "--write" in sys.argv

# the packages of this repository: a directory with a bend.toml
def packages():
    out = {}
    for name in sorted(os.listdir(ROOT)):
        manifest = os.path.join(ROOT, name, "bend.toml")
        if not os.path.isfile(manifest):
            continue
        text = open(manifest).read()
        m = re.search(r'^name\s*=\s*"([^"]+)"', text, re.M)
        v = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
        if m and v:
            out[m.group(1)] = (v.group(1), os.path.join(ROOT, name))
    return out


def main() -> int:
    pkgs = packages()
    changed = 0
    for name, (version, directory) in sorted(pkgs.items()):
        for entry in sorted(os.listdir(directory)):
            if not entry.endswith(".bend"):
                continue
            path = os.path.join(directory, entry)
            text = open(path).read()
            new = text
            for other, (other_version, _) in pkgs.items():
                if other == name:
                    continue
                # import other@old/path.bend as Alias
                new = re.sub(
                    r"(import\s+" + re.escape(other) + r")@\d+\.\d+\.\d+\.\d+(/)",
                    r"\1@" + other_version + r"\2",
                    new,
                )
            if new != text:
                changed += 1
                print(f"{os.path.relpath(path, ROOT)}: the sibling versions are fixed")
                if WRITE:
                    open(path, "w").write(new)
    if changed == 0:
        print("every sibling import holds the current version")
    elif not WRITE:
        print(f"{changed} files would change; run it again with --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
