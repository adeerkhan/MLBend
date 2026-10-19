"""Prints every law statement of every package: the specification, in one place.

  reference/.venv/bin/python reference/list_laws.py
  reference/.venv/bin/python reference/list_laws.py --check-readme
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def packages():
    return sorted(d for d in os.listdir(ROOT)
                  if os.path.isdir(os.path.join(ROOT, d)) and os.path.exists(os.path.join(ROOT, d, "main.bend")))


def laws(text):
    """The law statements of a file: each 'law' block, up to its proof."""
    out = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("law "):
            block = [lines[i]]
            i += 1
            while i < len(lines) and not lines[i].startswith("def "):
                block.append(lines[i])
                i += 1
            out.append("\n".join(block))
        else:
            i += 1
    return out


def main():
    check_readme = "--check-readme" in sys.argv
    missing = []
    for pkg in packages():
        f = os.path.join(ROOT, pkg, "main.bend")
        blocks = laws(open(f).read())
        print(f"\n=== {pkg} ({len(blocks)} laws)")
        for b in blocks:
            print(b)
            name = b.split(":")[0].split()[1]
            readme = os.path.join(ROOT, pkg, "README.md")
            if check_readme and os.path.exists(readme):
                if name not in open(readme).read():
                    missing.append(f"{pkg}: {name} is not in the package README")
    if missing:
        for m in missing:
            print("ERROR " + m)
        sys.exit(1)
    if check_readme:
        print("\nok   every law is in its package README")


if __name__ == "__main__":
    main()
