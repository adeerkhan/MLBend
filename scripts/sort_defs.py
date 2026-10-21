#!/usr/bin/env python3
"""Sort the defs of a Bend file so that every def comes after the defs it
uses.

Bend has no forward references: a def must be defined before it is used,
and a name that is used before it is defined reads as an unfilled law, so
the checker refuses it. Writing the file in dependency order by hand is
tedious and easy to get wrong, so this does it.

    scripts/sort_defs.py <file.bend>...       say what would change
    scripts/sort_defs.py --write <file.bend>...  change it

A name used before it is defined is a *call* to it: `f(x)`, `f x` or a
function passed by name, `f,` — but not `x.f(x)`, which is a method on
something else.
"""
import os
import re
import sys

DEF = re.compile(r"^def ([A-Za-z_][\w]*)\(")
TYPE = re.compile(r"^type ([A-Za-z_][\w]*)")
LAW = re.compile(r"^law ([A-Za-z_][\w]*)")
CALL = re.compile(r"(?<![.\w])([a-z][\w]*)\s*(\(|,)")


def blocks(lines):
    """the file's pieces: the head, then the types and laws, then the defs"""
    head, kinds = [], []
    pending = []
    i = 0
    while i < len(lines):
        m = DEF.match(lines[i]) or TYPE.match(lines[i]) or LAW.match(lines[i])
        if m is None:
            head.append(lines[i])
            i += 1
            continue
        kind = "def" if DEF.match(lines[i]) else ("type" if TYPE.match(lines[i]) else "law")
        body = pending + [lines[i]]
        pending = []
        j = i + 1
        while j < len(lines) and not (DEF.match(lines[j]) or TYPE.match(lines[j]) or LAW.match(lines[j])):
            body.append(lines[j])
            j += 1
        # the comments at the end of a block belong to the next one
        while body and (not body[-1].strip() or body[-1].lstrip().startswith("#")):
            pending.insert(0, body.pop())
        kinds.append((kind, m.group(1), body))
        i = j
    if pending:
        head.extend(pending)
    return head, kinds


def deps_of(body, names):
    """the local names a body calls"""
    text = "\n".join(re.sub(r"#.*$", "", l) for l in body)
    return {n for n in names if re.search(CALL.pattern.replace("(", r"\s*\("), text) and n != n}


def sort_file(path, write):
    lines = open(path).read().split("\n")
    head, kinds = blocks(lines)
    defs = [(n, b) for k, n, b in kinds if k == "def"]
    others = [(k, n, b) for k, n, b in kinds if k != "def"]
    names = [n for n, _ in defs]

    deps = {}
    for n, b in defs:
        text = "\n".join(re.sub(r"#.*$", "", l) for l in b)
        deps[n] = {m for m in names if m != n and re.search(r"(?<![.\w])" + re.escape(m) + r"\s*(\(|,)", text)}

    order, seen, stack = [], set(), set()

    def visit(n):
        if n in seen:
            return
        if n in stack:
            raise SystemExit(f"{path}: a cycle at {n}")
        stack.add(n)
        for d in sorted(deps[n]):
            visit(d)
        stack.discard(n)
        seen.add(n)
        order.append(n)

    for n in sorted(deps):
        visit(n)

    out = list(head)
    for k, n, b in others:
        out.extend(b)
    for n in order:
        for name, b in defs:
            if name == n:
                out.extend(b)
                break
    new = "\n".join(out)
    old = "\n".join(lines)
    if new == old:
        print(f"{path}: the defs are in order")
        return
    print(f"{path}: the defs are sorted")
    if write:
        open(path, "w").write(new)


def main() -> int:
    argv = sys.argv[1:]
    write = "--write" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print(__doc__)
        return 1
    for path in paths:
        sort_file(path, write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
