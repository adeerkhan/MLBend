#!/usr/bin/env python3
"""Mark the parameters of a Bend def as reusable when the body reads them twice.

Bend consumes a value once unless it is bound with a `+`. A parameter that
the body reads in two branches has to be reusable, and the error the
checker gives ("consumed more than once") costs a round trip per def.

    scripts/reuse.py file.bend          # rewrite the file
    scripts/reuse.py --check file.bend  # only say what would change
"""

import re
import sys
from pathlib import Path

DEF = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_.]*)\s*\(", re.M)


def params_of(text: str, m):
    """The parameter list of a def, and the text that follows it."""
    i = m.end() - 1
    depth = 0
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[m.end() : i], i


def body_after(text: str, end: int) -> str:
    """The body of a def: up to the next def, type or law."""
    rest = text[end + 1 :]
    nxt = re.search(r"^(def|type|law)\b", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def split_params(params: str):
    """The parameters of a def, one by one."""
    out, depth, cur = [], 0, ""
    for ch in params:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def name_of(param: str):
    """The name of a parameter, or None when it is not a plain name."""
    p = param.strip().lstrip("+")
    p = p.split(":")[0].strip()
    return p if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) else None


def strip_strings(text: str) -> str:
    return re.sub(r'"[^"]*"', '""', text)


def rewrite_params(params: str, body: str) -> str:
    """The parameters of a def, with the twice-read ones made reusable."""
    parts = split_params(params)
    for k, part in enumerate(parts):
        nm = name_of(part)
        if nm is None or part.strip().startswith("+"):
            continue
        reads = len(re.findall(r"\b%s\b" % re.escape(nm), strip_strings(body)))
        if reads > 1:
            lead = part[: len(part) - len(part.lstrip())]
            parts[k] = lead + "+" + part.lstrip()
    return ",".join(parts)


CASE = re.compile(r"^(\s*)case\s+([A-Za-z0-9_.]*)\{([^}]*)\}:\s*$", re.M)


def rewrite_cases(text: str) -> str:
    """Mark the binders of a case as reusable when the arm reads them twice."""
    lines = text.split("\n")
    out = []
    for n, line in enumerate(lines):
        m = CASE.match(line)
        if not m:
            out.append(line)
            continue
        indent = len(m.group(1))
        binders = []
        for part in m.group(3).split(","):
            nm = name_of(part)
            if nm is not None:
                binders.append((nm, part.strip()))
        if not binders:
            out.append(line)
            continue
        # the arm: the lines under the case, until one comes back
        arm = []
        for line2 in lines[n + 1 :]:
            if line2.strip() and (len(line2) - len(line2.lstrip())) <= indent:
                break
            arm.append(line2)
        block = strip_strings("\n".join(arm))
        changed = []
        for nm, part in binders:
            reads = len(re.findall(r"\b%s\b" % re.escape(nm), block))
            if reads > 1 and not part.startswith("+"):
                changed.append("+" + part)
            else:
                changed.append(part)
        out.append(m.group(1) + "case " + m.group(2) + "{" + ", ".join(changed) + "}:")
    return "\n".join(out)


BIND = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)(.*)$")


def rewrite_binds(text: str) -> str:
    """Mark a bind as reusable when the body reads it twice."""
    out = []
    pos = 0
    for m in DEF.finditer(text):
        params, i = params_of(text, m)
        body = body_after(text, i)
        out.append(text[pos : m.end()])
        out.append(rewrite_params(params, body))
        out.append(text[i : i + 1])
        out.append(rewrite_binds_of(body))
        pos = i + 1 + len(body)
    out.append(text[pos:])
    return "".join(out)


def rewrite_binds_of(body: str) -> str:
    """The body of a def, with the twice-read binds made reusable."""
    lines = body.split("\n")
    binds = {}
    for n, line in enumerate(lines):
        m = BIND.match(line)
        if m:
            binds.setdefault(m.group(2), []).append(n)
    if not binds:
        return body
    plain = strip_strings(body)
    for nm, ns in binds.items():
        total = len(re.findall(r"\b%s\b" % re.escape(nm), plain))
        if total - len(ns) > 1:
            for n in ns:
                m = BIND.match(lines[n])
                lines[n] = m.group(1) + "+" + m.group(2) + " =" + m.group(3)
    return "\n".join(lines)


def rewrite(text: str) -> str:
    out = []
    pos = 0
    for m in DEF.finditer(text):
        params, i = params_of(text, m)
        body = body_after(text, i)
        out.append(text[pos : m.end()])
        out.append(rewrite_params(params, body))
        out.append(text[i : i + 1])
        out.append(body)
        pos = i + 1 + len(body)
    out.append(text[pos:])
    return rewrite_cases(rewrite_binds("".join(out)))


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    check = "--check" in argv
    files = [a for a in argv[1:] if not a.startswith("-")]
    for f in files:
        p = Path(f)
        text = p.read_text()
        new = rewrite(text)
        if new == text:
            print(f"{f}: nothing to reuse")
            continue
        if check:
            print(f"{f}: would change")
        else:
            p.write_text(new)
            print(f"{f}: the parameters are reusable")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
