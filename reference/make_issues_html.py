"""Render .local/issues-hit.md as .local/issues-hit.html, in the bend-lang.com theme.

  reference/.venv/bin/python reference/make_issues_html.py

One source of truth: the markdown log stays the thing that is edited, and this
script turns it into a page with the palette of https://bend-lang.com (`--bg`
#f2eee7, `--line` #e4dfd5, `--b01` #69665f, and the four accents), the same
monospace stack, and a thin rule per section.

Each issue carries the date in its own text when it has one; the ledger is
local and untracked, so git history is not used. The footer records when the
page was generated. Nothing here is a gate; `make check` does not run it.
"""
import html
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, ".local", "issues-hit.md")
OUT = os.path.join(ROOT, ".local", "issues-hit.html")

STATUS_TONE = {
    "FIXED": "grn",
    "OPEN": "ora",
    "UPSTREAM": "vio",
    "WORKED AROUND": "b0",
    "DISMISSED": "b0",
}


def first_seen(heading, body=""):
    """The date the entry itself gives, or nothing."""
    m = re.search(r"(20[0-9]{2}-[0-9]{2}-[0-9]{2})", body)
    return m.group(1) if m else ""


def inline(text):
    text = text.replace("\\|", "|")
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def blocks(lines):
    """A small markdown subset: paragraphs, bullets, tables, fences."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.strip() == "---":
            i += 1
            continue
        if line.lstrip().startswith("```"):
            lang = line[3:].strip()
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append(f'<pre class="code{" " + lang if lang else ""}"><code>'
                       + html.escape("\n".join(code)) + "</code></pre>")
            continue
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().lstrip("|").rstrip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            head, body = rows[0], rows[1:]
            t = ['<div class="tablewrap"><table><thead><tr>']
            t += [f"<th>{inline(c)}</th>" for c in head]
            t.append("</tr></thead><tbody>")
            for r in body:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            continue
        if line.lstrip().startswith("- "):
            items = []
            while i < len(lines) and (lines[i].lstrip().startswith("- ") or lines[i].startswith("  ")):
                if lines[i].lstrip().startswith("- "):
                    items.append([lines[i].lstrip()[2:]])
                elif items:
                    items[-1].append(lines[i].strip())
                i += 1
            out.append("<ul>" + "".join(
                "<li>" + inline(" ".join(it)) + "</li>" for it in items) + "</ul>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and lines[i].strip() != "---" \
                and not lines[i].lstrip().startswith(("|", "- ", "```", "#")):
            para.append(lines[i].strip())
            i += 1
        out.append("<p>" + inline(" ".join(para)) + "</p>")
    return "\n".join(out)


def main():
    src = open(SRC).read()
    chunks = re.split(r"^## ", src, flags=re.M)
    intro = chunks[0]
    sections = ["## " + c for c in chunks[1:]]

    title = "issues hit"
    m = re.search(r"^# (.+)$", intro, flags=re.M)
    if m:
        title = m.group(1).strip()
        intro = intro.replace(m.group(0), "", 1)

    cards, counts = [], {"FIXED": 0, "OPEN": 0, "UPSTREAM": 0, "other": 0}
    for sec in sections:
        head, _, body = sec.partition("\n")
        heading = head[3:].strip()
        parts = [p.strip() for p in heading.split("—")]
        name = parts[0]
        status = parts[-1].upper() if parts[-1].upper() in STATUS_TONE else ""
        desc = " — ".join(parts[1:-1]) if status else " — ".join(parts[1:])
        tone = STATUS_TONE.get(status, "b0")
        counts["FIXED" if status == "FIXED" else
               "UPSTREAM" if status == "UPSTREAM" else
               "OPEN" if status == "OPEN" else "other"] += 1
        seen = first_seen(heading, body)
        seen_html = ""
        if seen:
            if len(seen) == 10:
                seen_html = f'<span class="when">logged {seen}</span>'
            else:
                try:
                    dt = datetime.fromisoformat(seen)
                    seen_html = f'<span class="when" title="{seen}">logged {dt:%Y-%m-%d %H:%M} UTC</span>'
                except ValueError:
                    seen_html = f'<span class="when">{html.escape(seen)}</span>'
        cards.append(f'''<section class="issue" id="{re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")}">
  <header class="ihead">
    <span class="id">{inline(name)}</span>
    {f'<span class="badge {tone}">{status.lower()}</span>' if status else ''}
    {seen_html}
  </header>
  {f'<h2>{inline(desc)}</h2>' if desc else ''}
  {blocks(body.splitlines())}
</section>''')

    now = datetime.now(timezone.utc)
    page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — bend-lab</title>
<style>
:root {{
  --bg:#f2eee7; --bg2:#e9e4db; --line:#e4dfd5;
  --b1:#a9a59d; --b0:#87847d; --b01:#69665f; --b02:#4d4a44;
  --ora:#c4845c; --red:#c46a60; --vio:#8b83b5; --grn:#7e9a5e;
}}
*{{box-sizing:border-box}}
html{{background:var(--bg);color:var(--b01);
     font:15px/1.6 ui-monospace,Menlo,"SF Mono",Consolas,"Liberation Mono",monospace;
     -webkit-font-smoothing:antialiased}}
body{{max-width:92ch;margin:0 auto;padding:0 2ch 6ch}}
a{{color:var(--vio);text-decoration:none;border-bottom:1px solid var(--line)}}
a:hover{{color:var(--ora);border-bottom-color:var(--ora)}}
h1{{font-size:1.5rem;font-weight:600;color:var(--b02);margin:2.4rem 0 .2rem;letter-spacing:-.01em}}
h2{{font-size:1rem;font-weight:600;color:var(--b02);margin:.2rem 0 .8rem}}
p,li{{color:var(--b01)}}
code{{color:var(--b02);background:var(--bg2);padding:.05em .35ch;border-radius:4px;font-size:.94em}}
pre.code{{background:var(--bg2);border:1px solid var(--line);border-radius:6px;
          padding:.9em 1.2ch;overflow-x:auto;margin:1em 0}}
pre.code code{{background:none;padding:0;font-size:.9em;line-height:1.5}}
header.top{{display:flex;justify-content:space-between;align-items:baseline;
            padding:1.6rem 0 .6rem;border-bottom:1px solid var(--line)}}
header.top .home{{color:var(--b02);font-weight:600}}
header.top .meta{{color:var(--b0);font-size:.82rem}}
.lede{{color:var(--b01);max-width:78ch;margin:1.2rem 0 0}}
.counts{{display:flex;gap:1.2rem;flex-wrap:wrap;margin:1.4rem 0 2.4rem;
         color:var(--b0);font-size:.85rem}}
.counts b{{color:var(--b02);font-weight:600}}
section.issue{{border-top:1px solid var(--line);padding:1.6rem 0 1rem}}
.ihead{{display:flex;gap:.9rem;align-items:baseline;flex-wrap:wrap;
        font-size:.8rem;color:var(--b0);letter-spacing:.02em}}
.id{{color:var(--b02);font-weight:600;font-size:.86rem}}
.badge{{border-radius:999px;padding:.08em 1.1ch;font-size:.72rem;
        text-transform:uppercase;letter-spacing:.08em;border:1px solid currentColor}}
.badge.grn{{color:var(--grn)}} .badge.ora{{color:var(--ora)}}
.badge.vio{{color:var(--vio)}} .badge.b0{{color:var(--b0)}}
.when{{color:var(--b1)}}
ul{{padding-left:2.2ch;margin:.6em 0}}
li{{margin:.2em 0}}
.tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:6px;margin:1em 0}}
table{{border-collapse:collapse;width:100%;font-size:.88rem}}
th,td{{text-align:left;padding:.45em 1.2ch;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{color:var(--b02);font-weight:600;background:var(--bg2)}}
tr:last-child td{{border-bottom:0}}
footer{{border-top:1px solid var(--line);margin-top:3rem;padding-top:1rem;
        color:var(--b0);font-size:.8rem;display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap}}
</style>
</head>
<body>
<header class="top">
  <a class="home" href="https://bend-lang.com/">~/bend</a>
  <span class="meta">MLBend · diffusion · issue ledger</span>
</header>
<h1>{html.escape(title)}</h1>
<div class="lede">{blocks(intro.splitlines())}</div>
<div class="counts">
  <span><b>{counts["FIXED"]}</b> fixed</span>
  <span><b>{counts["OPEN"]}</b> open</span>
  <span><b>{counts["UPSTREAM"]}</b> upstream</span>
  <span><b>{counts["other"]}</b> other</span>
</div>
{"".join(cards)}
<footer>
  <span>generated from the local ledger at {now:%Y-%m-%d %H:%M} UTC</span>
  <span>theme: bend-lang.com</span>
</footer>
</body>
</html>
'''
    open(OUT, "w").write(page)
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({len(page)} bytes, "
          f"{len(cards)} entries: {counts['FIXED']} fixed, {counts['OPEN']} open, "
          f"{counts['UPSTREAM']} upstream, {counts['other']} other)")


if __name__ == "__main__":
    main()
